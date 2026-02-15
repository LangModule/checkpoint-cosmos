"""Asynchronous LangGraph Checkpoint Saver for Azure Cosmos DB.

This module provides an async checkpoint saver implementation that persists
LangGraph workflow state to Azure Cosmos DB. For synchronous support, see the
main `langgraph_checkpoint_cosmos` module.
"""
from __future__ import annotations

import asyncio
import base64
import random
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Optional, Union

from azure.cosmos.aio import CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosBatchOperationError
from azure.identity.aio import DefaultAzureCredential

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

from .utils import (
    search_where,
    prepare_batch_items,
    _make_cosmosdb_checkpoint_key,
    _make_cosmosdb_checkpoint_writes_key,
    _make_cosmosdb_writes_partition_key,
    _make_cosmosdb_tip_key,
)

# Separator used in composite document IDs (e.g., "writes$thread_id$ns$checkpoint_id$task_id$idx").
COSMOSDB_KEY_SEPARATOR = "$"

class AsyncCosmosDBSaver(BaseCheckpointSaver[str]):
    """An asynchronous checkpoint saver that stores checkpoints in Azure Cosmos DB.

    This class provides a durable, scalable backend for storing LangGraph checkpoints.
    It leverages Cosmos DB's partitioning and atomic batch capabilities to ensure
    consistency and performance.

    ### Key Architectural Decisions

    1.  **Tip Document Optimization**:
        To efficiently retrieve the latest checkpoint (O(1) access), we maintain a separate 
        "Tip Document" (`type='tip'`) for each thread/namespace. This document always 
        points to the ID of the most recent checkpoint, avoiding the need for expensive 
        sorted queries (SCAN) or composite indexes to find the latest state.

    2.  **Partitioning Strategy**:
        Data is sharded by logical partitions to optimize for read/write performance:
        *   **Checkpoints**: Stored with partition key `checkpoint${thread_id}${checkpoint_ns}`.
            This groups all checkpoints for a single thread together.
        *   **Writes**: Stored with partition key `writes${thread_id}${checkpoint_ns}`.
            This isolates potentially large write histories from checkpoint metadata.

    3.  **Atomic Consistency**:
        We use Cosmos DB's transactional batch (`execute_item_batch`) to ensure that
        saving a checkpoint and updating the Tip Document happens atomically. This guarantees 
        that the "latest" pointer never references a missing or partial checkpoint.

    Note:
        This class is designed for high-performance, asynchronous applications (e.g., FastAPI)
        and is the recommended saver for production workloads.

    Args:
        container (azure.cosmos.aio.ContainerProxy): The Cosmos DB container client.
        serde (Optional[SerializerProtocol]): The serializer to use. Defaults to JsonPlusSerializer.

    Attributes:
        container (Any): The Cosmos DB container client.
    Examples:
        Usage within an async StateGraph:

        ```python
        import asyncio
        from azure.cosmos.aio import CosmosClient
        from langgraph_checkpoint_cosmos.aio import AsyncCosmosDBSaver
        from langgraph.graph import StateGraph

        async def main():
            # 1. Connect to Cosmos DB
            url = "https://<your-account>.documents.azure.com:443/"
            key = "<your-key>"
            
            async with CosmosClient(url, credential=key) as client:
                database = client.get_database_client("my-database")
                container = database.get_container_client("my-container")
                
                # 2. Create the saver
                memory = AsyncCosmosDBSaver(container)
                
                # 3. Use with StateGraph
                builder = StateGraph(int)
                builder.add_node("add_one", lambda x: x + 1)
                builder.set_entry_point("add_one")
                builder.set_finish_point("add_one")
                
                graph = builder.compile(checkpointer=memory)
                config = {"configurable": {"thread_id": "thread-1"}}
                
                # Run the graph
                result = await graph.ainvoke(1, config)
                print(result) # Output: 2

        if __name__ == "__main__":
            asyncio.run(main())
        ```
    """

    container: Any
    client: CosmosClient
    database: Any

    def __init__(
        self,
        container: Any,
        *,
        serde: SerializerProtocol | None = None,
    ):
        """Initialize the AsyncCosmosDBSaver.

        Args:
            container (azure.cosmos.aio.ContainerProxy): A Cosmos DB async container client
                obtained from `database.get_container_client()`.
            serde (Optional[SerializerProtocol]): Custom serializer for checkpoint data.
                Defaults to LangGraph's JsonPlusSerializer if not provided.
        """
        super().__init__(serde=serde)
        self.container = container
        self.loop = asyncio.get_running_loop()

    @classmethod
    @asynccontextmanager
    async def from_conn_info(
        cls, 
        *, 
        endpoint: str, 
        database_name: str, 
        container_name: str,
        credential: Optional[Union[str, Any]] = None
    ) -> AsyncIterator[AsyncCosmosDBSaver]:
        """Create a new AsyncCosmosDBSaver instance from connection information.

        This context manager handles the lifecycle of the Cosmos Client, ensuring proper 
        cleanup of resources (e.g., closing the client) when the context exits.

        Args:
            endpoint (str): The Cosmos DB account endpoint URL.
            database_name (str): The name of the database.
            container_name (str): The name of the container.
            credential (Optional[Union[str, Any]]): The account key or a credential object (e.g., DefaultAzureCredential).
                If None, DefaultAzureCredential is used for secure, keyless authentication.

        Yields:
            AsyncIterator[AsyncCosmosDBSaver]: A managed instance of AsyncCosmosDBSaver.
        """
        created_credential = False
        if not credential:
            credential = DefaultAzureCredential()
            created_credential = True
            
        client = CosmosClient(endpoint, credential=credential)
        try:
            database = client.get_database_client(database_name)
            # Ensure container exists with correct partition key
            # We follow the existing pattern of /partition_key
            container = database.get_container_client(container_name)
            yield cls(container)
        finally:
            await client.close()
            if created_credential and hasattr(credential, "close"):
                await credential.close()

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database.

        This method is a synchronous bridge to the asynchronous `aget_tuple` method.
        It is provided for compatibility with the synchronous `BaseCheckpointSaver` interface.

        Warning:
            This method blocks the current thread until the async operation completes.
            It should only be called from a background thread to avoid blocking the
            event loop.

        Args:
            config (RunnableConfig): Configuration containing `thread_id` and optional `checkpoint_ns`.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if not found.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            # we don't check in other methods to avoid the overhead
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncCosmosDBSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.aget_tuple(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint and its associated writes. It employs the 
        "Tip Document" optimization to fetch the latest state in O(1) time without 
        performing a query scan.

        Algorithm:
        1.  If `checkpoint_id` is provided, fetch that specific document by ID.
        2.  If not provided (latest), first fetch the "Tip Document" for the thread.
        3.  Use the `checkpoint_id` from the Tip to fetch the actual checkpoint document.
        4.  Fetch all associated writes by querying the `writes` partition.

        Args:
            config (RunnableConfig): Configuration containing `thread_id` and optional `checkpoint_ns`.

        Returns:
            Optional[CheckpointTuple]: The checkpoint tuple if found, otherwise None.
        """
        # Tip Document optimization: fixed ID ('tip$...') allows O(1) access
        # to the latest checkpoint ID, avoiding expensive query scans.
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        partition_key = _make_cosmosdb_checkpoint_key(thread_id, checkpoint_ns, '')

        if checkpoint_id:
            # Direct Access: If the checkpoint ID is explicitly provided in the config,
            # we fetch the specific document using its unique ID within the thread partition.
            key = _make_cosmosdb_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
            try:
                item = await self.container.read_item(item=key, partition_key=partition_key)
            except CosmosHttpResponseError as e:
                # 404 indicates the specific checkpoint doesn't exist, which is a valid result.
                if e.status_code == 404:
                    return None
                raise
        else:
             # Optimization: Use 'tip' document as a head pointer to keep lookups O(1)
             # and avoid database scanners or composite index requirements.
             tip_key = _make_cosmosdb_tip_key(thread_id, checkpoint_ns)
             try:
                 tip_item = await self.container.read_item(item=tip_key, partition_key=partition_key)
                 latest_checkpoint_id = tip_item["checkpoint_id"]
             except CosmosHttpResponseError as e:
                 # If the Tip is missing, the thread has no history yet.
                 if e.status_code == 404:
                     return None
                 raise
                 
             # Once we have the latest ID from the pointer, we perform a direct second read
             # to fetch the actual checkpoint content.
             key = _make_cosmosdb_checkpoint_key(thread_id, checkpoint_ns, latest_checkpoint_id)
             try:
                 item = await self.container.read_item(item=key, partition_key=partition_key)
             except CosmosHttpResponseError as e:
                 # If Tip exists but checkpoint is missing, atomic batch likely failed or data was moved.
                 if e.status_code == 404:
                     return None
                 raise

        # Deserialize checkpoint from Base64-encoded blob.
        c_type = item.get("type", "")
        c_bytes = base64.b64decode(item["checkpoint"])
        checkpoint = self.serde.loads_typed((c_type, c_bytes))
        
        # Metadata Recovery Strategy:
        # 1. Prefer `metadata_blob` (Base64) for high-fidelity recovery of complex types.
        # 2. Fall back to `metadata` (JSON) for backward compatibility or simple dicts.
        metadata_val = item.get("metadata_blob")
        if metadata_val and isinstance(metadata_val, str):
            m_type = item.get("metadata_type")
            m_bytes = base64.b64decode(metadata_val)
            if m_type:
                metadata = self.serde.loads_typed((m_type, m_bytes))
            else:
                metadata = item.get("metadata", {})
        else:
            metadata = item.get("metadata", {})
            if isinstance(metadata, str):
                metadata = {}
        
        parent_checkpoint_id = item.get("parent_checkpoint_id")
        parent_config = (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }
            if parent_checkpoint_id
            else None
        )

        # Helper: Fetch Writes
        # Writes are stored in a separate logical partition (`writes$...`) to avoid 
        # performance hotspots in the checkpoint partition as history grows.
        writes_pk = _make_cosmosdb_writes_partition_key(thread_id, checkpoint_ns)
        query_writes = "SELECT * FROM c WHERE c.partition_key=@pk AND c.checkpoint_id=@cp_id"
        params_writes = [
            {"name": "@pk", "value": writes_pk},
            {"name": "@cp_id", "value": item["checkpoint_id"]}
        ]
        
        parsed_writes = []
        async for write_item in self.container.query_items(query=query_writes, parameters=params_writes):
             # Parse write key to get task_id and idx
             # Key format: writes$thread_id$ns$checkpoint_id$task_id$idx
             key_parts = write_item["id"].split(COSMOSDB_KEY_SEPARATOR)
             task_id = key_parts[4]
             
             w_type = write_item["type"]
             w_bytes = base64.b64decode(write_item["value"])
             value = self.serde.loads_typed((w_type, w_bytes))
             channel = write_item["channel"]
             
             parsed_writes.append((task_id, channel, value))

        # Sort in-memory
        parsed_writes.sort(key=lambda x: (x[0], WRITES_IDX_MAP.get(x[1], 0)))
             
        return CheckpointTuple(
            config,
            checkpoint,
            metadata,
            parent_config,
            parsed_writes
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method is a synchronous bridge to the asynchronous `alist` method.
        It converts the asynchronous iterator into a synchronous one.

        Warning:
            This method blocks the current thread during iteration. It should only be
            called from a background thread to avoid blocking the event loop.

        Args:
            config (RunnableConfig): Base configuration for filtering checkpoints.
            filter (Optional[dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified ID are returned.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            Iterator[CheckpointTuple]: An iterator of matching checkpoint tuples.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            # we don't check in other methods to avoid the overhead
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncCosmosDBSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `checkpointer.alist(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield asyncio.run_coroutine_threadsafe(
                    anext(aiter_),
                    self.loop,
                ).result()
            except StopAsyncIteration:
                break


    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method supports filtering by configuration (thread_id) and metadata.
        It handles pagination using the `before` cursor and `limit` parameters.
        
        Optimization Note:
        Where possible, sorting and limiting are pushed down to the database query.
        However, for complex metadata queries without composite indexes, consistent sorting
        is enforced in-memory to guarantee correctness.

        Args:
            config (RunnableConfig): Base configuration for filtering checkpoints.
            filter (Optional[dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of matching checkpoint tuples.
        """
        where_clause, parameters = search_where(config, filter, before)
        
        # Query Scope Selection:
        # - If thread_id AND checkpoint_ns are known, scope to single partition (faster, cheaper).
        # - Otherwise, enable cross-partition query to search all threads/namespaces.
        query_partition_key = None
        if config:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                query_partition_key = _make_cosmosdb_checkpoint_key(thread_id, checkpoint_ns, '')
             
        query = f"SELECT * FROM c {where_clause}"
        
        # Optimization: Push ORDER BY/LIMIT to DB when no metadata filters are present.
        if not filter:
            # Exclude the 'tip' document and write documents from the results.
            # Write documents are stored in partitions starting with 'writes' prefix.
            if "WHERE" in query:
                query += " AND c.type != 'tip' AND STARTSWITH(c.partition_key, 'checkpoint')"
            else:
                query += " WHERE c.type != 'tip' AND STARTSWITH(c.partition_key, 'checkpoint')"
                
            query += " ORDER BY c.checkpoint_id DESC"
            
            if limit is not None:
                query += f" OFFSET 0 LIMIT {limit}"
        else:
            # Push ORDER BY requires composite indexes for metadata filters; fetch and sort in-memory instead.
            # Exclude the 'tip' document and write documents from the results.
            if "WHERE" in query:
                query += " AND c.type != 'tip' AND STARTSWITH(c.partition_key, 'checkpoint')"
            else:
                query += " WHERE c.type != 'tip' AND STARTSWITH(c.partition_key, 'checkpoint')"

        # Execute query with partition scope if available.
        kwargs = {}
        if query_partition_key:
            kwargs["partition_key"] = query_partition_key

        # Fetch all matching items into memory for consistent sorting when metadata filters are present.
        items = [item async for item in self.container.query_items(query=query, parameters=parameters, **kwargs)]
        
        # Sorted in-memory for cases where metadata filters prevent push-down ORDER BY.
        items.sort(key=lambda x: x["checkpoint_id"], reverse=True)
        
        if limit:
            items = items[:limit]

        for item in items:
            if item.get("type") == "tip":
                continue
            # Parse item to CheckpointTuple (simplified version of aget_tuple logic)
            # Re-parsing key might be needed if query didn't specify everything
            thread_id = item["thread_id"]
            checkpoint_ns = item["checkpoint_ns"]
            checkpoint_id = item["checkpoint_id"]
            
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }
            
            # Deserialize checkpoint from Base64-encoded blob.
            c_type = item["type"]
            c_bytes = base64.b64decode(item["checkpoint"])
            checkpoint = self.serde.loads_typed((c_type, c_bytes))
            
            # Metadata Recovery: prefer blob for fidelity, fall back to JSON.
            metadata_val = item.get("metadata_blob")
            if metadata_val and isinstance(metadata_val, str):
                m_type = item.get("metadata_type")
                m_bytes = base64.b64decode(metadata_val)
                if m_type:
                    metadata = self.serde.loads_typed((m_type, m_bytes))
                else:
                    metadata = item.get("metadata", {})
            else:
                metadata = item.get("metadata", {})
                if isinstance(metadata, str):
                    metadata = {}
            
            parent_id = item.get("parent_checkpoint_id")
            parent_config = (
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": parent_id}}
                if parent_id else None
            )
            
            # Helper: Writes partition key and query params
            writes_pk = _make_cosmosdb_writes_partition_key(thread_id, checkpoint_ns)
            w_query = "SELECT * FROM c WHERE c.partition_key=@pk AND c.checkpoint_id=@cp_id"
            w_params = [
                {"name": "@pk", "value": writes_pk},
                {"name": "@cp_id", "value": checkpoint_id}
            ]
            
            parsed_writes = []
            async for w_item in self.container.query_items(query=w_query, parameters=w_params):
                w_type = w_item["type"]
                w_bytes = base64.b64decode(w_item["value"])
                val = self.serde.loads_typed((w_type, w_bytes))
                parsed_writes.append((w_item["task_id"], w_item["channel"], val))
            
            # Sort in-memory
            parsed_writes.sort(key=lambda x: (x[0], WRITES_IDX_MAP.get(x[1], 0)))
            
            yield CheckpointTuple(
                config, checkpoint, metadata, parent_config, parsed_writes
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method is a synchronous bridge to the asynchronous `aput` method.
        It is provided for compatibility with the synchronous `BaseCheckpointSaver` interface.

        Warning:
            This method blocks the current thread until the async operation completes.
            It should only be called from a background thread to avoid blocking the
            event loop.

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save.
            new_versions (ChannelVersions): New channel versions.

        Returns:
            RunnableConfig: Updated configuration with the saved checkpoint ID.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()


    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        Updates the Cosmos DB container with the new checkpoint and atomically updates
        the "Tip Document" to point to it. This ensures that concurrent reads always see
        a consistent state.

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save (saved in both JSON and blob formats).
            new_versions (ChannelVersions): New channel versions (unused in storage but part of signature).

        Returns:
            RunnableConfig: Updated configuration with the saved checkpoint ID.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        
        partition_key = _make_cosmosdb_checkpoint_key(thread_id, checkpoint_ns, '')
        key = _make_cosmosdb_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
        
        # Merge metadata from config
        metadata = get_checkpoint_metadata(config, metadata)

        # Serialize checkpoint and metadata to Base64 for Cosmos DB storage.
        type_, checkpoint_bytes = self.serde.dumps_typed(checkpoint)
        checkpoint_b64 = base64.b64encode(checkpoint_bytes).decode('utf-8')
        
        md_type, metadata_bytes = self.serde.dumps_typed(metadata)
        metadata_b64 = base64.b64encode(metadata_bytes).decode('utf-8')
        
        # Metadata is stored as searchable JSON (for filters) and Base64 blob (for high-fidelity recovery).
        data = {
            "id": key,
            "partition_key": partition_key,
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": parent_checkpoint_id,
            "type": type_,
            "checkpoint": checkpoint_b64,
            "metadata": metadata, # Searchable JSON
            "metadata_type": md_type,
            "metadata_blob": metadata_b64 # High-fidelity blob
        }
        
        # Tip Document: single-partition document acting as a head pointer for O(1) head lookups.
        tip_key = _make_cosmosdb_tip_key(thread_id, checkpoint_ns)
        tip_data = {
            "id": tip_key,
            "partition_key": partition_key, # Tip lives in the same partition for atomicity
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id, # Point to the new head
            "type": "tip"
        }
        
        # Transactional batch ensures Checkpoint and Tip update are atomic.
        batch_operations = [
            ("upsert", (data,), {}),
            ("upsert", (tip_data,), {})
        ]
        
        await self.container.execute_item_batch(batch_operations, partition_key)
        
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method is a synchronous bridge to the asynchronous `aput_writes` method.
        It is provided for compatibility with the synchronous `BaseCheckpointSaver` interface.

        Warning:
            This method blocks the current thread until the async operation completes.
            It should only be called from a background thread to avoid blocking the
            event loop.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[tuple[str, Any]]): List of writes to store.
            task_id (str): Identifier for the task creating the writes.
            task_path (str): Path of the task creating the writes.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id, task_path), self.loop
        ).result()


    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        Writes are stored in a partition dedicated to the thread's write history.
        This keeps the checkpoint partition clean and performant.

        Concurrency Handling:
        *   **System Channels** (fixed indices): Use `upsert` (last write wins).
        *   **Stream Channels**: Use `create` (optimistic concurrency). If a write already exists 
            (e.g., race condition or retry), it will fail (Status 409), ensuring we strictly append 
            or fail rather than silently overwriting history.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[tuple[str, Any]]): List of writes to store.
            task_id (str): Identifier for the task creating the writes.
            task_path (str): Path of the task creating the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]
        
        partition_key = _make_cosmosdb_writes_partition_key(thread_id, checkpoint_ns)
        
        # Concurrency Strategy Selection - Why Upsert vs Create?
        # 1. System Channels (all_special): These use fixed indices. If multiple writes hit the same 
        #    system index, we use 'upsert' so the last valid write wins.
        # 2. Stream Channels (dynamic): These are timestamps or sequential IDs. We use 'create'
        #    to fail (409) if a write already exists, ensuring strictly monotonic history
        #    without silent overwrites during retries or race conditions.
        all_special = all(w[0] in WRITES_IDX_MAP for w in writes)
        batch_operations = []
        
        # Shared logic in `utils.prepare_batch_items`.
        write_items = prepare_batch_items(
            writes=writes,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            partition_key=partition_key,
            serde=self.serde
        )

        # 1. System Channels (all_special): Safe Upsert Batch
        if all_special:
            for item in write_items:
                batch_operations.append(("upsert", (item,), {}))
        # 2. Stream Channels: Optimistic Create Batch with Fallback
        else:
            for item in write_items:
                 batch_operations.append(("create", (item,), {}))
                 
        # Cosmos DB Transactional Batches have a strict limit of 100 operations.
        # We chunk large write sets into multiple atomic blocks.
        if batch_operations:
            tasks = []
            for i in range(0, len(batch_operations), 100):
                batch_chunk = batch_operations[i : i + 100]
                # We wrap the batch call in a helper to handle the fallback per-batch
                tasks.append(self._execute_batch_with_fallback(batch_chunk, partition_key, all_special))
            
            await asyncio.gather(*tasks)

    async def _execute_batch_with_fallback(self, batch_ops: list[tuple], pk: str, is_upsert: bool) -> None:
        """Execute a batch of operations with fallback to single-item writes on conflict.
        
        If a batch fails due to a conflict (409) in `create` mode (non-upsert), 
        we fall back to processing items individually to isolate the failure 
        and allow non-conflicting writes to succeed. This ensures robustness 
        during concurrent access.

        Args:
            batch_ops: List of batch operation tuples (op_type, (item,), {}).
            pk: The partition key for the batch.
            is_upsert: If True, operations are upserts (no fallback needed).
        """
        try:
             await self.container.execute_item_batch(batch_ops, pk)
        except CosmosBatchOperationError as e:
            if not is_upsert and e.status_code == 409:
                # Fallback: Batch failed generally, retry items one-by-one
                # Extract items from ops structure: ("create", (item,), {})
                for op in batch_ops:
                    item = op[1][0]
                    try:
                         await self.container.create_item(item)
                    except CosmosHttpResponseError as e2:
                        if e2.status_code == 409:
                            pass
                        else:
                            raise
            else:
                raise

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID.

        This method is a synchronous bridge to the asynchronous `adelete_thread` method.
        It is provided for compatibility with the synchronous `BaseCheckpointSaver` interface.

        Warning:
            This method blocks the current thread until the async operation completes.
            It should only be called from a background thread to avoid blocking the
            event loop.

        Args:
            thread_id (str): The thread ID to delete.

        Raises:
            RuntimeError: If "Partition Key Delete" is not enabled on the Azure Cosmos DB account.
            CosmosHttpResponseError: If deletion fails for other reasons.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            # we don't check in other methods to avoid the overhead
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncCosmosDBSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `checkpointer.alist(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.adelete_thread(thread_id), self.loop
        ).result()


    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID asynchronously.

        Efficiently deletes data using the "Delete All Items By Partition Key" feature.
        This performs a server-side bulk delete of the entire logical partition, 
        which is significantly faster and cheaper (RU/s) than deleting items one by one.

        Args:
            thread_id (str): The ID of the thread to delete.

        Raises:
            RuntimeError: If "Partition Key Delete" is not enabled on the Azure Cosmos DB account.
            CosmosHttpResponseError: If deletion fails for other reasons.
        """
        # Step 1: Identify all logical partitions associated with this thread.
        # DISTINCT reduces the number of server-side delete calls needed.
        query = "SELECT DISTINCT c.partition_key FROM c WHERE STARTSWITH(c.partition_key, @cp_prefix) OR STARTSWITH(c.partition_key, @wr_prefix)"
        params = [
            {"name": "@cp_prefix", "value": f"checkpoint${thread_id}$"},
            {"name": "@wr_prefix", "value": f"writes${thread_id}$"}
        ]
        
        # Step 2: Perform server-side bulk deletion (requires "Delete All Items By Partition Key" enabled).
        async for item in self.container.query_items(query=query, parameters=params):
            pk = item["partition_key"]
            try:
                await self.container.delete_all_items_by_partition_key(pk)
            except CosmosHttpResponseError as e:
                err_msg = str(e)
                if e.status_code == 400 and ("Partition key delete feature is disabled" in err_msg):
                    raise RuntimeError(
                        "Failed to delete thread. The 'Delete All Items By Partition Key' feature is not enabled "
                        "on your Azure Cosmos DB account. Please enable this feature in the Azure Portal "
                        "(Features > 'Delete All Items By Partition Key') or via CLI."
                    ) from e
                elif e.status_code == 404:
                     pass # Already gone
                else:
                    raise

    def get_next_version(self, current: str | None, channel: Any) -> str:
        """Generate the next version ID for a channel.

        This method creates a new version identifier for a channel based on its current version.
        Versions are strings representing a decimal number followed by a random component 
        to ensure uniqueness even in high-concurrency environments with identical timestamps.

        Args:
            current (Optional[str]): The current version identifier of the channel.
            channel (Any): The channel object (unused in version generation).

        Returns:
            str: The next version identifier, which is guaranteed to be lexicographically 
            monotonically increasing.
        """
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        
        # Increment version and append random float as a collision tie-breaker.
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"
