"""Utility functions for LangGraph Checkpoint Cosmos DB.

This module provides helper functions for:
- Document key generation (checkpoints, writes, tips)
- Partition key construction  
- Query building with parameterized WHERE clauses
- Batch item preparation for transactional writes

These utilities support the core checkpoint saver classes and handle
Cosmos DB-specific concerns like document IDs and SQL API parameterization.
"""
from __future__ import annotations

import base64
import re
from typing import Any, Optional, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    SerializerProtocol,
    get_checkpoint_id,
)

# Delimiter for composite document IDs (e.g., "checkpoint$thread-1$ns$id").
# Using '$' as it's URL-safe and unlikely to appear in user-provided IDs.
COSMOSDB_KEY_SEPARATOR = "$"

# Pattern for validating metadata filter keys to prevent injection attacks.
# Allows: alphanumeric, underscore, dot, hyphen (covers typical JSON property names).
_FILTER_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")


def _make_cosmosdb_checkpoint_key(thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
    """Generate a unique document ID for a checkpoint.

    Checkpoint IDs follow the schema: `checkpoint${thread_id}${checkpoint_ns}${id}`.
    This structured ID allows for direct lookups and defines the document's 
    place within its logical partition.
    """
    return COSMOSDB_KEY_SEPARATOR.join([
        "checkpoint", thread_id, checkpoint_ns, checkpoint_id
    ])

def _make_cosmosdb_checkpoint_writes_key(thread_id: str, checkpoint_ns: str, checkpoint_id: str, task_id: str, idx: Optional[int]) -> str:
    """Generate a unique document ID for a specific write.

    Write IDs follow the schema: `writes${thread_id}${checkpoint_ns}${checkpoint_id}${task_id}${idx}`.
    The inclusion of `task_id` and `idx` ensures uniqueness for multiple writes within 
    the same task and checkpoint.
    """
    if idx is None:
        return COSMOSDB_KEY_SEPARATOR.join([
            "writes", thread_id, checkpoint_ns, checkpoint_id, task_id
        ])
    return COSMOSDB_KEY_SEPARATOR.join([
        "writes", thread_id, checkpoint_ns, checkpoint_id, task_id, str(idx)
    ])

def _make_cosmosdb_writes_partition_key(thread_id: str, checkpoint_ns: str) -> str:
    """Generate a partition key for a thread's write history.

    All writes for a specific thread/namespace are stored in a dedicated logical 
    partition (`writes${thread_id}${checkpoint_ns}`) to isolate them from high-traffic 
    checkpoint metadata.
    """
    return COSMOSDB_KEY_SEPARATOR.join([
        "writes", thread_id, checkpoint_ns
    ])

def _make_cosmosdb_tip_key(thread_id: str, checkpoint_ns: str) -> str:
    """Generate the document ID for a thread's pointer document ("Tip").

    The Tip Document has a fixed ID (`tip${thread_id}${checkpoint_ns}`) allowing for
    O(1) retrieval of the latest checkpoint ID from any state.
    """
    return COSMOSDB_KEY_SEPARATOR.join([
        "tip", thread_id, checkpoint_ns
    ])


def prepare_batch_items(
    writes: Sequence[tuple[str, Any]],
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    task_id: str,
    partition_key: str,
    serde: SerializerProtocol
) -> list[dict[str, Any]]:
    """Prepare a list of write items for atomic batch execution.

    This function serializes values using the provided `serde` and encodes them as 
    Base64 strings to ensure safe storage and transport within Cosmos DB's JSON 
    document structure.

    Args:
        writes (Sequence[tuple[str, Any]]): Sequence of (channel, value) tuples.
        thread_id (str): The thread ID.
        checkpoint_ns (str): The checkpoint namespace.
        checkpoint_id (str): The associated checkpoint ID.
        task_id (str): The ID of the task generating the writes.
        partition_key (str): The logical partition key where writes are stored.
        serde (SerializerProtocol): The serializer to use.

    Returns:
        list[dict[str, Any]]: A list of document dictionaries ready for `execute_item_batch`.
    """
    write_items = []
    for idx, (channel, value) in enumerate(writes):
        # We use a fixed index mapping for system channels (e.g., __start__) 
        # to ensure they occupy deterministic slots across retries.
        fixed_idx = WRITES_IDX_MAP.get(channel, idx)
        
        key = _make_cosmosdb_checkpoint_writes_key(
            thread_id, checkpoint_ns, checkpoint_id, task_id, fixed_idx
        )
        
        type_, val_bytes = serde.dumps_typed(value)
        val_b64 = base64.b64encode(val_bytes).decode('utf-8')
        
        data = {
            "id": key,
            "partition_key": partition_key,
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "task_id": task_id,
            "idx": fixed_idx,
            "channel": channel,
            "type": type_,
            "value": val_b64
        }
        write_items.append(data)
    return write_items


def _validate_filter_key(key: str) -> None:
    """Validate that a filter key is safe for handling in SQL queries.

    This ensures that keys used in dynamic queries contain only safe characters,
    preventing potential SQL injection vulnerabilities.

    Args:
        key (str): The filter key to validate.

    Raises:
        ValueError: If the key contains invalid characters.
    """
    # Allow alphanumeric characters, underscores, dots, and hyphens
    # This covers typical JSON property names while preventing SQL injection
    if not _FILTER_PATTERN.match(key):
        raise ValueError(
            f"Invalid filter key: '{key}'. Filter keys must contain only alphanumeric characters, underscores, dots, and hyphens."
        )


def _metadata_predicate(
    metadata_filter: dict[str, Any],
) -> tuple[Sequence[str], Sequence[dict[str, Any]]]:
    """Generate WHERE clause predicates for metadata filtering.

    Constructs parameterized SQL predicates for querying the `metadata` object.
    Parameters are used for all values to prevent SQL injection.

    Args:
        metadata_filter (dict[str, Any]): A dictionary of key-value pairs to filter by. 
            Keys must be safe (validated via `_validate_filter_key`).

    Returns:
        tuple[Sequence[str], Sequence[dict[str, Any]]]: 
            - A sequence of SQL predicate strings (e.g., "c.metadata['key'] = @metadata_0").
            - A sequence of parameter dictionaries.
    """

    predicates = []
    parameters = []

    for idx, (query_key, query_value) in enumerate(metadata_filter.items()):
        _validate_filter_key(query_key)
        
        # Why Bracket Notation?
        # Cosmos DB requires bracket notation (c.metadata["key-name"]) for properties 
        # that contain hyphens or other special characters. Dot notation (c.metadata.key-name)
        # would be interpreted as a subtraction expression.
        path_parts = query_key.split(".")
        path = "".join([f'["{p}"]' for p in path_parts])
        
        if query_value is None:
            # Handle None: match documents where property is missing (undefined) OR explicitly null.
            predicates.append(
                f"(NOT IS_DEFINED(c.metadata{path}) OR c.metadata{path} = null)"
            )
        else:
            param_name = f"@metadata_{idx}"
            predicates.append(f"c.metadata{path} = {param_name}")
            parameters.append({"name": param_name, "value": query_value})

    return (predicates, parameters)


def search_where(
    config: RunnableConfig | None,
    filter: dict[str, Any] | None,
    before: RunnableConfig | None = None,
) -> tuple[str, Sequence[dict[str, Any]]]:
    """Construct a parameterized SQL WHERE clause for checkpoint searches.

    This function orchestrates the generation of filter conditions for thread IDs, 
    namespaces, metadata, and pagination cursors. It strictly uses parameters 
    to protect against SQL injection.

    Args:
        config (RunnableConfig | None): Base configuration (thread_id, checkpoint_ns).
        filter (dict[str, Any] | None): Metadata criteria for filtering.
        before (RunnableConfig | None): Pagination cursor for checkpoints before a specific ID.

    Returns:
        tuple[str, Sequence[dict[str, Any]]]: 
            - The SQL WHERE clause string (e.g., "WHERE c.thread_id = @thread_id ...").
            - A list of parameters for the Cosmos DB client.
    """
    wheres = []
    parameters = []

    # 1. Base Filters: thread_id and checkpoint_ns are the primary partition keys.
    if config is not None:
        wheres.append("c.thread_id = @thread_id")
        parameters.append({"name": "@thread_id", "value": str(config["configurable"]["thread_id"])})
        
        checkpoint_ns = config["configurable"].get("checkpoint_ns")
        if checkpoint_ns is not None:
            wheres.append("c.checkpoint_ns = @checkpoint_ns")
            parameters.append({"name": "@checkpoint_ns", "value": checkpoint_ns})

        if checkpoint_id := get_checkpoint_id(config):
            wheres.append("c.checkpoint_id = @checkpoint_id")
            parameters.append({"name": "@checkpoint_id", "value": checkpoint_id})

    # 2. Metadata Filters: Pushed down as far as possible to the database query.
    if filter:
        metadata_predicates, metadata_params = _metadata_predicate(filter)
        wheres.extend(metadata_predicates)
        parameters.extend(metadata_params)

    # 3. Pagination Filter: Effectively implements the 'before' cursor logic.
    if before is not None:
        wheres.append("c.checkpoint_id < @before_checkpoint_id")
        parameters.append({"name": "@before_checkpoint_id", "value": get_checkpoint_id(before)})

    if not wheres:
        return ("", [])

    return ("WHERE " + " AND ".join(wheres), parameters)
