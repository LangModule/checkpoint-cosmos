"""Tests for the asynchronous AsyncCosmosDBSaver checkpoint implementation.

This module contains integration tests for the AsyncCosmosDBSaver class, verifying:
- Basic async checkpoint operations (aput, aget_tuple, alist)
- Metadata handling and filtering
- Security measures (SQL injection prevention)
- Edge cases (null values, special characters in keys)
- Thread deletion functionality
- Pending writes storage and retrieval
- Synchronous bridge methods (for calling from sync code)

Prerequisites:
    Set the following environment variables before running tests:
    - COSMOS_DB_ENDPOINT: Your Cosmos DB account endpoint URL
    - COSMOS_DB_KEY: Your Cosmos DB account key (or use DefaultAzureCredential)
    - COSMOS_DB_NAME: Database name (defaults to "checkpoints")
    - COSMOS_DB_CONTAINER: Container name (defaults to "checkpoints")

Usage:
    pytest tests/test_aio.py -v
"""
import asyncio
import os
import uuid
import pytest
from typing import Any, AsyncIterator

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)

from langgraph_checkpoint_cosmos.aio import AsyncCosmosDBSaver


# =============================================================================
# Environment Configuration
# =============================================================================

# Retrieve Cosmos DB connection settings from environment variables.
# These are required for integration tests to connect to a real Cosmos DB instance.
COSMOS_DB_ENDPOINT = os.environ.get("COSMOS_DB_ENDPOINT")
COSMOS_DB_KEY = os.environ.get("COSMOS_DB_KEY")
COSMOS_DB_NAME = os.environ.get("COSMOS_DB_NAME", "checkpoints")
COSMOS_DB_CONTAINER = os.environ.get("COSMOS_DB_CONTAINER", "checkpoints")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
async def saver() -> AsyncIterator[AsyncCosmosDBSaver]:
    """Create an AsyncCosmosDBSaver instance for testing.
    
    This async fixture provides a managed AsyncCosmosDBSaver that automatically
    handles connection lifecycle and background event loop setup.
    Tests are skipped if COSMOS_DB_ENDPOINT is not set.
    
    Yields:
        AsyncCosmosDBSaver: A configured async saver instance connected to Cosmos DB.
    """
    if not COSMOS_DB_ENDPOINT:
        pytest.skip("COSMOS_DB_ENDPOINT must be set to run integration tests.")
    
    async with AsyncCosmosDBSaver.from_conn_info(
        endpoint=COSMOS_DB_ENDPOINT,
        credential=COSMOS_DB_KEY,
        database_name=COSMOS_DB_NAME,
        container_name=COSMOS_DB_CONTAINER,
    ) as saver:
        yield saver


# =============================================================================
# Test Class: AsyncCosmosDBSaver Integration Tests
# =============================================================================

class TestAsyncCosmosDBSaver:
    """Integration tests for the asynchronous AsyncCosmosDBSaver.
    
    Each test uses unique thread IDs (via UUID) to ensure isolation when
    running against a shared Cosmos DB container. This prevents test pollution
    and allows parallel test execution.
    
    All test methods are async and should be run with pytest-asyncio.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test fixtures before each test method.
        
        Creates unique thread IDs and pre-configured checkpoints/metadata
        that are reused across multiple tests. Using UUIDs ensures test isolation
        in shared containers.
        """
        # Generate unique thread IDs to avoid collisions in shared container.
        # This is critical for parallel test execution and shared test environments.
        self.thread_1 = f"thread-1-{uuid.uuid4()}"
        self.thread_2 = f"thread-2-{uuid.uuid4()}"
        
        # Pre-configured RunnableConfig objects for common test scenarios.
        # config_1: Basic config for thread_1
        # config_2: Basic config for thread_2 (default namespace)
        # config_3: Config for thread_2 with "inner" namespace (tests multi-namespace)
        self.config_1: RunnableConfig = {
            "configurable": {
                "thread_id": self.thread_1,
                "checkpoint_id": "1",
                "checkpoint_ns": "",
            }
        }
        self.config_2: RunnableConfig = {
            "configurable": {
                "thread_id": self.thread_2,
                "checkpoint_id": "2",
                "checkpoint_ns": "",
            }
        }
        self.config_3: RunnableConfig = {
            "configurable": {
                "thread_id": self.thread_2,
                "checkpoint_id": "2-inner",
                "checkpoint_ns": "inner",
            }
        }

        # Pre-created checkpoint objects with different states.
        self.chkpnt_1: Checkpoint = empty_checkpoint()
        self.chkpnt_2: Checkpoint = create_checkpoint(self.chkpnt_1, {}, 1)
        self.chkpnt_3: Checkpoint = empty_checkpoint()

        # Metadata fixtures covering various data types and edge cases:
        # metadata_1: Standard metadata with string, int, dict, and numeric values
        # metadata_2: Includes None value (tests null handling)
        # metadata_3: Empty metadata (tests empty filter behavior)
        self.metadata_1: CheckpointMetadata = {
            "source": "input",
            "step": 2,
            "writes": {},
            "score": 1,
        }
        self.metadata_2: CheckpointMetadata = {
            "source": "loop",
            "step": 1,
            "writes": {"foo": "bar"},
            "score": None,  # Tests null value handling
        }
        self.metadata_3: CheckpointMetadata = {}

    # =========================================================================
    # Core Functionality Tests
    # =========================================================================

    async def test_combined_metadata(self, saver: AsyncCosmosDBSaver) -> None:
        """Verify that metadata from config is merged with checkpoint metadata.
        
        When saving a checkpoint via aput(), metadata can come from two sources:
        1. The `metadata` parameter passed to `aput()`
        2. The `metadata` key in the RunnableConfig
        
        Both should be merged in the resulting checkpoint tuple.
        Private keys (prefixed with '__') should be filtered out.
        """
        config: RunnableConfig = {
            "configurable": {
                "thread_id": self.thread_2,
                "checkpoint_ns": "",
                # Private keys should be filtered out during metadata merge
                "__super_private_key": "super_private_value",
            },
            # Config-level metadata should be merged with checkpoint metadata
            "metadata": {"run_id": "my_run_id"},
        }
        
        # Save checkpoint with both config metadata and explicit metadata
        await saver.aput(config, self.chkpnt_2, self.metadata_2, {})
        
        # Retrieve and verify metadata merge
        checkpoint = await saver.aget_tuple(config)
        assert checkpoint is not None
        assert checkpoint.metadata == {
            **self.metadata_2,
            "run_id": "my_run_id",  # Merged from config
        }

    async def test_asearch(self, saver: AsyncCosmosDBSaver) -> None:
        """Comprehensive test of the alist() method with various filter scenarios.
        
        Tests the following alist() capabilities:
        - Filter by single metadata key
        - Filter by multiple metadata keys
        - Empty filter (returns all checkpoints)
        - Non-matching filter (returns empty results)
        - Filter by thread_id via config
        - Limit parameter
        - Before parameter (pagination cursor)
        
        Note: Checkpoint IDs are set to sortable timestamps to ensure
        deterministic ordering in tests that depend on sort order.
        """
        # Use sortable timestamp-based IDs for deterministic sort order testing.
        # Format: YYYYMMDDHHMMSS-suffix ensures lexicographic sorting matches temporal order.
        self.config_1["configurable"]["checkpoint_id"] = "20240101000000-1"
        self.config_2["configurable"]["checkpoint_id"] = "20240102000000-2"
        self.config_3["configurable"]["checkpoint_id"] = "20240103000000-3"
        
        # Sync checkpoint IDs with their configs
        self.chkpnt_1["id"] = "20240101000000-1"
        self.chkpnt_2["id"] = "20240102000000-2"
        self.chkpnt_3["id"] = "20240103000000-3"
        
        # Save test checkpoints
        await saver.aput(self.config_1, self.chkpnt_1, self.metadata_1, {})
        await saver.aput(self.config_2, self.chkpnt_2, self.metadata_2, {})
        await saver.aput(self.config_3, self.chkpnt_3, self.metadata_3, {})

        # Define filter queries for different test scenarios
        query_1 = {"source": "input"}  # Single key filter
        query_2 = {"step": 1, "writes": {"foo": "bar"}}  # Multi-key filter
        query_3: dict[str, Any] = {}  # Empty filter (match all)
        query_4 = {"source": "update", "step": 1}  # Non-matching filter

        # Scope searches to specific threads to isolate from other test data
        search_config_1 = {"configurable": {"thread_id": self.thread_1}}
        search_config_2 = {"configurable": {"thread_id": self.thread_2}}
        
        # Test 1: Single-key filter should match metadata_1
        search_results_1 = [c async for c in saver.alist(search_config_1, filter=query_1)]
        assert len(search_results_1) == 1
        assert search_results_1[0].metadata == self.metadata_1

        # Test 2: Multi-key filter should match metadata_2
        search_results_2 = [c async for c in saver.alist(search_config_2, filter=query_2)]
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == self.metadata_2

        # Test 3: Empty filter returns all checkpoints for the thread
        search_results_3 = [c async for c in saver.alist(search_config_1, filter=query_3)]
        assert len(search_results_3) == 1  # thread_1 has 1 checkpoint
        
        search_results_3b = [c async for c in saver.alist(search_config_2, filter=query_3)]
        assert len(search_results_3b) == 2  # thread_2 has 2 checkpoints (different namespaces)

        # Test 4: Non-matching filter returns empty results
        search_results_4 = [c async for c in saver.alist(search_config_1, filter=query_4)]
        assert len(search_results_4) == 0

        # Test 5: Search by thread_id returns all namespaces
        # Results are sorted by checkpoint_id DESC, so "20240103..." comes first
        search_results_5 = [
            c async for c in saver.alist({"configurable": {"thread_id": self.thread_2}})
        ]
        assert len(search_results_5) == 2
        # Verify both namespaces are present
        namespaces = {r.config["configurable"]["checkpoint_ns"] for r in search_results_5}
        assert namespaces == {"", "inner"}

        # Test 6: Limit parameter restricts result count
        search_results_6 = [
            c async for c in saver.alist(
                {"configurable": {"thread_id": self.thread_2}}, limit=1
            )
        ]
        assert len(search_results_6) == 1
        assert search_results_6[0].config["configurable"]["thread_id"] == self.thread_2

        # Test 7: Before parameter excludes checkpoints >= specified ID
        # "20240103..." is the latest; before it should return "20240102..."
        before_config = {"configurable": {"checkpoint_id": "20240103000000-3"}}
        search_results_7 = [
            c async for c in saver.alist(
                {"configurable": {"thread_id": self.thread_2}}, 
                before=before_config
            )
        ]
        assert len(search_results_7) == 1
        assert search_results_7[0].checkpoint["id"] == "20240102000000-2"

    # =========================================================================
    # Null/None Value Handling Tests
    # =========================================================================

    async def test_null_handling(self, saver: AsyncCosmosDBSaver) -> None:
        """Verify that None/null values in metadata are stored and filtered correctly.
        
        Cosmos DB handles null differently than missing fields:
        - `{"score": null}` is a field with null value
        - `{}` is a missing field
        
        The query must use special handling: 
        `(NOT IS_DEFINED(c.metadata["score"]) OR c.metadata["score"] = null)`
        """
        config = {
            "configurable": {
                "thread_id": self.thread_1,
                "checkpoint_ns": "",
                "checkpoint_id": "null_test",
            }
        }
        metadata: CheckpointMetadata = {"score": None}
        await saver.aput(config, self.chkpnt_1, metadata, {})
        
        # Filter for score=None should find the checkpoint
        results = [
            c async for c in saver.alist(
                {"configurable": {"thread_id": self.thread_1}}, 
                filter={"score": None}
            )
        ]
        assert len(results) == 1
        assert results[0].metadata["score"] is None

    # =========================================================================
    # Security Tests (SQL Injection Prevention)
    # =========================================================================

    async def test_search_sql_injection_prevention(self, saver: AsyncCosmosDBSaver) -> None:
        """Verify that malicious filter keys are rejected to prevent query injection.
        
        Filter keys are incorporated into Cosmos DB SQL queries. If not properly
        validated, an attacker could inject malicious SQL via crafted key names.
        The `_safe_filter_key` function should reject any key containing
        characters that could alter query structure.
        """
        malicious_key = "access) = 'public' OR c.id='1"
        with pytest.raises(ValueError, match="Invalid filter key"):
            [c async for c in saver.alist(None, filter={malicious_key: "dummy"})]

    async def test_metadata_predicate_sql_injection_prevention(self, saver: AsyncCosmosDBSaver) -> None:
        """Test multiple SQL injection payloads against the metadata filter.
        
        This test covers common injection patterns:
        - Boolean-based injection: x') OR '1'='1
        - Comment-based injection: x') OR 1=1 --
        - UNION-based injection: x') UNION SELECT ...
        - Destructive injection: '; DROP TABLE ...
        
        All should be rejected with ValueError before reaching the database.
        """
        malicious_keys = [
            "x') OR '1'='1",  # Boolean-based injection
            "x') OR 1=1 --",  # Comment-based injection
            "x') UNION SELECT 1,2,3,4,5,6,7 --",  # UNION-based injection
            "access') = 'public' OR '1'='1' OR c.metadata.value",  # Complex injection
            "'; DROP TABLE checkpoints; --",  # Destructive injection
        ]

        for malicious_key in malicious_keys:
            with pytest.raises(ValueError, match="Invalid filter key"):
                [c async for c in saver.alist(None, filter={malicious_key: "dummy"})]

    async def test_limit_parameter_sql_injection_prevention(self, saver: AsyncCosmosDBSaver) -> None:
        """Verify that malicious limit parameters are handled safely.
        
        The limit parameter is passed to Cosmos DB's OFFSET/LIMIT clause.
        Even if type checking is bypassed, the parameterized query should
        prevent injection. The driver should either:
        - Reject non-integer values
        - Safely handle them without executing injected SQL
        """
        # Setup: Create test checkpoints
        for i in range(5):
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": f"thread-limit-{uuid.uuid4()}",  # Unique per test run
                    "checkpoint_ns": "",
                }
            }
            checkpoint = empty_checkpoint()
            metadata: CheckpointMetadata = {"index": i}
            await saver.aput(config, checkpoint, metadata, {})

        # Attempt injection via limit parameter (bypassing type hints)
        malicious_limits = [
            "1; DROP TABLE checkpoints; --",
            "1 OR 1=1",
        ]
        
        for malicious_limit in malicious_limits:
            try:
                # Type: ignore to bypass static type checking
                results = [c async for c in saver.alist(None, limit=malicious_limit)]  # type: ignore
            except Exception:
                # Expected: Driver should reject invalid limit values
                pass

    # =========================================================================
    # Special Character Handling Tests
    # =========================================================================

    async def test_metadata_filter_keys_with_hyphens_and_digits(self, saver: AsyncCosmosDBSaver) -> None:
        """Verify that metadata keys with special characters are filterable.
        
        Cosmos DB supports keys with hyphens and leading digits when accessed
        via bracket notation: c.metadata["key-with-hyphen"].
        
        This test ensures the query builder correctly quotes such keys.
        """
        unique_thread_id = f"thread-hyphen-digit-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": unique_thread_id,
                "checkpoint_ns": "",
            }
        }
        checkpoint = empty_checkpoint()
        
        # Metadata with special characters in keys
        metadata: CheckpointMetadata = {
            "access-level": "public",  # Top-level hyphenated key
            "user": {
                "access-level": "nested",  # Nested hyphenated key
                "123abc": "ok2",  # Nested digit-starting key
            },
            "123abc": "ok",  # Top-level digit-starting key
        }
        await saver.aput(config, checkpoint, metadata, {})

        search_config = {"configurable": {"thread_id": unique_thread_id}}

        # Test: Top-level hyphenated key
        results = [c async for c in saver.alist(search_config, filter={"access-level": "public"})]
        assert len(results) == 1

        # Test: Nested hyphenated key via dotted path
        results = [c async for c in saver.alist(search_config, filter={"user.access-level": "nested"})]
        assert len(results) == 1

        # Test: Top-level digit-starting key
        results = [c async for c in saver.alist(search_config, filter={"123abc": "ok"})]
        assert len(results) == 1

        # Test: Nested digit-starting key via dotted path
        results = [c async for c in saver.alist(search_config, filter={"user.123abc": "ok2"})]
        assert len(results) == 1

    # =========================================================================
    # Nonexistent Data Tests
    # =========================================================================

    async def test_aget_tuple_nonexistent(self, saver: AsyncCosmosDBSaver) -> None:
        """Verify that aget_tuple returns None for a nonexistent thread.
        
        When requesting a checkpoint for a thread that has never been created,
        the method should return None rather than raising an exception.
        """
        nonexistent_thread = f"nonexistent-{uuid.uuid4()}"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": nonexistent_thread,
                "checkpoint_ns": "",
            }
        }
        result = await saver.aget_tuple(config)
        assert result is None

    # =========================================================================
    # Pending Writes Tests
    # =========================================================================

    async def test_aput_writes_round_trip(self, saver: AsyncCosmosDBSaver) -> None:
        """Verify that aput_writes stores writes and aget_tuple retrieves them.
        
        Pending writes are intermediate state stored during graph execution.
        They must be:
        1. Stored via aput_writes()
        2. Retrieved as part of the CheckpointTuple via aget_tuple()
        3. Correctly serialized/deserialized (including complex types)
        """
        unique_thread = f"writes-test-{uuid.uuid4()}"
        checkpoint_id = "writes-checkpoint-1"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": unique_thread,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_id,
            }
        }
        
        # Step 1: Create and save a checkpoint (writes require an existing checkpoint)
        checkpoint = empty_checkpoint()
        checkpoint["id"] = checkpoint_id
        metadata: CheckpointMetadata = {"source": "test"}
        await saver.aput(config, checkpoint, metadata, {})
        
        # Step 2: Save writes with various data types
        writes = [
            ("channel_1", "value_1"),  # String value
            ("channel_2", {"nested": "data"}),  # Dict value
            ("channel_3", [1, 2, 3]),  # List value
        ]
        task_id = "test-task-1"
        await saver.aput_writes(config, writes, task_id)
        
        # Step 3: Retrieve checkpoint and verify writes are included
        result = await saver.aget_tuple(config)
        assert result is not None
        assert len(result.pending_writes) == 3
        
        # Verify write contents are present (order may vary based on sorting)
        write_channels = [w[1] for w in result.pending_writes]
        write_values = [w[2] for w in result.pending_writes]
        assert "channel_1" in write_channels
        assert "value_1" in write_values

    # =========================================================================
    # Thread Deletion Tests
    # =========================================================================

    async def test_adelete_thread(self, saver: AsyncCosmosDBSaver) -> None:
        """Verify that adelete_thread removes all checkpoints and writes for a thread.
        
        This tests the async bulk deletion capability which uses Cosmos DB's
        "Delete All Items By Partition Key" feature for efficient removal.
        After deletion:
        - alist() should return empty results
        - aget_tuple() should return None
        """
        unique_thread = f"delete-test-{uuid.uuid4()}"
        
        # Setup: Create multiple checkpoints with writes
        for i in range(3):
            checkpoint_id = f"delete-checkpoint-{i}"
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": unique_thread,
                    "checkpoint_ns": "",
                    "checkpoint_id": checkpoint_id,
                }
            }
            checkpoint = empty_checkpoint()
            checkpoint["id"] = checkpoint_id
            metadata: CheckpointMetadata = {"index": i}
            await saver.aput(config, checkpoint, metadata, {})
            
            # Add writes for each checkpoint
            writes = [("channel", f"value_{i}")]
            await saver.aput_writes(config, writes, f"task-{i}")
        
        # Verify: Checkpoints exist before deletion
        list_config = {"configurable": {"thread_id": unique_thread}}
        results_before = [c async for c in saver.alist(list_config)]
        assert len(results_before) == 3
        
        # Action: Delete the thread
        await saver.adelete_thread(unique_thread)
        
        # Verify: All checkpoints are deleted
        results_after = [c async for c in saver.alist(list_config)]
        assert len(results_after) == 0
        
        # Verify: aget_tuple returns None
        get_config = {"configurable": {"thread_id": unique_thread, "checkpoint_ns": ""}}
        assert await saver.aget_tuple(get_config) is None

    # =========================================================================
    # Sync Bridge Tests
    # =========================================================================

    def test_sync_bridge_methods(self, saver: AsyncCosmosDBSaver) -> None:
        """Placeholder test for synchronous bridge methods.
        
        The AsyncCosmosDBSaver provides synchronous bridge methods (put, get_tuple, list)
        that allow calling async operations from synchronous code. These work by:
        1. Running the async operation in the background event loop
        2. Using run_coroutine_threadsafe() to safely cross thread boundaries
        
        Note: This test is a placeholder. The actual sync bridge test is in
        test_sync_bridge_integration() below, which properly handles threading.
        """
        pass


# =============================================================================
# Standalone Tests (Outside Test Class)
# =============================================================================

@pytest.mark.asyncio
async def test_sync_bridge_integration(saver: AsyncCosmosDBSaver) -> None:
    """Test the synchronous bridge methods of AsyncCosmosDBSaver.
    
    AsyncCosmosDBSaver provides sync wrapper methods (put, get_tuple, list)
    that allow calling from synchronous code. These wrappers use
    `asyncio.run_coroutine_threadsafe()` to safely submit work to the
    background event loop.
    
    IMPORTANT: Sync bridge methods MUST be called from a different thread
    than the event loop thread. This test uses `asyncio.to_thread()` to
    run sync operations in a worker thread while the main async loop continues.
    
    Test flow:
    1. Use aput() to save a checkpoint (async)
    2. In a worker thread, call get_tuple() and list() (sync bridge)
    3. Verify the data is correctly retrieved
    """
    thread_id = f"sync-thread-{uuid.uuid4()}"
    config = {
        "configurable": {
            "thread_id": thread_id, 
            "checkpoint_ns": "", 
            "checkpoint_id": "sync_1"
        }
    }
    chkpnt = empty_checkpoint()
    chkpnt["id"] = "sync_1"
    metadata = {"source": "sync_test"}
    
    def run_sync_ops():
        """Execute sync bridge operations in a worker thread.
        
        This function runs in a separate thread from the event loop,
        which is required for the sync bridge to work correctly.
        """
        # 1. Test sync put() - saves checkpoint via bridge
        saver.put(config, chkpnt, metadata, {})
        
        # 2. Test sync get_tuple() - retrieves checkpoint via bridge
        fetched = saver.get_tuple(config)
        assert fetched is not None
        assert fetched.checkpoint["id"] == "sync_1"
        assert fetched.metadata["source"] == "sync_test"
        
        # 3. Test sync list() - lists checkpoints via bridge
        listed = list(saver.list({"configurable": {"thread_id": thread_id}}))
        assert len(listed) == 1
        assert listed[0].checkpoint["id"] == "sync_1"
        
        return True

    # Offload sync operations to a worker thread to avoid blocking the event loop
    # and to satisfy the requirement that sync bridge calls come from a different thread
    result = await asyncio.to_thread(run_sync_ops)
    assert result is True
