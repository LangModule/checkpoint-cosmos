# LangGraph Cosmos DB Checkpoint

A LangGraph checkpoint saver implementation for Azure Cosmos DB, providing durable persistence for LangGraph workflow state with both synchronous and asynchronous support.

> **Note:** This implementation is based on [langgraph-checkpoint-sqlite](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-sqlite), adapted for Azure Cosmos DB.

## Features

- **Synchronous & Asynchronous APIs**: Full support for both sync (`CosmosDBSaver`) and async (`AsyncCosmosDBSaver`) operations
- **Tip Document Optimization**: O(1) access to the latest checkpoint without expensive queries
- **Transactional Consistency**: Atomic batch operations ensure checkpoint and metadata are always in sync
- **Efficient Partitioning**: Separate partitions for checkpoints and writes optimize read/write performance
- **SQL Injection Prevention**: All queries use parameterized inputs for security
- **Azure Identity Support**: Works with connection strings or `DefaultAzureCredential` for keyless authentication

## Installation

```bash
pip install langgraph-checkpoint-cosmos
```

## Requirements

- Python >= 3.11
- Azure Cosmos DB account with a database and container
- Container must have partition key set to `/partition_key`

## Usage

### Synchronous

```python
from langgraph_checkpoint_cosmos import CosmosDBSaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

# You can also use CosmosDBSaver(container) constructor with a pre-configured container
with CosmosDBSaver.from_conn_info(
    endpoint="<your-endpoint>",
    credential="<your-key>",  # Or use DefaultAzureCredential()
    database_name="<your-db>",
    container_name="<your-container>"
) as checkpointer:
    checkpoint = {
        "v": 1,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "channel_values": {
            "my_key": "meow",
            "node": "node"
        },
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {
                "__start__": 1
            },
            "node": {
                "start:node": 2
            }
        },
        "pending_sends": [], 
    }

    # store checkpoint
    checkpointer.put(write_config, checkpoint, {}, {})

    # load checkpoint
    checkpointer.get(read_config)

    # list checkpoints
    list(checkpointer.list(read_config))
```

### Async

```python
from langgraph_checkpoint_cosmos.aio import AsyncCosmosDBSaver

# You can also use explicit constructor AsyncCosmosDBSaver(container) 
async with AsyncCosmosDBSaver.from_conn_info(
    endpoint="<your-endpoint>",
    credential="<your-key>",  # Or use DefaultAzureCredential()
    database_name="<your-db>",
    container_name="<your-container>"
) as checkpointer:
    checkpoint = {
        "v": 1,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "channel_values": {
            "my_key": "meow",
            "node": "node"
        },
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {
                "__start__": 1
            },
            "node": {
                "start:node": 2
            }
        },
        "pending_sends": [],
    }

    # store checkpoint
    await checkpointer.aput(write_config, checkpoint, {}, {})

    # load checkpoint
    await checkpointer.aget(read_config)

    # list checkpoints
    [c async for c in checkpointer.alist(read_config)]
```

### Using with LangGraph StateGraph

The primary use case is as a checkpointer for LangGraph workflows:

```python
from langgraph.graph import StateGraph
from langgraph_checkpoint_cosmos import CosmosDBSaver

# Define your graph
builder = StateGraph(int)
builder.add_node("add_one", lambda x: x + 1)
builder.set_entry_point("add_one")
builder.set_finish_point("add_one")

# Create checkpointer
with CosmosDBSaver.from_conn_info(
    endpoint="https://your-account.documents.azure.com:443/",
    credential="your-key",
    database_name="langgraph",
    container_name="checkpoints"
) as checkpointer:
    # Compile graph with checkpointer
    graph = builder.compile(checkpointer=checkpointer)
    
    # Run with thread_id for persistence
    config = {"configurable": {"thread_id": "user-123"}}
    result = graph.invoke(1, config)
    
    # Later, resume from the same thread
    state = graph.get_state(config)
    print(state.values)  # Output: 2
```

## Azure Cosmos DB Setup

1. Create an Azure Cosmos DB account (NoSQL API)
2. Create a database (e.g., `langgraph`)
3. Create a container with:
   - Partition key: `/partition_key`
   - (Optional) Enable "Delete All Items By Partition Key" for efficient thread deletion

### Using DefaultAzureCredential (Recommended for Production)

```python
from azure.identity import DefaultAzureCredential

with CosmosDBSaver.from_conn_info(
    endpoint="https://your-account.documents.azure.com:443/",
    credential=DefaultAzureCredential(),  # Keyless authentication
    database_name="langgraph",
    container_name="checkpoints"
) as checkpointer:
    ...
```

## Environment Variables

For testing, set the following environment variables:

```bash
export COSMOS_DB_ENDPOINT="https://your-account.documents.azure.com:443/"
export COSMOS_DB_KEY="your-primary-key"
export COSMOS_DB_NAME="langgraph"
export COSMOS_DB_CONTAINER="checkpoints"
```

## API Reference

### CosmosDBSaver (Synchronous)

| Method | Description |
|--------|-------------|
| `get(config)` | Get checkpoint values (convenience method) |
| `get_tuple(config)` | Get checkpoint with full metadata |
| `put(config, checkpoint, metadata, new_versions)` | Save a checkpoint |
| `list(config, *, filter, before, limit)` | List/search checkpoints |
| `put_writes(config, writes, task_id)` | Store pending writes |
| `delete_thread(thread_id)` | Delete all data for a thread |

### AsyncCosmosDBSaver (Asynchronous)

| Method | Description |
|--------|-------------|
| `aget(config)` | Get checkpoint values (async, convenience) |
| `aget_tuple(config)` | Get checkpoint with full metadata (async) |
| `aput(config, checkpoint, metadata, new_versions)` | Save a checkpoint (async) |
| `alist(config, *, filter, before, limit)` | List/search checkpoints (async generator) |
| `aput_writes(config, writes, task_id)` | Store pending writes (async) |
| `adelete_thread(thread_id)` | Delete all data for a thread (async) |

The async saver also provides synchronous bridge methods (`put`, `get_tuple`, `list`) that can be called from synchronous code when needed.

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (requires Cosmos DB connection)
pytest tests/ -v
```

## Acknowledgments

This project is based on [langgraph-checkpoint-sqlite](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-sqlite) from the LangChain team. The core architecture, serialization patterns, and checkpoint management logic were adapted for Azure Cosmos DB.

## License

MIT