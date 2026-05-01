# Grinning Cat Weaviate

Introduce Weaviate handler to the Grinning Cat.

This plugin allows the Cheshire Cat to use [Weaviate](https://weaviate.io) as its vector database backend. It implements the `BaseVectorDatabaseHandler` using the Weaviate Python SDK v4 (async).

## Features

- **Full Async Support**: Every network call is awaited directly, ensuring compatibility with the Cat's async architecture.
- **Tenant Isolation**: Uses a `tenant_id` property to isolate data between different agents/tenants, mirroring the Qdrant approach.
- **Hybrid Search**: Leverages Weaviate's native BM25 + vector search with Relative Score Fusion.
- **Efficient Batching**: Uses gRPC `insert_many` for fast data ingestion.
- **Snapshot Support**: Integrates with Weaviate's backup-filesystem module for memory snapshots.

## Configuration

You can configure the plugin via the following environment variables or through the Cat's settings GUI:

- `CAT_WEAVIATE_HOST`: The hostname of your Weaviate instance (default: `localhost`).
- `CAT_WEAVIATE_API_KEY`: Optional API key for authenticated deployments.

## Installation

1. Install the plugin from the Cheshire Cat plugin gallery or by cloning this repository into the `cat/plugins` folder.
2. Ensure you have a Weaviate instance running.
3. Configure the connection settings.

## Technical Details

- **Collection Names**: Collection names are automatically capitalized to meet Weaviate's requirements (e.g., `declarative` becomes `Declarative`).
- **Score Mapping**: Weaviate returns cosine distance (0.0 = perfect match). The handler converts this to a similarity score (`1.0 - distance`) to maintain consistency with other handlers.
- **Schema**: The plugin creates a schema with properties for `page_content`, `tenant_id`, `metadata_json`, `metadata__source`, and `embedder_name`.
