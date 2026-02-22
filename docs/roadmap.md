# embx roadmap

## Near-term (v0.2)

- Expand `compare` with aggregate ranking (latency, cost, and dimension normalization score).
- Add token and cost reports for all supported providers.
- Add retry with exponential backoff for transient API failures.
- Add optional concurrency control for large batches.

## Mid-term (v0.3)

- Add vector-db sink plugins (Chroma, Qdrant, Pinecone).
- Add embeddings quality mini-benchmark command.
- Add optional dimensionality projection output for quick visualization.
- Add embedding visualization examples in a 2D plane/graph view so users can inspect how vectors cluster.

## Long-term (v1.0)

- Stable plugin API for providers and exporters.
- Signed releases and Homebrew formula.
- Compatibility matrix tests on macOS, Linux, and Windows.
