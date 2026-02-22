# embx logic and architecture

## Data flow

```text
CLI command
  -> resolve config (defaults + file + env + flags)
  -> build embedding engine
  -> check cache per input item
  -> send misses to selected provider adapter
  -> normalize responses into EmbeddingResult
  -> write to cache
  -> render pretty output or emit json/jsonl
```

## Module responsibilities

- `src/embx/cli.py`
  - Command definitions, option validation, output formatting, process exit behavior.
- `src/embx/config.py`
  - Config path resolution, layered config loading, bootstrap config creation, secret masking.
- `src/embx/engine.py`
  - Core orchestration: cache lookup, provider dispatch, response ordering.
- `src/embx/cache.py`
  - SQLite-backed vector cache using deterministic content keys.
- `src/embx/providers/*`
  - Provider-specific HTTP API behavior hidden behind one async interface.

## Error handling

- Validation and config errors map to exit code `2`.
- Ctrl+C maps to `130`.
- User-facing errors avoid stack traces and provide direct action hints.

## Output contract

- `embed --format json` returns one JSON object.
- `batch --format json` returns one JSON array.
- `batch --format jsonl` returns one object per line.
- `compare --format json` returns one row per provider with status, latency, and error fields.
- `compare --rank-by latency|cost` sorts successful providers and adds rank metadata.
- `compare` runs provider calls concurrently by default (`--continue-on-error`) for faster side-by-side checks.
- Human mode (`pretty`) prioritizes readability and avoids noisy internals.

## Extension points

1. Add provider: implement `EmbeddingProvider` and register it.
2. Add command: create a Typer command in `cli.py`.
3. Add export format: implement serializer branch in CLI commands.
4. Add eval command: consume `EmbeddingResult` and compute similarity metrics.
