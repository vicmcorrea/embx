# embx planning

## Problem statement

Embedding workflows are fragmented across providers. APIs, auth, dimensions, and output formats differ enough that teams either write custom wrappers or stay locked to one provider.

`embx` solves this with a single CLI interface for local and hosted embedding providers.

## Product goals

1. One command surface for major providers (`openai`, `voyage`, `ollama` to start).
2. Machine-first output (`json`, `jsonl`) with human-friendly defaults.
3. Predictable config layering and reproducible execution.
4. Fast startup and composable Unix behavior.

## UX and command model

Command hierarchy:

```text
embx
|- providers
|- embed [text]
|- batch <input_file>
|- compare [text]
`- config
   |- init
   `- show
```

Design notes:

- `embed` supports argument, stdin, and interactive prompt fallback.
- `batch` takes line-delimited input and emits `jsonl` by default.
- `compare` runs the same input across multiple providers with latency and error visibility.
- `providers` makes discovery instant and scriptable (`--json`).
- `config` keeps setup explicit and easy to inspect.

## Configuration strategy

Priority (highest to lowest):

1. CLI flags
2. Environment variables
3. Config file (`~/.config/embx/config.json`)
4. Defaults

Why this order:

- Flags represent immediate user intent.
- Env vars fit CI/CD and secrets management.
- Config file captures local defaults.
- Built-ins keep first-run experience smooth.

## Performance strategy

- Keep imports lightweight at startup.
- Use `httpx` async client for provider calls.
- Use SQLite cache to avoid repeated remote requests.
- Keep rich rendering optional and bounded.

## Initial milestones

1. Scaffold command surface + config system.
2. Add provider abstraction and three providers.
3. Add cache and batch flow.
4. Add provider comparison command for decision-making.
5. Add tests for CLI contract and error paths.
6. Add shell completion and release packaging docs.
