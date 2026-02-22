# embx

`embx` is a multi-provider embeddings CLI for quick experimentation and automation.

## Why this exists

- Use one command surface across OpenAI, Voyage, and Ollama.
- Keep output script-friendly (`json`, `jsonl`) but readable by default.
- Add cache and config layering so provider switching is painless.

## Install

```bash
pip install -e .
```

## Quick start

```bash
# Show providers
embx providers

# Initialize local config
embx config init

# Embed a single text (argument)
embx embed "vector databases are useful"

# Embed from stdin
printf "semantic search" | embx embed --format json

# Batch embed line-delimited file
embx batch inputs.txt --format jsonl --output outputs.jsonl

# Compare providers for the same input
embx compare "semantic retrieval" --providers openai,voyage,ollama

# Compare in machine-readable mode
embx compare "semantic retrieval" --format json --output compare.json

# Rank providers by latency or cost
embx compare "semantic retrieval" --providers openai,voyage --rank-by latency

# Rank providers by embedding agreement quality
embx compare "semantic retrieval" --providers openai,voyage,ollama --rank-by quality

# Emit CSV for spreadsheets or BI tools
embx compare "semantic retrieval" --providers openai,voyage --format csv
embx batch inputs.txt --format csv --output embeddings.csv
```

## Config precedence

1. CLI flags
2. Environment variables (`EMBX_*`)
3. Config file (`~/.config/embx/config.json`)
4. Built-in defaults

## Shell completions

Typer provides completion out of the box:

```bash
embx --install-completion
embx --show-completion
```

## Current status

- Core command scaffolding is implemented.
- Provider integrations are intentionally minimal and extensible.
- `compare` command is available for side-by-side provider checks.
- Docs in `docs/` explain architecture and roadmap.
