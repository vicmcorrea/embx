# embx

`embx` is a multi-provider embeddings CLI for quick experimentation and automation.

## why this exists

- use one command surface across OpenAI, OpenRouter, Voyage, and Ollama.
- keep output script-friendly (`json`, `jsonl`) but readable by default.
- add cache and config layering so provider switching is painless.
- friends wanted to use it and try it out.

## Install

```bash
pip install embx-cli
```

For local development from source:

```bash
pip install -e ".[dev]"
```

## Quick start

```bash
# One-command guided flow: connect, pick model, run embedding
embx quickstart

# One-command non-interactive flow
embx quickstart "semantic retrieval" --provider openrouter --connect --api-key "$EMBX_OPENROUTER_API_KEY" --non-interactive --format json

# Show providers
embx providers

# Interactive setup flow
embx connect

# Configure multiple providers in one run and test connectivity
embx connect --all --test

# Test one provider in non-interactive mode
embx connect --provider openrouter --api-key "$EMBX_OPENROUTER_API_KEY" --non-interactive --test

# Configure HuggingFace provider
embx connect --provider huggingface --api-key "$EMBX_HUGGINGFACE_API_KEY" --non-interactive

# Configure HuggingFace and default to local cached model discovery
embx connect --provider huggingface --api-key "$EMBX_HUGGINGFACE_API_KEY" --model-source local --non-interactive

# List available embedding models
embx models --provider openrouter
embx models --provider openrouter --format json

# HuggingFace model discovery (remote/local/all)
embx models --provider huggingface --source remote --search mini --limit 10
embx models --provider huggingface --source local
embx models --provider huggingface --source all

# Select one model id for shell pipelines
embx models --provider openrouter --pick 1
embx models --provider openrouter --choose

# Interactive model browsing
embx models --interactive

# Interactive config editing
embx config set
embx config set --key default_provider --value huggingface --non-interactive

# Check provider configuration and readiness
embx doctor
embx doctor --only-configured --check-network
embx doctor --only-configured --check-auth

# Optional: initialize config manually
embx config init

# Embed a single text (argument)
embx embed "vector databases are useful"

# Embed from stdin
printf "semantic search" | embx embed --format json

# Embed as CSV
embx embed "semantic search" --format csv

# Batch embed line-delimited file
embx batch inputs.txt --format jsonl --output outputs.jsonl

# Compare providers for the same input
embx compare "semantic retrieval" --providers openai,openrouter,voyage,ollama

# Compare in machine-readable mode
embx compare "semantic retrieval" --format json --output compare.json

# Rank providers by latency or cost
embx compare "semantic retrieval" --providers openai,voyage --rank-by latency

# Rank providers by embedding agreement quality
embx compare "semantic retrieval" --providers openai,voyage,ollama --rank-by quality

# Emit CSV for spreadsheets or BI tools
embx compare "semantic retrieval" --providers openai,voyage --format csv
embx batch inputs.txt --format csv --output embeddings.csv

# Skip providers without configured credentials
embx compare "semantic retrieval" --providers openai,voyage,ollama --only-configured

# Show top 2 ranked providers and hide failed rows
embx compare "semantic retrieval" --rank-by quality --top 2 --hide-errors

# Markdown table output
embx compare "semantic retrieval" --format md

# Retries with backoff for transient provider failures
embx embed "semantic retrieval" --provider openrouter --retries 2 --retry-backoff 0.2

# HuggingFace embeddings inference
embx embed "semantic retrieval" --provider huggingface --model sentence-transformers/all-MiniLM-L6-v2
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
- Docs in `docs/` explain architecture, roadmap, and release workflow.
