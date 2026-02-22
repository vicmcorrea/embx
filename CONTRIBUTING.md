# Contributing

Thank you for contributing to embx.

## Development setup

1. Clone the repository.
2. Install dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Local quality checks

Run all checks before opening a pull request:

```bash
ruff check .
pytest --cov=embx --cov-report=term-missing --cov-fail-under=70 -q
```

## Code style

- Use type hints for public function signatures.
- Keep CLI behavior script-friendly; avoid breaking command contracts.
- Add tests for every new feature and bug fix.

## Commit messages

Use Conventional Commits:

- `feat(scope): message`
- `fix(scope): message`
- `refactor(scope): message`
- `test(scope): message`
- `ci(scope): message`

## Pull requests

Include the following in each pull request:

1. A short problem statement.
2. A concise description of the change.
3. Test evidence (command + output summary).
