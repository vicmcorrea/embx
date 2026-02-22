# embx release checklist

## Version bump

1. Update `version` in `pyproject.toml`.
2. Update `__version__` in `src/embx/__init__.py`.
3. Update user-facing docs if command behavior changed.

## Quality gates

Run before tagging:

```bash
ruff check .
pytest --cov=embx --cov-report=term-missing --cov-fail-under=70 -q
python -m build
python -m twine check dist/*
```

## Publish flow (Trusted Publisher)

1. Commit and push release changes to `main`.
2. Create and push version tag:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

3. Verify GitHub Actions publish run succeeds (`.github/workflows/publish.yml`).
4. Verify package install from PyPI in a clean virtual environment:

```bash
python -m venv /tmp/embx-release-check
/tmp/embx-release-check/bin/pip install --upgrade pip
/tmp/embx-release-check/bin/pip install --no-cache-dir --index-url https://pypi.org/simple embx-cli==X.Y.Z
/tmp/embx-release-check/bin/embx --version
rm -rf /tmp/embx-release-check
```

## Post-release

1. Create GitHub release notes for the new tag.
2. Verify README rendering on PyPI.
3. Start next patch version changes on `main`.
