# Contributing

Thanks for improving PaperLens.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api,pdf,ocr]"
pytest
ruff check .
mypy src/paperlens
```

## Pull Requests

- Keep changes focused.
- Add or update tests for behavior changes.
- Run the test suite before opening a pull request.
- Document new CLI or API behavior in `README.md`.
