.PHONY: install test lint typecheck api

install:
	pip install -e ".[dev,api,pdf,ocr]"

test:
	pytest

lint:
	ruff check .

typecheck:
	mypy src/paperlens

api:
	uvicorn paperlens.api:app --reload
