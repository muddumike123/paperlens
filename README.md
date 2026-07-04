# PaperLens by Saaketh Mudunuri

## Features

- PDF and text ingestion
- Optional OCR for images via `pytesseract`
- Page/block/entity data model
- Document classification for invoices, receipts, resumes, contracts, bank
  statements, medical records, reports, letters, and unknown files
- Regex-based entity extraction for emails, phone numbers, dates, URLs, money,
  and invoice-like IDs
- Optional TensorFlow-backed text feature extraction
- Markdown and JSON export
- CLI for batch analysis
- FastAPI upload UI and `/classify` endpoint
- Extensible pipeline components

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[api,pdf,ocr,ml,dev]"
paperlens analyze examples/sample_invoice.txt --format markdown
pytest
```

If you only need text-file support, install without extras:

```bash
pip install -e .
```

## CLI

Analyze a file and print JSON:

```bash
paperlens analyze path/to/document.pdf
```

Export Markdown:

```bash
paperlens analyze path/to/document.txt --format markdown --output report.md
```

Enable TensorFlow features:

```bash
pip install -e ".[ml]"
paperlens analyze examples/sample_invoice.txt --tensorflow
```

Show basic document metadata:

```bash
paperlens inspect path/to/document.txt
```

## API Server

```bash
pip install -e ".[api,pdf,ocr,ml]"
uvicorn paperlens.api:app --reload
```

Then open the upload UI:

```text
http://127.0.0.1:8000
```

Or call the classifier endpoint directly:

```bash
curl -F "file=@examples/sample_invoice.txt" http://127.0.0.1:8000/classify
```

The response includes a direct answer:

```json
{
  "document_type": "invoice",
  "answer": "This looks like a invoice."
}
```

## Docker

```bash
docker build -f docker/Dockerfile -t paperlens .
docker run --rm -p 8000:8000 paperlens
```

## Project Layout

```text
src/paperlens/
  api.py          FastAPI app
  cli.py          command-line interface
  document.py     typed document data model
  exporters.py    JSON and Markdown exporters
  io.py           file loading and text extraction
  pipeline.py     processing pipeline
  processors.py   entity and layout processors
  tensorflow_features.py TensorFlow-backed feature processor
```

## Notes

PaperLens keeps the core small and dependency-light. PDF, OCR, and TensorFlow
support are optional extras, so the package can run in constrained deployment
environments and grow as needed.

The TensorFlow step runs in an isolated worker process. That means a broken local
TensorFlow native install returns a clean CLI/API error instead of crashing the
main PaperLens process.

## License

MIT
