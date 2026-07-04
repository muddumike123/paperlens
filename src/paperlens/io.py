"""Input adapters for PDFs, images, and text files."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from paperlens.document import Document, Page


class DocumentLoadError(RuntimeError):
    """Raised when a document cannot be loaded."""


def load_document(path: str | Path) -> Document:
    """Load a supported file into a :class:`Document`."""

    file_path = Path(path)
    if not file_path.exists():
        raise DocumentLoadError(f"File does not exist: {file_path}")

    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    suffix = file_path.suffix.lower()

    if suffix in {".txt", ".md", ".csv", ".tsv"}:
        return _load_text(file_path, mime_type)
    if suffix == ".pdf":
        return _load_pdf(file_path, mime_type)
    if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
        return _load_image(file_path, mime_type)

    raise DocumentLoadError(f"Unsupported file type: {suffix or '<none>'}")


def _load_text(path: Path, mime_type: str) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace")
    return Document(
        source=str(path),
        mime_type=mime_type,
        pages=[Page(number=1, text=text)],
        metadata={"filename": path.name, "bytes": path.stat().st_size},
    )


def _load_pdf(path: Path, mime_type: str) -> Document:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise DocumentLoadError("PDF support requires: pip install paperlens[pdf]") from exc

    reader = PdfReader(str(path))
    pages: list[Page] = []
    for index, page in enumerate(reader.pages, start=1):
        pages.append(Page(number=index, text=page.extract_text() or ""))

    return Document(
        source=str(path),
        mime_type=mime_type,
        pages=pages,
        metadata={
            "filename": path.name,
            "bytes": path.stat().st_size,
            "pdf_pages": len(reader.pages),
        },
    )


def _load_image(path: Path, mime_type: str) -> Document:
    try:
        from PIL import Image
        import pytesseract
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise DocumentLoadError("OCR support requires: pip install paperlens[ocr]") from exc

    with Image.open(path) as image:
        text = pytesseract.image_to_string(image)
        width, height = image.size

    return Document(
        source=str(path),
        mime_type=mime_type,
        pages=[Page(number=1, text=text, width=float(width), height=float(height))],
        metadata={"filename": path.name, "bytes": path.stat().st_size, "ocr": "tesseract"},
    )
