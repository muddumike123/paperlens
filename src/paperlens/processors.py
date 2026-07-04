"""Reusable pipeline processors."""

from __future__ import annotations

import re
from collections.abc import Callable

from paperlens.document import Block, Document, Entity


Processor = Callable[[Document], Document]


ENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "url": re.compile(r"\bhttps?://[^\s<>()]+", re.IGNORECASE),
    "phone": re.compile(
        r"(?<![\w-])(?:\+?\d{1,3}[\s().-])?(?:\(?\d{3}\)?[\s.-])\d{3}[\s.-]\d{4}(?![\w-])"
    ),
    "money": re.compile(r"(?:[$€£]\s?\d[\d,]*(?:\.\d{2})?|\d[\d,]*(?:\.\d{2})?\s?(?:USD|EUR|GBP))"),
    "date": re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"),
    "document_id": re.compile(
        r"\b(?:INV|PO|SO|DOC|ID)[-:\s]?[A-Z0-9][A-Z0-9-]*\d[A-Z0-9-]*\b",
        re.IGNORECASE,
    ),
}


def normalize_whitespace(document: Document) -> Document:
    """Normalize page text while preserving paragraph breaks."""

    for page in document.pages:
        lines = [" ".join(line.split()) for line in page.text.splitlines()]
        page.text = "\n".join(line for line in lines if line)
    return document


def detect_blocks(document: Document) -> Document:
    """Create simple layout blocks from text lines."""

    for page in document.pages:
        blocks: list[Block] = []
        for raw_line in page.text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            kind = _classify_line(line)
            blocks.append(Block(kind=kind, text=line, page_number=page.number))
        page.blocks = blocks
    return document


def extract_entities(document: Document) -> Document:
    """Extract regex-based entities from each page."""

    entities: list[Entity] = []
    for page in document.pages:
        for kind, pattern in ENTITY_PATTERNS.items():
            for match in pattern.finditer(page.text):
                entities.append(
                    Entity(
                        kind=kind,
                        value=match.group(0).strip(),
                        page_number=page.number,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95,
                    )
                )
    document.entities = entities
    return document


def summarize(document: Document) -> Document:
    """Attach lightweight statistics to the document metadata."""

    words = re.findall(r"\b\w+\b", document.text)
    document.metadata.update(
        {
            "page_count": document.page_count,
            "word_count": len(words),
            "entity_count": len(document.entities),
            "block_count": sum(len(page.blocks) for page in document.pages),
        }
    )
    return document


def _classify_line(line: str) -> str:
    if len(line) <= 80 and line == line.upper() and any(char.isalpha() for char in line):
        return "title"
    if re.search(r"\s{2,}|\t|\|", line):
        return "table"
    if re.match(r"^[-*•]\s+", line) or re.match(r"^\d+[.)]\s+", line):
        return "list"
    if ":" in line and len(line.split(":", 1)[0]) <= 40:
        return "key_value"
    return "paragraph"
