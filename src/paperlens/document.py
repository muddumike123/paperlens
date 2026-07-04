"""Typed data structures shared across the processing pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


BlockKind = Literal["title", "paragraph", "list", "table", "key_value", "unknown"]


class Entity(BaseModel):
    """A normalized semantic object extracted from document text."""

    kind: str
    value: str
    page_number: int
    start: int | None = None
    end: int | None = None
    confidence: float = 1.0


class Classification(BaseModel):
    """Predicted document class."""

    label: str
    confidence: float
    scores: dict[str, float] = Field(default_factory=dict)
    method: str = "keyword"


class Block(BaseModel):
    """A coarse layout block detected from page text."""

    kind: BlockKind = "unknown"
    text: str
    page_number: int
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Page(BaseModel):
    """A single page of a document."""

    number: int
    text: str = ""
    width: float | None = None
    height: float | None = None
    blocks: list[Block] = Field(default_factory=list)


class Document(BaseModel):
    """Complete document representation."""

    source: str
    mime_type: str = "application/octet-stream"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pages: list[Page] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    classification: Classification | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @classmethod
    def from_text(cls, text: str, source: str = "memory") -> "Document":
        return cls(
            source=source,
            mime_type="text/plain",
            pages=[Page(number=1, text=text)],
            metadata={"filename": Path(source).name if source != "memory" else source},
        )
