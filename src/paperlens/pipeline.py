"""Composable document processing pipeline."""

from __future__ import annotations

from collections.abc import Iterable

from paperlens.classifier import classify_document
from paperlens.document import Document
from paperlens.processors import (
    Processor,
    detect_blocks,
    extract_entities,
    normalize_whitespace,
    summarize,
)
from paperlens.tensorflow_features import add_tensorflow_features


class Pipeline:
    """Run processors over a document in sequence."""

    def __init__(
        self,
        processors: Iterable[Processor] | None = None,
        *,
        use_tensorflow: bool = False,
    ) -> None:
        self.processors = list(processors or default_processors(use_tensorflow=use_tensorflow))

    def run(self, document: Document) -> Document:
        for processor in self.processors:
            document = processor(document)
        return document


def default_processors(*, use_tensorflow: bool = False) -> list[Processor]:
    processors: list[Processor] = [
        normalize_whitespace,
        detect_blocks,
        extract_entities,
        classify_document,
    ]
    if use_tensorflow:
        processors.append(add_tensorflow_features)
    processors.append(summarize)
    return processors
