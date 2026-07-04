"""Export analyzed documents to common formats."""

from __future__ import annotations

import json

from paperlens.document import Document


def to_json(document: Document, *, indent: int = 2) -> str:
    return json.dumps(document.model_dump(mode="json"), indent=indent, ensure_ascii=False)


def to_markdown(document: Document) -> str:
    lines = [
        f"# Analysis: {document.metadata.get('filename', document.source)}",
        "",
        "## Summary",
        "",
        f"- Pages: {document.page_count}",
        f"- Words: {document.metadata.get('word_count', 0)}",
        f"- Blocks: {document.metadata.get('block_count', 0)}",
        f"- Entities: {len(document.entities)}",
        "",
    ]

    if document.classification:
        lines.extend(
            [
                "## Classification",
                "",
                f"- Label: **{document.classification.label}**",
                f"- Confidence: {document.classification.confidence:.2f}",
                "",
            ]
        )

    if document.entities:
        lines.extend(["## Entities", ""])
        for entity in document.entities:
            lines.append(f"- **{entity.kind}**: `{entity.value}` (page {entity.page_number})")
        lines.append("")

    lines.extend(["## Pages", ""])
    for page in document.pages:
        lines.extend([f"### Page {page.number}", ""])
        for block in page.blocks:
            if block.kind == "title":
                lines.append(f"#### {block.text.title()}")
            elif block.kind == "list":
                lines.append(block.text)
            else:
                lines.append(block.text)
            lines.append("")

    return "\n".join(lines).strip() + "\n"
