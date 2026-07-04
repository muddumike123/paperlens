"""Document classification processors."""

from __future__ import annotations

import re

from paperlens.document import Classification, Document


CLASS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "invoice": (
        "invoice",
        "invoice number",
        "inv-",
        "bill to",
        "payment due",
        "total due",
        "subtotal",
        "tax",
    ),
    "receipt": (
        "receipt",
        "merchant",
        "transaction",
        "cashier",
        "change",
        "total paid",
        "visa",
        "mastercard",
    ),
    "resume": (
        "resume",
        "curriculum vitae",
        "experience",
        "education",
        "skills",
        "projects",
        "linkedin",
    ),
    "contract": (
        "agreement",
        "contract",
        "party",
        "parties",
        "terms and conditions",
        "whereas",
        "signature",
    ),
    "bank_statement": (
        "bank statement",
        "account number",
        "opening balance",
        "closing balance",
        "debit",
        "credit",
        "transaction date",
    ),
    "medical_record": (
        "patient",
        "diagnosis",
        "prescription",
        "doctor",
        "clinical",
        "lab result",
        "medical record",
    ),
    "report": (
        "report",
        "summary",
        "findings",
        "recommendation",
        "analysis",
        "conclusion",
    ),
    "letter": (
        "dear",
        "sincerely",
        "regards",
        "to whom it may concern",
        "subject:",
    ),
}


def classify_document(document: Document) -> Document:
    """Classify a document using transparent keyword scoring."""

    text = re.sub(r"\s+", " ", document.text.lower())
    raw_scores: dict[str, float] = {}
    for label, keywords in CLASS_KEYWORDS.items():
        score = 0.0
        for keyword in keywords:
            count = text.count(keyword)
            if count:
                score += count * _keyword_weight(keyword)
        raw_scores[label] = score

    if not raw_scores or max(raw_scores.values()) == 0:
        label = _fallback_label(document)
        confidence = 0.55 if label != "unknown" else 0.0
        document.classification = Classification(
            label=label,
            confidence=confidence,
            scores={label: 0.0 for label in CLASS_KEYWORDS} | {label: confidence},
        )
        return document

    total = sum(raw_scores.values())
    normalized = {label: round(score / total, 4) for label, score in raw_scores.items()}
    label = max(normalized, key=normalized.get)
    document.classification = Classification(
        label=label,
        confidence=normalized[label],
        scores=normalized,
    )
    return document


def _keyword_weight(keyword: str) -> float:
    if " " in keyword:
        return 2.0
    if keyword.endswith("-") or keyword.endswith(":"):
        return 1.5
    return 1.0


def _fallback_label(document: Document) -> str:
    mime_type = document.mime_type.lower()
    if document.metadata.get("ocr") or mime_type.startswith("image/"):
        return "ocr_scan"
    if document.mime_type == "application/pdf" and not document.text.strip():
        return "scanned_pdf"
    return "unknown"
