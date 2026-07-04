from paperlens.document import Document
from paperlens.pipeline import Pipeline
from paperlens.tensorflow_features import add_tensorflow_features


def test_pipeline_extracts_entities_and_blocks() -> None:
    document = Document.from_text(
        "ACME CONSULTING\nInvoice: INV-2026-0715\nEmail: team@example.com\nTotal: $42.00"
    )

    analyzed = Pipeline().run(document)

    assert analyzed.metadata["page_count"] == 1
    assert analyzed.metadata["block_count"] == 4
    assert {entity.kind for entity in analyzed.entities} >= {"email", "money", "document_id"}
    assert analyzed.classification
    assert analyzed.classification.label == "invoice"


def test_document_text_combines_pages() -> None:
    document = Document.from_text("Hello world")

    assert document.text == "Hello world"
    assert document.page_count == 1


def test_entity_patterns_avoid_common_false_positives() -> None:
    document = Document.from_text(
        "Invoice: INV-2026-0715\nDate: 2026-07-01\n"
        "Payment URL: https://pay.example/invoices/INV-2026-0715\n"
        "Phone: +1 555 202 0199"
    )

    analyzed = Pipeline().run(document)
    phones = [entity.value for entity in analyzed.entities if entity.kind == "phone"]
    document_ids = [entity.value for entity in analyzed.entities if entity.kind == "document_id"]

    assert phones == ["+1 555 202 0199"]
    assert "Invoice" not in document_ids
    assert "invoices" not in document_ids


def test_tensorflow_feature_processor_can_be_mocked(monkeypatch) -> None:
    def fake_tensorflow_features(document: Document) -> Document:
        document.metadata["tensorflow_features"] = {"backend": "tensorflow"}
        return document

    monkeypatch.setattr("paperlens.pipeline.add_tensorflow_features", fake_tensorflow_features)

    analyzed = Pipeline(use_tensorflow=True).run(Document.from_text("Tiny test"))

    assert analyzed.metadata["tensorflow_features"]["backend"] == "tensorflow"
    assert add_tensorflow_features is not None


def test_pipeline_classifies_resume() -> None:
    document = Document.from_text(
        "Resume\nExperience: Python developer\nEducation: Computer Science\nSkills: APIs"
    )

    analyzed = Pipeline().run(document)

    assert analyzed.classification
    assert analyzed.classification.label == "resume"
