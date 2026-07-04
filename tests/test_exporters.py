from paperlens.document import Document
from paperlens.exporters import to_json, to_markdown
from paperlens.pipeline import Pipeline


def test_exports_json_and_markdown() -> None:
    document = Pipeline().run(Document.from_text("Email: hello@example.com"))

    assert '"entities"' in to_json(document)
    assert "hello@example.com" in to_markdown(document)
