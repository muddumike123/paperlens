"""Command-line interface for PaperLens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paperlens.exporters import to_json, to_markdown
from paperlens.io import DocumentLoadError, load_document
from paperlens.pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paperlens", description="Analyze documents.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze a document.")
    analyze.add_argument("path", type=Path)
    analyze.add_argument("--format", choices=["json", "markdown"], default="json")
    analyze.add_argument("--output", type=Path)
    analyze.add_argument(
        "--tensorflow",
        action="store_true",
        help="Enable TensorFlow-backed text feature extraction.",
    )

    inspect_cmd = subparsers.add_parser("inspect", help="Show document metadata.")
    inspect_cmd.add_argument("path", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        document = load_document(args.path)
    except DocumentLoadError as exc:
        print(f"paperlens: {exc}", file=sys.stderr)
        return 2

    if args.command == "inspect":
        print(to_json(document))
        return 0

    try:
        analyzed = Pipeline(use_tensorflow=args.tensorflow).run(document)
    except RuntimeError as exc:
        print(f"paperlens: {exc}", file=sys.stderr)
        return 2
    rendered = to_markdown(analyzed) if args.format == "markdown" else to_json(analyzed)

    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
