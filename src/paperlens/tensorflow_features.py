"""TensorFlow-backed feature extraction for document text."""

from __future__ import annotations

import json
import subprocess
import sys

from paperlens.document import Document


class TensorFlowUnavailableError(RuntimeError):
    """Raised when TensorFlow features are requested without TensorFlow installed."""


def add_tensorflow_features(document: Document) -> Document:
    """Attach simple TensorFlow-computed text statistics to document metadata.

    This intentionally uses TensorFlow for deterministic feature extraction instead
    of downloading a large model. It keeps the project easy to deploy while leaving
    a real TensorFlow integration point for later classifiers.
    """

    result = subprocess.run(
        [sys.executable, "-m", "paperlens._tensorflow_worker"],
        input=document.text,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or "TensorFlow worker exited unexpectedly."
        raise TensorFlowUnavailableError(
            "TensorFlow support requires a working install: pip install paperlens[ml]\n"
            f"Worker error: {details}"
        )

    document.metadata["tensorflow_features"] = json.loads(result.stdout)
    return document
