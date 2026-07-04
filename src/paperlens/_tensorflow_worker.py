"""Subprocess worker for TensorFlow feature extraction."""

from __future__ import annotations

import json
import math
import re
import sys


def main() -> int:
    import tensorflow as tf

    text = sys.stdin.read()
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        print(
            json.dumps(
                {
                    "backend": "tensorflow",
                    "token_count": 0,
                    "unique_token_count": 0,
                    "lexical_diversity": 0.0,
                    "mean_token_length": 0.0,
                    "token_length_stddev": 0.0,
                }
            )
        )
        return 0

    token_lengths = tf.constant([len(token) for token in tokens], dtype=tf.float32)
    unique_count = len(set(tokens))
    token_count = len(tokens)
    mean_length = tf.reduce_mean(token_lengths)
    variance = tf.reduce_mean(tf.square(token_lengths - mean_length))

    print(
        json.dumps(
            {
                "backend": "tensorflow",
                "token_count": token_count,
                "unique_token_count": unique_count,
                "lexical_diversity": round(unique_count / token_count, 4),
                "mean_token_length": round(float(mean_length.numpy()), 4),
                "token_length_stddev": round(math.sqrt(float(variance.numpy())), 4),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
