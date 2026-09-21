"""CLI for inspecting and smoke-testing embedders.

    python -m tools.embedders --list
    python -m tools.embedders --smoke
    python -m tools.embedders --smoke --set embed.batch_size=8
"""

from __future__ import annotations

import argparse
import sys

from tools.config import ConfigError, add_config_args, load_config
from tools.embedders.base import EmbedderError, available_embedders, get_embedder


def _cmd_list() -> int:
    names = available_embedders()
    if not names:
        print("No embedders registered.", file=sys.stderr)
        return 1
    print("Registered embedders:")
    for name in names:
        print(f"  {name}")
    print(
        "\nOnly the seam is built for hosted providers. To add one, implement "
        "the\ncontract in tools/embedders/base.py and decorate its factory with "
        "@register_embedder."
    )
    return 0


def _cmd_smoke(config_path: str, overrides: list[str]) -> int:
    """Embed three strings and verify the interface contract holds."""
    import numpy as np

    from tools.runlog import RunLog

    config = load_config(config_path, overrides=overrides)
    texts = [
        "The Company's total net sales increased 2% during fiscal 2024.",
        "We face risks related to supply chain concentration in Asia.",
        "Item 7. Management's Discussion and Analysis of Financial Condition.",
    ]

    with RunLog(config, run_id="embedder-smoke", log_dir="temp") as log:
        with log.stage("embed_smoke") as stage:
            embedder = get_embedder(config, stage)
            print(f"embedder: {embedder!r}")
            for key, value in embedder.describe().items():
                print(f"  {key}: {value}")
            print(f"\nembedding {len(texts)} documents...")
            docs = embedder.embed_documents(texts, stage)
            query = embedder.embed_query("How did net sales change?", stage)

    checks = [
        ("documents shape is (3, dim)", docs.shape == (3, embedder.dim)),
        ("documents dtype is float32", docs.dtype == np.float32),
        ("query shape is (dim,)", query.shape == (embedder.dim,)),
        ("query dtype is float32", query.dtype == np.float32),
        ("dim was declared before the call", isinstance(embedder.dim, int)),
    ]
    if embedder.normalized:
        norms = np.linalg.norm(docs, axis=1)
        checks.append(
            ("vectors are unit norm as declared", bool(np.allclose(norms, 1.0, atol=1e-4)))
        )
        checks.append(
            (
                "query is unit norm as declared",
                bool(np.allclose(np.linalg.norm(query), 1.0, atol=1e-4)),
            )
        )

    # Retrieval sanity: the sales query should rank the sales sentence first.
    scores = docs @ query
    checks.append(("nearest neighbour is the sales sentence", int(scores.argmax()) == 0))

    print("\n--- similarity to query ---")
    for text, score in zip(texts, scores):
        print(f"  {score:+.4f}  {text[:60]}")

    print("\n--- checks ---")
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= passed
    print("\nSMOKE " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.embedders",
        description="Inspect and smoke-test embedding backends.",
    )
    add_config_args(parser)
    parser.add_argument("--list", action="store_true", help="list registered embedders")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="embed three strings and verify the contract (requires deps installed)",
    )
    args = parser.parse_args(argv)

    if args.list:
        return _cmd_list()
    if args.smoke:
        try:
            return _cmd_smoke(args.config, args.overrides)
        except (ConfigError, EmbedderError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        except ImportError as exc:
            print(f"ERROR: missing dependency ({exc}). Run: uv sync", file=sys.stderr)
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
