"""Executable evaluation: ``python -m app.evaluation --dataset <path>``."""

from __future__ import annotations

import argparse
import json

from app.bootstrap import build_container


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval/generation quality")
    parser.add_argument("--dataset", required=True, help="Path to JSON/JSONL evaluation dataset")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--generate", action="store_true", help="Also score generation with the LLM")
    parser.add_argument("--out", default=None, help="Write report JSON to this path")
    args = parser.parse_args()

    container = build_container()
    runner = __import__("app.evaluation.runner", fromlist=["EvaluationRunner"]).EvaluationRunner(container)
    report = runner.run(args.dataset, generate=args.generate, top_k=args.top_k)
    print(json.dumps(report, indent=2))
    if args.out:
        import pathlib

        pathlib.Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote report to {args.out}")


if __name__ == "__main__":
    main()