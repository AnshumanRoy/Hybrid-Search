"""CLI entrypoint — ``python -m app <command>``.

Commands:
    python -m app ingest <path|dir>  – ingest all files in <dir> (or <path>)
    python -m app query "question"   – run a RAG query end-to-end
    python -m app evaluate           – run evaluation against a dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

logger = logging.getLogger(__name__)


def cmd_ingest(args):
    from app.bootstrap import build_container
    container = build_container()
    directory = args.path or str(container.settings.data_dir / "sample")
    results = container.ingestion_service.process_dir(directory)
    if not results:
        print(f"  (no files found under {directory})")
        return
    for r in results:
        status = f"[{r.status}]"
        print(f"  {status:<12} {r.filename:<40} chunks={r.chunks_embedded}  ms={r.duration_ms}")


def cmd_query(args):
    from app.bootstrap import build_container
    container = build_container()
    response = container.rag_service.answer(args.query, top_k=args.top_k)
    print(f"\n{response.answer}\n")
    if response.citations:
        print("Citations:")
        for c in response.citations:
            page = f" p.{c.page_number}" if c.page_number else ""
            print(f"  {c.filename}{page}  [{c.chunk_id}]")
    print(f"\n  query_id={response.metadata.get('query_id', 'N/A')}")
    print(f"  retrieval_ms={response.metadata.get('retrieval_ms', 'N/A')}")
    print(f"  generation_ms={response.metadata.get('generation_ms', 'N/A')}")


def cmd_evaluate(args):
    from app.bootstrap import build_container
    from app.evaluation.runner import EvaluationRunner

    container = build_container()
    report = EvaluationRunner(container).run(args.dataset, generate=args.generate, top_k=args.top_k)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Hybrid Search RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest files from a directory")
    ingest_parser.add_argument("path", type=str, nargs="?", default=None, help="Directory (or file) to ingest; defaults to data/sample")
    ingest_parser.set_defaults(func=cmd_ingest)

    query_parser = subparsers.add_parser("query", help="Run a RAG query")
    query_parser.add_argument("query", type=str, help="User query text")
    query_parser.add_argument("--top-k", type=int, default=10)
    query_parser.set_defaults(func=cmd_query)

    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation against a dataset")
    eval_parser.add_argument("dataset", type=str, default="data/evaluation/retrieval_samples.json", nargs="?", help="Path to evaluation dataset (JSON or JSONL)")
    eval_parser.add_argument("--generate", action="store_true", help="Run the full RAG pipeline to compute generation metrics")
    eval_parser.add_argument("--top-k", type=int, default=10)
    eval_parser.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()