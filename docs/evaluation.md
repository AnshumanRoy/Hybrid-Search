# Evaluation

The evaluation harness measures how well retrieval finds the right evidence and — when an LLM is
configured — how faithfully the answer is grounded in that evidence.

Entry points:

- `python -m app evaluate [dataset] [--generate] [--top-k N]`
- `python -m app.evaluation --dataset <path> [--generate] [--top-k N]`

Both load the same dataset format; the second is the canonical module entrypoint.

## Dataset schema

JSON (with a `questions` key) or JSONL, one dict per question:

```json
{
  "question": "What is Acme's refund window for physical products?",
  "expected_answer": "Customers may request a full refund within 30 days of purchase ...",
  "relevant_terms": ["refund", "30 days", "original packaging"],
  "relevant_docs": ["acme_handbook.txt"],
  "relevant_chunk_ids": ["doc_xxx::p1::i0"]
}
```

Relevance (`relevant_*`) must identify the ground truth *retrievably*:

- `relevant_chunk_ids` — exact chunk ids (matched directly against retrieved chunk ids).
- `relevant_docs` — source filenames. The runner expands each filename to all chunk ids that
  belong to that document via the document store, so a chunk counts as relevant when it comes
  from a ground-truth document.

At least one of the two must be present for retrieval metrics to be meaningful.

## Retrieval metrics

`evaluate_retrieval(ranked_lists, relevant_sets, top_ks)` in `app/evaluation/metrics.py`:

| Metric | Definition |
| --- | --- |
| `recall@k` | fraction of relevant chunks in the top-k |
| `precision@k` | fraction of top-k that is relevant |
| `ndcg@k` | normalized discounted cumulative gain |
| `mrr` | mean reciprocal rank of the first relevant hit |

Top-k set defaults to `[1, 5, 10]` (`EVAL_TOP_KS`). The runner reports the aggregate over all
questions, e.g.:

```json
{
  "retrieval": { "recall@1": 0.17, "recall@5": 0.5, "recall@10": 1.0,
                 "precision@1": 0.67, "ndcg@10": 0.77, "mrr": 0.83 },
  "questions": 3
}
```

## Generation metrics (`--generate`)

When `--generate` is passed, the full RAG pipeline runs per question and three LLM-as-judge-style
metrics are computed:

| Metric | Meaning |
| --- | --- |
| `faithfulness` | share of claims in the answer that are supported by the retrieved context |
| `answer_relevance` | how well the answer addresses the question |
| `context_relevance` | relevance of the retrieved context to the question |

Faithfulness and answer relevance compare the generated answer against the expected answer /
context; context relevance compares the retrieved chunk ids against the ground-truth set. These
are heuristic, deterministic implementations — plug in a stronger judge LLM by extending the
metric functions without touching the runner.

## As a regression gate

The runner is a pure function of (dataset, container): `EvaluationRunner(container).run(dataset)`.
Unit tests exercise the metric implementations directly; integration tests can pin a minimum
`recall@10` / `mrr` on a fixed sample dataset. Because the metric computations are deterministic
for a fixed index, the same dataset can be run after any retrieval change to compare regressions.

## Working with the bundled sample

```bash
python -m app.evaluation --dataset data/evaluation/retrieval_samples.json
```

The bundled dataset contains 3 questions over `acme_handbook.txt` with `relevant_docs`
ground truth. Expected ballpark on a fresh index (see `docs/retrieval.md` for the default
pipeline):

- `recall@10` ≈ 1.0, `mrr` ≈ 0.8+.

Ingest the sample corpus first (`python -m app ingest`), otherwise the question set simply has
no relevant documents and metrics collapse to 0.