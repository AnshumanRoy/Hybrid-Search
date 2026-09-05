# Ingestion pipeline

Ingestion converts files into queryable units across three indexes: a SQLite (SQLAlchemy)
relational store for metadata, a Chroma collection for dense vectors, and an in-process BM25
index for lexical scoring.

Entry points: `python -m app ingest [path]` (CLI), `POST /documents/ingest` (single uploaded
file), `POST /documents/ingest-dir` (server-side directory).

## Stage 1 — Source acquisition

`LocalFileSource` wraps a root directory and yields `FileReference` records. It:

- recurses the allowed extensions (`ALLOWED_EXTENSIONS`, default `pdf, txt, md`);
- resolves every candidate path inside the root and rejects path-traversal attempts;
- enforces per-file size (default 10 MB) and a per-batch document cap;
- auto-creates its root directory when it does not exist (defaults to `DATA_DIR`/`sample`
  from the CLI, `DATA_DIR` from the API).

The API's file upload path bypasses the file source entirely: the byte payload is handed
directly to `IngestionService.process_file(reference, data=bytes)`.

## Stage 2 — Parsing

`DocumentParserRouter` inspects the file extension and delegates to a registered parser:

| Parser | Source | Details |
| --- | --- | --- |
| `PDFParser` | PyMuPDF (`import pymupdf`, legacy `fitz` fallback) | Extracts text per page, tracks `page_number`, respects `MAX_PDF_PAGES` |
| `TextParser` | built-in | Reads the file as UTF‑8 text, one logical "page" |

Parsers return a `ParsedDocument` (pages of text destined for chunking). The router raises
`UnsupportedFormatError` for unknown extensions before any IO happens. Because page counts must
be captured while the PDF document handle is open, `PDFParser` snapshots metadata (page count,
title) on the parsed object before closing the underlying document.

## Stage 3 — Sentence-aware chunking

`SentenceAwareChunker(window_chars, overlap_chars)` produces chunks that never split a sentence:

1. Text is split into sentences with a boundary-aware splitter (handles common abbreviations and
   edge contractions).
2. Sentences are packed into a sliding window limited by `CHUNK_SIZE` characters.
3. A chunk is emitted when adding the next sentence would exceed the window.
4. Each new chunk begins with the tail of the previous chunk (up to `CHUNK_OVERLAP` chars),
   preserving context across the boundary.

Cross-field validation in settings guarantees `chunk_overlap < chunk_size`.

Chunk identity: each chunk is identified by `document_id::page::index` (stable across re-ingests
of the same document). `EmbeddedChunk` pairs every chunk with its embedding vector, the embedding
model name, and an `embedding_version`.

## Stage 4 — Embedding

`SentenceTransformerEmbedder` loads `EMBEDDING_MODEL` (default `all-MiniLM-L6-v2`, 384 dims) on a
resolved device (`auto` → cuda/mps/cpu) and embeds batches with L2-normalized vectors.
All chunks share one model per process; the vector collection name is derived from the model
slug (`hybrid_search_<model_slug>`) so data never mixes across model versions silently.

## Stage 5 — Persistence (three-way index)

`IndexPersister` coordinates the writes atomically per document:

1. **Relational store** — `DocumentRecord` (id, filename, status, pages, content hash, timestamps)
   and `ChunkRecord` (id, document id, page, index, text, token ratio, embedding metadata).
   A `QueryLogRow` table exists for future query logging.
2. **Vector store** — chunk vectors upserted into Chroma, keyed by chunk id.
3. **Lexical index** — BM25 token frequencies updated for each chunk.

## Incremental & idempotent ingestion

Every document gets a `content_hash` (SHA-256 over the raw bytes). Before processing,
`IndexPersister.find_by_hash(hash)` is consulted:

- **Hash found** → the file is skipped (unchanged content). No embeddings, no index writes.
- **Hash missing / changed** → `replace_document` runs: old rows for that document id are
  removed from SQLite, Chroma, and BM25, then the new document is inserted. There is no
  partial state — a re-ingest of a modified file yields a fresh, consistent index.

The CLI prints per-file status (`[indexed]`, `[skipped]`, `[failed]`) and the API returns the
same shape as `IngestResponse`.

## Failure behavior

- Unsupported extension or malformed PDF → per-document `failed` status with the error message;
  other documents in the batch still proceed.
- Unexpected exceptions are caught per file so one bad document cannot abort a directory ingest.
- A missing/offline embedding model raises `EmbeddingError`; a reranker that cannot load degrades
  the pipeline to no-rerank with a warning.