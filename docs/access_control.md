# Multi-user access control

Documents are owned by user accounts and can be shared with other users.
The API requires a **bearer token** for everything except `/health` and the
`/auth/register` + `/auth/login` endpoints.

## Accounts

- `POST /auth/register` — create an account. Returns an access token immediately.
- `POST /auth/login` — exchange email+password for a token.
- `GET /auth/me` — current user.

Passwords are hashed with **PBKDF2-HMAC-SHA256** (310k iterations, per-user salt,
stdlib only). Tokens are **standard JWTs** (`header.payload.signature`, compact
serialization, `alg=HS256` pinned on verification, `typ=JWT`, claims `iat`/`exp`/`sub`/`email`),
signed with the shared `AUTH_SECRET_KEY`. The algorithm is pinned at `HS256` during verification
so `alg=none`/algorithm-confusion tokens are rejected. Tokens expire via the `exp` claim
(`AUTH_TOKEN_TTL_SECONDS`, default 24h). There are no external auth dependencies.

Send the token on every request:

```
Authorization: Bearer <token>
```

## Access model

For a document, a user's effective permission is resolved in this order:

| Condition | Permission |
| --- | --- |
| `documents.owner_id == user` | **owner** (edit, delete, share) |
| shared as `editor` | **editor** (re-index / update) |
| shared as `viewer`, or `owner_id IS NULL` (public) | **viewer** (read only) |
| otherwise | **none** (404 — existence is hidden) |

`owner_id = NULL` describes **legacy/public** documents (e.g. rows ingested
before the migration, or out-of-band content). They are readable by every
authenticated user but owned by nobody, so they can never be edited, deleted,
or shared.

## Ownership, sharing and query scoping

- Uploads via `POST /documents/ingest` are owned by the caller.
- `document_id` is **deterministic** (`doc_<sha256(source::path)[:16]>`): re-indexing
  a document via `PUT /documents/{id}` keeps its id and its shares, but replaces
  the underlying chunks/content.
- `GET /documents` returns only documents you can see (own + public + shared),
  each annotated with `permission` (`owner`/`editor`/`viewer`), `shared`, and
  `owner_email`.
- `POST /query` is scoped **inside retrieval**: the caller's accessible document ids
  are passed down as `SearchFilters.document_ids`, so both the vector store (Chroma
  `$in` / pgvector `IN`) and the BM25 index are filtered **before** fusion and
  reranking. Restricted documents are never scored by the reranker. A post-filter
  over the returned chunks remains as a defense-in-depth safety net.

## Sharing endpoints (owner only)

| Method | Path | Body |
| --- | --- | --- |
| `POST`  | `/documents/{id}/share`            | `{"user_email", "permission": "viewer"\|"editor"}` |
| `GET`   | `/documents/{id}/shares`           | — |
| `DELETE`| `/documents/{id}/share/{email}`    | — |

## Behavior notes

- 403 vs 404: operations on documents you *cannot see* return `404` (no
  existence leak); operations you *can see* but lack permission for return `403`.
- Deleting a document removes its chunks from every index (vector, BM25, SQL)
  and its shares.
- Chunk-level access is intentionally **not** modelled; access is document-scoped.

## Configuration

| Env var | Default | Meaning |
| --- | --- | --- |
| `AUTH_SECRET_KEY` | `dev-only-change-me` | Token signing secret — set a long random value in deployment |
| `AUTH_TOKEN_TTL_SECONDS` | `86400` | Token lifetime |

Both maps to `Settings.auth_secret_key` / `auth_token_ttl_seconds`.