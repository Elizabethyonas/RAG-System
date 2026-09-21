# Deploy to Render (FastAPI + pgvector)

This project runs well on Render as one web service plus a managed PostgreSQL database that has the `vector` extension enabled.

## 1) Create PostgreSQL and enable pgvector

Use either:
- Render Postgres
- Neon
- Supabase

Then run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## 2) Deploy using `render.yaml`

1. Push this repo to GitHub.
2. In Render, click **New +** -> **Blueprint**.
3. Select your repo. Render will detect `render.yaml` and create `rag-system-api`.
4. After first deploy, add secret env vars in the service settings.

## 3) Required environment variables

Set these in Render service settings:

- `DATABASE_URL`: `postgresql+psycopg://<user>:<pass>@<direct-host>/<db>?sslmode=require`
  - SQLAlchemy needs `postgresql+psycopg://`, not the Prisma/Neon `postgresql://` form.
  - On Neon, use the **direct** host (no `-pooler`). The pooler is for short serverless queries, not pgvector + bulk inserts.
- `JWT_SECRET`: strong random secret (required for production auth)
- `LLM_PROVIDER=hf_space` (do not run Qwen 3B on a Render Starter instance)

Defaults already in `render.yaml`:

- `JWT_ALGORITHM=HS256`
- `RAG_KB_BACKEND=pgvector`
- `PYTHON_VERSION=3.13.0`

Optional auth hardening:

- `JWT_AUDIENCE`
- `JWT_ISSUER`

## 4) Run one-time indexing job

After deploy, run in a Render Shell (or local machine with the same env vars):

```bash
python scripts/build_index.py --batch-size 200
# If a previous load died mid-way (Neon SSL drop), continue without wiping rows:
python scripts/build_index.py --resume --batch-size 200
```

This populates `rag_kb_chunks` in Postgres for retrieval. A full 179k-chunk load to Neon can take a while; embeddings are reused from `models/global_embeddings.npy` when present.

## 5) Sync manuals (optional)

For a single PDF:

```bash
python scripts/upload_manual.py --pdf path/to/manual.pdf --user-id <jwt-sub>
```

For Express admin manuals:

```bash
export EXPRESS_PROJECT_ROOT=/absolute/path/to/driver-garage-backend
python scripts/sync_education_manuals.py
```

## 6) Validate deployment

- Health check: `GET /health` -> `{"status":"ok"}`
- Open docs: `/docs`
- Confirm JWT auth is active by calling a protected endpoint without a token (should return `401`).

## Notes

- `server.py` currently allows all CORS origins (`*`). Restrict this in production to your app domains.
- Do not rely on `X-User-Id` mode in production; keep `JWT_SECRET` set.
