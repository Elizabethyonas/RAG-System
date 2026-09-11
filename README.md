### RAG Car Driver Assistant 

This repo implements a minimal Retrieval-Augmented Generation (RAG) pipeline:

1. Embed a small set of documents
2. Store them in a FAISS vector index
3. Retrieve relevant chunks for a user question
4. Send retrieved context to a small LLM to generate an answer

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Postgres + pgvector (Docker)

Use this if local Homebrew Postgres and `pgvector` versions do not match.

```bash
docker compose up -d
export DATABASE_URL="postgresql+psycopg://postgres:root@localhost:5433/fyp"
```

- **Port 5433** on the host maps to **5432** in the container (see `docker-compose.yml`). Change the left side if 5433 is already in use.
- On first start with an empty volume, `docker/initdb/01-vector.sql` runs `CREATE EXTENSION vector`.
- **driver-garage-backend:** set the same `DATABASE_URL` (same host, port, db `fyp`, user/password) in `.env` so Prisma and the RAG app share one database.
- Reset container data (destructive): `docker compose down -v && docker compose up -d`

#### Single database for Prisma + RAG (Option B)

Use one Postgres for Express (Prisma) and this repo. Do **not** set `PRISMA_DATABASE_URL` here.

1. Start Docker: `docker compose up -d`.
2. In **driver-garage-backend** `.env`, point Prisma at the same instance. Prisma expects a `postgresql://` URL (no `+psycopg`), for example:
   `DATABASE_URL="postgresql://postgres:root@localhost:5433/fyp"`
3. From the **driver-garage-backend** root, apply the schema so tables such as `"EducationContent"` exist:
   `npx prisma migrate deploy`  
   (or `npx prisma db push` if you intentionally use push in dev.)
4. In **RAG-System**, keep the Python URL form with the **psycopg** driver, e.g.  
   `export DATABASE_URL="postgresql+psycopg://postgres:root@localhost:5433/fyp"`  
   Leave `PRISMA_DATABASE_URL` unset.
5. Build the global KB and sync manuals as usual (`python scripts/build_index.py`, then `python scripts/sync_education_manuals.py` with `EXPRESS_PROJECT_ROOT` if you read PDFs from disk).

## Step 1: Build the FAISS index

```bash
python scripts/build_index.py
```

This generates:
- `models/faiss_index`
- `models/docs.txt`

## Step 2: Run the chatbot

```bash
python scripts/chat.py
```

Try asking:
- `What does engine oil warning mean?`
- `What does ABS warning light mean?`

## Chat History API (for Flutter)

Run the API server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

By default, chat history is stored in SQLite at `models/chat_history.db`.
To use Postgres instead, set `DATABASE_URL`:

```bash
export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/rag_chat"
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Authentication

User identity comes from a JWT (recommended for Flutter production), not from the request body.

- Set `JWT_SECRET` (HS256 by default). Send `Authorization: Bearer <token>` where the payload includes `"sub": "<your-user-id>"`.
- Optional: `JWT_ALGORITHM` (default `HS256`), `JWT_AUDIENCE`, `JWT_ISSUER` if you issue tokens with those claims.
- **Local dev** (no `JWT_SECRET`): send header `X-User-Id: user1` instead. Do not deploy without `JWT_SECRET` if the API is exposed.

### Endpoints

- `POST /sessions`
  - body: `{ "title": "My chat", "car_context": "2024 Honda Civic", "vehicle_id": "<Prisma Vehicle.id>" }`
  - **`vehicle_id`** (recommended): persists on the session so every message can load **Vehicle** + **VehicleMaintenanceHealth** from Postgres without the client resending them. Run `docs/migrations/001_chat_sessions_vehicle_id.sql` once if your DB predates this column.
  - `user_id` is taken from JWT `sub` (or `X-User-Id` in dev).
- `GET /sessions` — list sessions for the authenticated user
- `GET /sessions/{session_id}` — session metadata (404 if not yours)
- `DELETE /sessions/{session_id}` — delete session and all messages (204 No Content)
- `GET /sessions/{session_id}/messages?limit=50` — returns up to the latest `limit` messages in chronological order (JSON array).
- `POST /sessions/{session_id}/messages`
  - body: `{ "message": "What does ABS light mean?", "car_context": "", "vehicle_id": "<optional override>", "vehicle_context": {...}, "use_user_manual": true }`
  - If the session already has `vehicle_id`, you may omit it on each message.

The `POST /sessions/{session_id}/messages` endpoint:
1. stores the user message,
2. runs the RAG assistant,
3. stores the assistant answer,
4. returns both messages.

## Knowledge base: PostgreSQL + pgvector (Express admin DB)

Retrieval no longer requires local FAISS files when `DATABASE_URL` points at **PostgreSQL** with the **`vector`** extension. Chunks live in **`rag_kb_chunks`** (same DB your Express app can use for manual uploads).

- **Env**
  - `DATABASE_URL=postgresql+psycopg://user:pass@host:5432/dbname`
  - `RAG_KB_BACKEND=auto` (default): use pgvector when URL is Postgres; use local FAISS when URL is SQLite/file.
  - `RAG_KB_BACKEND=faiss`: always use `models/faiss_index` + `models/docs.txt` (offline / no Postgres).
  - `RAG_KB_BACKEND=pgvector`: require Postgres + tables (fails fast if URL is not Postgres).

- **Schema** (reference SQL): `docs/schema_rag_pgvector.sql`  
  Docker loads the same tables + **HNSW index** from `docker/initdb/02-rag_kb.sql` on first database init.
- **Existing Postgres** (already full of chunks): create the vector index once —  
  `export DATABASE_URL=... && python scripts/apply_vector_index.py`  
  (or run `docs/migrations/002_rag_kb_chunks_embedding_hnsw.sql` with `CONCURRENTLY` during production maintenance).

- **Sync global KB from this repo** (embeds + upserts global rows):

```bash
export DATABASE_URL="postgresql+psycopg://..."
python scripts/build_index.py
```

- **Sync a user PDF** (or use `--manual-id <uuid>` to link an Express `manuals.id`):

```bash
python scripts/upload_manual.py --pdf path/to/manual.pdf --user-id <sub-from-jwt>
```

Express can store uploaded files and metadata; either **ETL text into `rag_kb_chunks`** and run a small embed worker, or **call/trigger** the Python scripts above after upload.

## Collect web car manuals (PDF)

Use this helper to discover and download owner manual PDFs from public websites into `data/raw/car_manual_pdfs/`:

```bash
python scripts/collect_car_manual_pdfs.py --source all --limit 300 --max-per-make 30
```

Useful options:
- `--source nissan|carmans|search|all` chooses discovery sources (Carmans is multi-brand).
- `--makes toyota,honda,ford` and `--years 2020,2021,2022` to scope the crawl.
- `--max-per-make 30` keeps the dataset balanced across brands.
- `--metadata-file data/raw/car_manual_pdfs/metadata.csv` stores source URLs and SHA-256 hashes.
- Re-run the command to continue collecting; duplicates are skipped by hash.

### driver-garage-backend: index admin `MANUALS` (Education CRUD)

Express stores PDFs under `uploads/education-manuals/` and saves public URLs in `EducationContent.pdf_url` (Prisma). The RAG repo can index those into `rag_kb_chunks` (same Postgres, `manual_id` = education row UUID).

```bash
export DATABASE_URL="postgresql+psycopg://..."   # same DB as Express .env
export EXPRESS_PROJECT_ROOT="/path/to/driver-garage-backend"
python scripts/sync_education_manuals.py
# one item:
python scripts/sync_education_manuals.py --content-id <EducationContent.id>
```

- **`EXPRESS_PROJECT_ROOT`**: absolute path to the Express repo so PDFs are read from disk (`uploads/education-manuals/<file>`). If files are not on this machine, the script tries **HTTP GET** on `pdf_url` (Express must be running and serving `/uploads`).
- Re-run sync after admins add or replace a manual PDF.

Optional Express integration checklist: `docs/EXPRESS_RAG_PROMPT.md`.

