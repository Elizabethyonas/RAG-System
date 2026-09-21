-- Accounts for FastAPI register/login. id is JWT sub / chat_sessions.user_id.
CREATE TABLE IF NOT EXISTS rag_users (
  id VARCHAR(36) PRIMARY KEY,
  email VARCHAR(255) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  name VARCHAR(255) NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_rag_users_email ON rag_users (email);
