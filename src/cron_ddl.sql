CREATE TABLE IF NOT EXISTS cron_jobs (
    id                TEXT PRIMARY KEY,
    description       TEXT NOT NULL,
    schedule          TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    last_run          TEXT,
    one_shot          INTEGER NOT NULL DEFAULT 0,
    created_by_model  INTEGER NOT NULL DEFAULT 0,
    parent_session_id TEXT,
    status            TEXT NOT NULL DEFAULT 'pending'
);
