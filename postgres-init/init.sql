CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS conversation_history (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversation_history_session_time
ON conversation_history(session_id, created_at);

CREATE TABLE IF NOT EXISTS knowledge_documents (
    document_id UUID PRIMARY KEY,
    original_file_name TEXT NOT NULL,
    stored_file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    mime_type TEXT,
    content_hash TEXT NOT NULL,
    source_title TEXT,
    source_url TEXT,
    source_type TEXT NOT NULL DEFAULT 'admin_upload',
    access_level TEXT NOT NULL DEFAULT 'public',
    version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_documents_hash
ON knowledge_documents(content_hash);

CREATE INDEX IF NOT EXISTS idx_knowledge_documents_status
ON knowledge_documents(status);

-- LlamaIndex creates and manages:
--     data_hkpl_knowledge
-- after the first ingestion.
