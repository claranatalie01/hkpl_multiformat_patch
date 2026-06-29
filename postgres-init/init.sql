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

CREATE TABLE IF NOT EXISTS prohibited_keywords (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    keyword TEXT NOT NULL,

    category TEXT NOT NULL DEFAULT 'general',

    language TEXT NOT NULL DEFAULT 'en',

    fallback_response TEXT NOT NULL,

    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    created_by TEXT DEFAULT '',

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prohibited_keywords_active
ON prohibited_keywords(is_active);

CREATE INDEX IF NOT EXISTS idx_prohibited_keywords_category
ON prohibited_keywords(category);

CREATE INDEX IF NOT EXISTS idx_prohibited_keywords_created_at
ON prohibited_keywords(created_at DESC);

CREATE TABLE IF NOT EXISTS prohibited_keyword_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    keyword_id UUID,
    action TEXT NOT NULL,
    staff_id TEXT NOT NULL,
    old_value JSONB,
    new_value JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prohibited_keyword_audit_keyword_id
ON prohibited_keyword_audit_log(keyword_id);

CREATE INDEX IF NOT EXISTS idx_prohibited_keyword_audit_staff_id
ON prohibited_keyword_audit_log(staff_id);

CREATE INDEX IF NOT EXISTS idx_prohibited_keyword_audit_action
ON prohibited_keyword_audit_log(action);

CREATE INDEX IF NOT EXISTS idx_prohibited_keyword_audit_created_at
ON prohibited_keyword_audit_log(created_at DESC);