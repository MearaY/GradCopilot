-- GradCopilot 数据库初始化

-- ============================================================
-- 0. 前置扩展
-- ============================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- 1. 表：sessions
-- ============================================================
CREATE TABLE IF NOT EXISTS sessions (
    id          VARCHAR(64)  PRIMARY KEY,
    name        VARCHAR(100) NOT NULL DEFAULT '新会话',
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    message_count INTEGER    NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions (created_at DESC);

-- ============================================================
-- 2. 表：papers（论文元数据，含 session 隔离）
-- ============================================================
CREATE TABLE IF NOT EXISTS papers (
    id              SERIAL      PRIMARY KEY,
    paper_id        VARCHAR(64) NOT NULL,
    session_id      VARCHAR(64) NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    title           TEXT        NOT NULL,
    authors         TEXT[]      NOT NULL DEFAULT '{}',
    abstract        TEXT,
    published_date  DATE,
    pdf_url         TEXT        NOT NULL,
    arxiv_url       TEXT        NOT NULL,
    local_path      TEXT,
    downloaded_at   TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (paper_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_papers_session_id ON papers (session_id);

-- ============================================================
-- 3. 表：paper_chunks（含向量）
-- ============================================================
CREATE TABLE IF NOT EXISTS paper_chunks (
    id          SERIAL       PRIMARY KEY,
    paper_id    VARCHAR(64)  NOT NULL,
    session_id  VARCHAR(64)  NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    chunk_index INTEGER      NOT NULL,
    content     TEXT         NOT NULL,
    page_number INTEGER,
    embedding   VECTOR(1536) NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_session_id ON paper_chunks (session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_paper_id   ON paper_chunks (paper_id, session_id);

-- ivfflat 向量索引（需在数据量足够后手动执行 VACUUM ANALYZE paper_chunks）
-- 注意：ivfflat 索引需要数据才能建立，初始化阶段跳过，待首次 build_knowledge 后执行
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON paper_chunks
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
