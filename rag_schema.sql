-- RAG corpus storage for PhishSentry threat-intel retrieval
-- Embedding model: BAAI/bge-m3 (dense, 1024-dim, cosine similarity)

CREATE TABLE IF NOT EXISTS rag_chunks (
    id          BIGSERIAL    PRIMARY KEY,
    doc_id      TEXT         NOT NULL,        -- source document identifier
    title       TEXT,                          -- human-readable source title
    source      TEXT,                          -- provenance (curated / URL / feed)
    chunk_index INT          NOT NULL,         -- position within the document
    content     TEXT         NOT NULL,         -- the chunk text
    embedding   vector(1024) NOT NULL,         -- BGE-M3 dense embedding
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (doc_id, chunk_index)               -- lets ingestion be idempotent (upsert)
);

-- Approximate-nearest-neighbour index for fast cosine similarity search.
CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);

-- Fast lookup / cleanup by source document.
CREATE INDEX IF NOT EXISTS rag_chunks_doc_id_idx
    ON rag_chunks (doc_id);
