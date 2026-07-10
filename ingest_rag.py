"""
Ingest the curated threat-intel corpus into rag_chunks.

Chunks each document by paragraph, embeds with the sentence-transformers model,
and upserts into Postgres. Idempotent: re-running updates existing rows via
ON CONFLICT (doc_id, chunk_index), so you can edit the corpus and re-ingest
without duplicates.

Run once after the rag service is up:
    docker compose exec rag python ingest_rag.py
"""
import os
import json
import logging

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest")

DATABASE_URL = os.environ["DATABASE_URL"]
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
CORPUS_PATH = os.environ.get("RAG_CORPUS", "rag_corpus.json")


def chunk_paragraphs(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def main():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        docs = json.load(f)
    logger.info("Loaded %d documents from %s", len(docs), CORPUS_PATH)

    rows = []  # (doc_id, title, source, chunk_index, content)
    for d in docs:
        for i, para in enumerate(chunk_paragraphs(d["text"])):
            rows.append((d["doc_id"], d.get("title"), d.get("source"), i, para))
    logger.info("Prepared %d chunks", len(rows))

    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: %s", EMBED_MODEL)
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

    contents = [r[4] for r in rows]
    logger.info("Embedding %d chunks ...", len(contents))
    vecs = embedder.encode(
        contents, batch_size=16, normalize_embeddings=True, show_progress_bar=True
    )
    vecs = np.asarray(vecs, dtype=np.float32)

    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    try:
        with conn.cursor() as cur:
            for (doc_id, title, source, idx, content), vec in zip(rows, vecs):
                cur.execute(
                    """
                    INSERT INTO rag_chunks
                        (doc_id, title, source, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id, chunk_index) DO UPDATE SET
                        title = EXCLUDED.title,
                        source = EXCLUDED.source,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        created_at = now()
                    """,
                    (doc_id, title, source, idx, content, vec),
                )
        conn.commit()
        logger.info("Upserted %d chunks into rag_chunks.", len(rows))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
