"""
PhishSentry RAG retrieval microservice.

Loads a sentence-transformers embedding model (bge-large-en-v1.5) and a CrossEncoder
reranker (bge-reranker-base), runs a cosine-similarity search over the rag_chunks
table (pgvector), then reranks the candidates and returns the top results. Models
load once at startup; /health returns 503 until they are ready, so the orchestrator
can gate on readiness.
"""
import os
import math
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")

DATABASE_URL = os.environ["DATABASE_URL"]
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")

_state = {"embedder": None, "reranker": None}


def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def embed_query(text: str) -> np.ndarray:
    vec = _state["embedder"].encode([text], normalize_embeddings=True)[0]
    return np.asarray(vec, dtype=np.float32)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from sentence_transformers import SentenceTransformer, CrossEncoder
    logger.info("Loading embedding model: %s", EMBED_MODEL)
    _state["embedder"] = SentenceTransformer(EMBED_MODEL, device="cpu")
    logger.info("Loading reranker model: %s", RERANK_MODEL)
    _state["reranker"] = CrossEncoder(RERANK_MODEL, device="cpu")
    logger.info("RAG models loaded; service ready.")
    yield
    _state["embedder"] = None
    _state["reranker"] = None


app = FastAPI(title="PhishSentry RAG Service", lifespan=lifespan)


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    candidates: int = Field(default=20, ge=1, le=100)


class Chunk(BaseModel):
    doc_id: str
    title: Optional[str]
    source: Optional[str]
    chunk_index: int
    content: str
    vector_score: float
    rerank_score: float


@app.get("/health")
def health():
    ready = _state["embedder"] is not None and _state["reranker"] is not None
    if not ready:
        raise HTTPException(status_code=503, detail="models not loaded")
    return {"status": "ok", "embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL}


@app.post("/retrieve", response_model=list)
def retrieve(req: RetrieveRequest):
    if _state["embedder"] is None or _state["reranker"] is None:
        raise HTTPException(status_code=503, detail="models not loaded")

    qvec = embed_query(req.query)

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_id, title, source, chunk_index, content,
                       1 - (embedding <=> %s) AS cosine_sim
                FROM rag_chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (qvec, qvec, req.candidates),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    # Cross-encoder rerank of the vector-search candidates.
    pairs = [[req.query, r[4]] for r in rows]
    raw = _state["reranker"].predict(pairs)
    scores = [1.0 / (1.0 + math.exp(-float(x))) for x in raw]  # sigmoid -> [0,1]

    ranked = sorted(zip(rows, scores), key=lambda x: x[1], reverse=True)[: req.top_k]
    return [
        Chunk(
            doc_id=r[0], title=r[1], source=r[2], chunk_index=r[3], content=r[4],
            vector_score=float(r[5]), rerank_score=float(s),
        ).model_dump()
        for r, s in ranked
    ]
