"""
FastAPI server — the "restaurant window" that exposes your RAG engine to the world.
Run with:  uvicorn backend.api:app --reload --port 8000
"""

import os
import json
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag_engine import (
    load_files_from_folder,
    chunk_files,
    build_vector_store,
    load_vector_store,
    HybridRetriever,
    answer_question,
)

# ── Global state ─────────────────────────────────────────────────────────────
# These are loaded once when the server starts, then reused for every request.
retriever: Optional[HybridRetriever] = None
indexed_folder: Optional[str] = None
chunks_cache = []

CHROMA_DIR = "./chroma_db"
META_FILE = "./chroma_db/meta.json"   # tracks which folder was indexed


# ── Startup ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load existing vector store on startup if one exists."""
    global retriever, indexed_folder, chunks_cache

    if Path(CHROMA_DIR).exists() and Path(META_FILE).exists():
        try:
            with open(META_FILE) as f:
                meta = json.load(f)
            indexed_folder = meta.get("folder")

            print(f"[✓] Loading existing index for: {indexed_folder}")
            vector_store = load_vector_store(CHROMA_DIR)

            # Reload chunks for BM25 (re-read the same files)
            files = load_files_from_folder(indexed_folder)
            chunks_cache = chunk_files(files)
            retriever = HybridRetriever(vector_store, chunks_cache)
            print("[✓] RAG system ready!")
        except Exception as e:
            print(f"[!] Could not load existing index: {e}")

    yield  # server runs here


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Gitty API",
    description="Gitty — Ask questions about any GitHub repo using AI",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ───────────────────────────────────────────────────
class IndexRequest(BaseModel):
    folder_path: str          # e.g. "C:/Users/you/projects/my-app" or "~/projects/my-app"

class QuestionRequest(BaseModel):
    query: str
    use_reranker: bool = True  # set False to go faster (slightly worse quality)

class Source(BaseModel):
    file: str
    snippet: str

class QuestionResponse(BaseModel):
    answer: str
    sources: list[Source]
    query: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Quick check to see if the server is running."""
    return {
        "status": "ok",
        "indexed": indexed_folder is not None,
        "indexed_folder": indexed_folder,
        "chunks": len(chunks_cache),
    }


@app.post("/index")
async def index_codebase(request: IndexRequest):
    """
    INDEX endpoint — reads a folder, chunks files, embeds them, stores in ChromaDB.
    This is the slow step (1-5 minutes for a large repo). Only needs to run once.
    """
    global retriever, indexed_folder, chunks_cache

    folder = os.path.expanduser(request.folder_path)  # expand ~ to home dir

    if not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")

    try:
        # Step 1: Read files
        files = load_files_from_folder(folder)
        if not files:
            raise HTTPException(status_code=400, detail="No supported files found in that folder.")

        # Step 2: Chunk
        chunks = chunk_files(files)
        chunks_cache = chunks

        # Step 3: Clear old database directory to avoid mixing past project data
        import shutil
        if Path(CHROMA_DIR).exists():
            try:
                shutil.rmtree(CHROMA_DIR)
            except Exception as e:
                print(f"[!] Warning: Could not delete old database folder: {e}")

        # Step 3.5: Embed + store new project
        vector_store = build_vector_store(chunks, persist_dir=CHROMA_DIR)

        # Step 4: Build retriever
        retriever = HybridRetriever(vector_store, chunks)
        indexed_folder = folder

        # Save metadata so the index survives a server restart
        Path(CHROMA_DIR).mkdir(exist_ok=True)
        with open(META_FILE, "w") as f:
            json.dump({"folder": folder, "files": len(files), "chunks": len(chunks)}, f)

        return {
            "status": "indexed",
            "folder": folder,
            "files_indexed": len(files),
            "chunks_created": len(chunks),
            "message": "Your codebase is now ready to answer questions!"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest):
    """
    ASK endpoint — the main one. Send a question, get an answer + sources back.
    """
    if retriever is None:
        raise HTTPException(
            status_code=400,
            detail="No codebase indexed yet. Call POST /index first."
        )

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        answer, sources = answer_question(
            query=request.query,
            retriever=retriever,
            use_reranker=request.use_reranker,
        )

        return QuestionResponse(
            answer=answer,
            query=request.query,
            sources=[
                Source(
                    file=doc.metadata.get("source", "unknown"),
                    snippet=doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
                )
                for doc in sources
            ]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status():
    """Returns info about the current index."""
    return {
        "indexed_folder": indexed_folder,
        "total_chunks": len(chunks_cache),
        "ready": retriever is not None,
    }
