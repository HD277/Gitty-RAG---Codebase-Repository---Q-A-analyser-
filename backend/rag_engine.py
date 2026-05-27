"""
RAG Engine — the brain of the project.
This file handles: reading files, chunking, embedding, storing, and retrieving.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

# ── Text splitting ──────────────────────────────────────────────────────────
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Embeddings & Vector store ───────────────────────────────────────────────
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Keyword search (BM25) ───────────────────────────────────────────────────
from rank_bm25 import BM25Okapi

# ── Reranker ────────────────────────────────────────────────────────────────
from sentence_transformers import CrossEncoder

# ── LLM ─────────────────────────────────────────────────────────────────────
import google.generativeai as genai


# ---------------------------------------------------------------------------
# STEP 1 — Read all files from a local folder
# ---------------------------------------------------------------------------
def load_files_from_folder(folder_path: str) -> List[Dict]:
    """
    Walk through every folder and file.
    Collect text content from code and doc files.
    Skip junk folders like .git and node_modules.
    """
    supported_extensions = [
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".md", ".txt", ".json", ".yaml", ".yml",
        ".html", ".css", ".java", ".go", ".rs", ".cpp", ".c", ".h"
    ]

    skip_dirs = {
        ".git", "node_modules", "__pycache__", ".venv",
        "venv", "dist", "build", ".next", ".idea", ".vscode"
    }

    files = []

    for root, dirs, filenames in os.walk(folder_path):
        # Remove junk directories so os.walk skips them entirely
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for filename in filenames:
            # Only process files with supported extensions
            if not any(filename.endswith(ext) for ext in supported_extensions):
                continue

            full_path = os.path.join(root, filename)

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Skip empty files or files that are too large (>50KB)
                if not content.strip() or len(content) > 50_000:
                    continue

                files.append({
                    "path": full_path,
                    "filename": filename,
                    "extension": Path(filename).suffix,
                    "content": content,
                    "size": len(content)
                })

            except Exception as e:
                print(f"Could not read {full_path}: {e}")

    print(f"[✓] Loaded {len(files)} files from {folder_path}")
    return files


# ---------------------------------------------------------------------------
# STEP 2 — Chunk files into small pieces
# ---------------------------------------------------------------------------
def chunk_files(files: List[Dict]) -> List:
    """
    Split each file's content into overlapping chunks.
    We keep chunk_overlap so that context is never lost at the boundary.
    Metadata (file path, name) is attached to every chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,        # each chunk = ~800 characters (roughly 200 words)
        chunk_overlap=80,      # 80-character overlap between consecutive chunks
        length_function=len,
        separators=["\n\n", "\n", "def ", "class ", " ", ""]  # prefers splitting at function/class boundaries
    )

    all_chunks = []

    for file in files:
        try:
            chunks = splitter.create_documents(
                texts=[file["content"]],
                metadatas=[{
                    "source": file["path"],
                    "filename": file["filename"],
                    "extension": file["extension"],
                }]
            )
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Could not chunk {file['filename']}: {e}")

    print(f"[✓] Created {len(all_chunks)} chunks from {len(files)} files")
    return all_chunks


# ---------------------------------------------------------------------------
# STEP 3 — Embed chunks and save to ChromaDB
# ---------------------------------------------------------------------------
def build_vector_store(chunks: List, persist_dir: str = "./chroma_db") -> Chroma:
    """
    Convert each chunk to a vector (list of numbers that represents meaning).
    Store all vectors in ChromaDB — a local vector database saved to disk.
    Uses a free, local HuggingFace model (no API key needed).
    """
    print("[…] Loading embedding model (first time downloads ~90MB) ...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # fast, free, local model
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("[…] Building vector store (this takes a moment for large repos) ...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="codebase"
    )

    print(f"[✓] Saved {len(chunks)} vectors to {persist_dir}")
    return vector_store


def load_vector_store(persist_dir: str = "./chroma_db") -> Chroma:
    """Load an already-built vector store from disk (fast, no re-embedding)."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="codebase"
    )


# ---------------------------------------------------------------------------
# STEP 4 — Hybrid Retriever (semantic + keyword)
# ---------------------------------------------------------------------------
class HybridRetriever:
    """
    Combines two retrieval methods:
      1. Semantic search  — finds chunks with similar *meaning* using vectors
      2. BM25 keyword     — finds chunks with exact *keyword matches*
    Merging both gives much better results than either alone.
    """

    def __init__(self, vector_store: Chroma, chunks: List):
        self.vector_store = vector_store
        self.chunks = chunks

        # Build BM25 index: tokenize every chunk by splitting on whitespace
        tokenized_corpus = [c.page_content.split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int = 10) -> List:
        """Return the top-k most relevant chunks for a query."""

        # --- Semantic search ---
        try:
            semantic_hits = self.vector_store.similarity_search(query, k=k)
        except Exception:
            semantic_hits = []

        # --- BM25 keyword search ---
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:k]
        keyword_hits = [self.chunks[i] for i in top_bm25_indices]

        # --- Merge and deduplicate ---
        seen_content = set()
        combined = []
        for doc in semantic_hits + keyword_hits:
            key = doc.page_content[:100]   # use first 100 chars as unique key
            if key not in seen_content:
                seen_content.add(key)
                combined.append(doc)

        return combined[:k]


# ---------------------------------------------------------------------------
# STEP 5 — Reranker
# ---------------------------------------------------------------------------
def rerank(query: str, candidates: List, top_n: int = 4) -> List:
    """
    A cross-encoder reranker reads both the query AND each chunk together
    and gives a precise relevance score. Much more accurate than vector search alone.
    Downloads ~80MB on first run.
    """
    print("[…] Loading reranker model (first time downloads ~80MB) ...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)

    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]


# ---------------------------------------------------------------------------
# STEP 6 — Ask Gemini for an answer
# ---------------------------------------------------------------------------
def ask_gemini(query: str, top_docs: List) -> str:
    """
    Build a prompt from the retrieved code chunks and ask Gemini to answer.
    Uses the free Gemini API (gemini-1.5-flash model).
    Returns Gemini's response as a string.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "❌ GEMINI_API_KEY not set. Please add it to your .env file."

    # Configure Gemini with your API key
    genai.configure(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(model_name)

    # Build the context block — paste each retrieved chunk with its file path
    context_parts = []
    for i, doc in enumerate(top_docs, 1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Chunk {i}] File: {source}\n{doc.page_content}")

    context = ("\n" + "─" * 60 + "\n").join(context_parts)

    prompt = f"""You are an expert code assistant. You answer questions about a codebase
using the code snippets provided below.

Rules:
- Always reference the file name when explaining something.
- If the answer is not in the provided chunks, say so honestly.
- Be concise but complete.
- Use markdown formatting for code snippets.

Here are the most relevant code snippets from the codebase:

{context}

Question: {query}"""

    response = model.generate_content(prompt)
    return response.text


# ---------------------------------------------------------------------------
# STEP 7 — Full pipeline: question → answer
# ---------------------------------------------------------------------------
def answer_question(
    query: str,
    retriever: HybridRetriever,
    use_reranker: bool = True
) -> Tuple[str, List]:
    """
    End-to-end pipeline:
      query → hybrid retrieve → (optional rerank) → ask Gemini → return answer + sources
    """
    # Retrieve candidates
    candidates = retriever.retrieve(query, k=10)

    if not candidates:
        return "No relevant code found for your question.", []

    # Rerank if enabled
    if use_reranker and len(candidates) > 1:
        top_docs = rerank(query, candidates, top_n=4)
    else:
        top_docs = candidates[:4]

    # Get answer from Gemini
    answer = ask_gemini(query, top_docs)

    return answer, top_docs
