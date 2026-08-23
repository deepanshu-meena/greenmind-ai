"""
knowledge_base.py
Fetches Wikipedia articles for all 17 SDGs,
chunks them, embeds with Gemini, stores in ChromaDB.
"""

import requests
import chromadb
import google.generativeai as genai
from config import (
    SDG_TOPICS, CHROMA_PATH, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, EMBEDDING_MODEL
)


# ── Helpers ──────────────────────────────────────────────────

def fetch_wikipedia(page_title: str) -> str:
    """Fetch plain-text extract from Wikipedia."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": page_title,
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        pages = r.json()["query"]["pages"]
        page  = next(iter(pages.values()))
        return page.get("extract", "")[:6000]   # cap at 6 000 chars
    except Exception:
        return ""


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words  = text.split()
    chunks = []
    step   = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if len(chunk.strip()) > 50:          # skip tiny leftovers
            chunks.append(chunk)
    return chunks


def get_embedding(text: str, api_key: str) -> list[float]:
    """Embed a single text string with Gemini."""
    genai.configure(api_key=api_key)
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text[:2000],
        task_type="retrieval_document",
    )
    return result["embedding"]


# ── Public API ───────────────────────────────────────────────

def build_knowledge_base(api_key: str, progress_callback=None) -> chromadb.Collection:
    """
    Fetch Wikipedia content for every SDG, embed it,
    and persist into ChromaDB. Returns the collection.
    """
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # If already populated, skip rebuild
    if collection.count() > 0:
        return collection

    total = len(SDG_TOPICS)
    for idx, (sdg_id, display_name, wiki_title) in enumerate(SDG_TOPICS):
        if progress_callback:
            progress_callback(idx / total, f"Loading {sdg_id}: {display_name}…")

        text = fetch_wikipedia(wiki_title)
        if not text:
            continue

        chunks = chunk_text(text)
        for c_idx, chunk in enumerate(chunks):
            try:
                emb = get_embedding(chunk, api_key)
                collection.add(
                    documents=[chunk],
                    embeddings=[emb],
                    ids=[f"{sdg_id}_{c_idx}"],
                    metadatas=[{
                        "sdg_id":       sdg_id,
                        "display_name": display_name,
                        "source":       f"Wikipedia: {wiki_title}",
                    }],
                )
            except Exception:
                continue   # skip if embedding fails

    if progress_callback:
        progress_callback(1.0, "✅ Knowledge base ready!")
    return collection


def query_knowledge_base(collection: chromadb.Collection,
                         query: str,
                         api_key: str) -> str:
    """
    Embed the query and retrieve TOP_K relevant chunks.
    Returns formatted context string.
    """
    if collection.count() == 0:
        return "Knowledge base is empty."

    try:
        q_emb    = get_embedding(query, api_key)
        results  = collection.query(
            query_embeddings=[q_emb],
            n_results=min(TOP_K, collection.count()),
            include=["documents", "metadatas"],
        )
        docs  = results["documents"][0]
        metas = results["metadatas"][0]

        context = ""
        for doc, meta in zip(docs, metas):
            context += (
                f"\n[{meta['sdg_id']} – {meta['display_name']}]\n"
                f"Source: {meta['source']}\n"
                f"{doc}\n"
                f"{'─'*60}\n"
            )
        return context.strip()
    except Exception as e:
        return f"RAG query failed: {e}"
