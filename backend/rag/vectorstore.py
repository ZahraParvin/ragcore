"""Chroma vector store wrapper."""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path

CHROMA_PATH = Path("chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"

_client = None  # chromadb.PersistentClient
_embedder = None  # SentenceTransformer


def _get_client() -> chromadb.Client:
    global _client
    if _client is None:
        CHROMA_PATH.mkdir(exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_or_create_collection(name: str) -> chromadb.Collection:
    return _get_client().get_or_create_collection(name)


def add_documents(collection_name: str, chunks: list[str], doc_id: str) -> int:
    collection = get_or_create_collection(collection_name)
    embedder = _get_embedder()
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc_id, "chunk": i} for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
    return len(chunks)


def query(collection_name: str, question: str, n_results: int = 5) -> list[dict]:
    collection = get_or_create_collection(collection_name)
    embedder = _get_embedder()
    query_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({"text": doc, "source": meta.get("source", ""), "score": 1 - dist})
    return output


def list_collections() -> list[str]:
    return [c.name for c in _get_client().list_collections()]


def delete_collection(name: str) -> None:
    _get_client().delete_collection(name)


def collection_count(name: str) -> int:
    try:
        return get_or_create_collection(name).count()
    except Exception:
        return 0
