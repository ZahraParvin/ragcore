"""RAG chain: retrieve context then generate with Claude."""
import os
import anthropic
from . import vectorstore

CLAUDE_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are an intelligent document assistant. You answer questions based on the provided context retrieved from the user's documents.

Rules:
- Answer only from the provided context. If the context is insufficient, say so clearly.
- Cite your sources by referencing the document name when relevant.
- Be concise and precise.
- For numerical data or sensor readings, highlight anomalies or patterns when asked.
- For financial data, be careful and note that this is not financial advice."""


def rag_query(
    collection_name: str,
    question: str,
    chat_history: list[dict] | None = None,
    n_results: int = 5,
    use_tools: bool = True,
) -> dict:
    """Run a RAG query: retrieve relevant chunks then generate a response."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Retrieve context
    chunks = vectorstore.query(collection_name, question, n_results=n_results)
    context = "\n\n---\n\n".join(
        [f"[Source: {c['source']}]\n{c['text']}" for c in chunks]
    )

    # Build messages
    messages = []
    if chat_history:
        messages.extend(chat_history[-6:])  # keep last 3 turns

    messages.append({
        "role": "user",
        "content": f"Context from documents:\n\n{context}\n\n---\n\nQuestion: {question}",
    })

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    answer = response.content[0].text
    sources = list({c["source"] for c in chunks})

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
        "model": CLAUDE_MODEL,
    }
