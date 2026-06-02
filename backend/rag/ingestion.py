"""Document ingestion: load, chunk, and prepare documents for embedding."""
import io
import pandas as pd
import pdfplumber
from pathlib import Path


def load_text(file_bytes: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(file_bytes)
    elif suffix in (".csv", ".tsv"):
        return _load_csv(file_bytes, suffix)
    elif suffix in (".txt", ".md"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        return file_bytes.decode("utf-8", errors="ignore")


def _load_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n\n".join(text_parts)


def _load_csv(file_bytes: bytes, suffix: str) -> str:
    sep = "\t" if suffix == ".tsv" else ","
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
    summary = f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns\nColumns: {', '.join(df.columns)}\n\n"
    stats = df.describe(include="all").to_string()
    sample = df.head(20).to_string(index=False)
    return summary + "Statistics:\n" + stats + "\n\nSample rows:\n" + sample


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
