"""RAGCore FastAPI backend."""
import hashlib
import os
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from rag.ingestion import load_text, chunk_text
from rag import vectorstore
from rag.chain import rag_query
from tools.data_tools import agentic_data_query

app = FastAPI(title="RAGCore", version="1.0.0", description="Multi-domain RAG document intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent.parent / "frontend" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------- Models ----------

class QueryRequest(BaseModel):
    collection: str
    question: str
    history: list[dict] = []
    n_results: int = 5


class AgentQueryRequest(BaseModel):
    question: str


# ---------- Routes ----------

@app.get("/")
def root():
    index = static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "RAGCore API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY"))}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form(...),
):
    contents = await file.read()
    doc_id = hashlib.md5(contents).hexdigest()[:8] + "_" + Path(file.filename).stem

    text = load_text(contents, file.filename)
    if not text.strip():
        raise HTTPException(400, "Could not extract text from file.")

    chunks = chunk_text(text)
    n = vectorstore.add_documents(collection, chunks, doc_id)
    return {"doc_id": doc_id, "chunks_added": n, "collection": collection}


@app.post("/query")
def query_documents(req: QueryRequest):
    count = vectorstore.collection_count(req.collection)
    if count == 0:
        raise HTTPException(404, f"Collection '{req.collection}' is empty. Upload documents first.")
    return rag_query(req.collection, req.question, req.history, req.n_results)


@app.post("/agent/analyze")
async def agent_analyze(file: UploadFile = File(...), question: str = Form(...)):
    contents = await file.read()
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Agent analysis currently supports CSV files only.")
    return agentic_data_query(contents, question)


@app.get("/collections")
def list_collections():
    cols = vectorstore.list_collections()
    return {"collections": [{"name": c, "count": vectorstore.collection_count(c)} for c in cols]}


@app.delete("/collections/{name}")
def delete_collection(name: str):
    vectorstore.delete_collection(name)
    return {"deleted": name}
