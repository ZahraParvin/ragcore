# RAGCore — Multi-domain Document Intelligence

A production-ready RAG (Retrieval-Augmented Generation) system that lets you upload documents and query them intelligently using Claude AI. Supports industrial sensor data, financial reports, and general documents — with an agentic analysis mode for structured CSV data.

## Features

- **RAG pipeline**: PDF, CSV, and text ingestion → chunking → embedding (sentence-transformers) → Chroma vector store → Claude API generation with source citations
- **Agentic data analysis**: Claude uses function calling to run statistical analysis, detect anomalies, and filter datasets — not just summarise them
- **Multi-turn chat**: Conversation history maintained across queries
- **Domain presets**: Separate vector store collections for industrial, financial, and general documents
- **REST API**: FastAPI backend with full OpenAPI docs at `/docs`
- **UI**: Streamlit chat interface + HTML landing page
- **Containerised**: Docker Compose for local development; Azure Container Apps / Railway for deployment

## Architecture

```
User → Streamlit UI ──► FastAPI backend
                              │
               ┌──────────────┴──────────────┐
               │                             │
         RAG Pipeline                 Agent Pipeline
               │                             │
     Chroma vector DB              Claude tool use (function calling)
     sentence-transformers         pandas: stats, anomalies, filtering
               │                             │
          Claude API ◄────────────────────────┘
               │
      Answer + source citations
```

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Claude API (`claude-haiku-4-5`) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB (persistent) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Data parsing | pdfplumber, pandas |
| Containerisation | Docker + Docker Compose |
| Cloud deploy | Azure Container Apps / Railway |

## Quick Start

```bash
git clone https://github.com/ZahraParvin/ragcore
cd ragcore
cp .env.example .env        # add your ANTHROPIC_API_KEY
docker compose up           # API on :8000, Streamlit on :8501
```

Or run locally without Docker:

```bash
pip install -r requirements.txt
cd backend
uvicorn main:app --reload    # API on :8000

# in another terminal
cd frontend
streamlit run streamlit_app.py
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Ingest a document into a collection |
| `POST` | `/query` | RAG query with chat history |
| `POST` | `/agent/analyze` | Agentic CSV analysis with tool use |
| `GET` | `/collections` | List all collections and chunk counts |
| `DELETE` | `/collections/{name}` | Delete a collection |
| `GET` | `/health` | Health check |

## Usage Examples

**Upload and query a document:**
```python
import requests

# Upload
with open("report.pdf", "rb") as f:
    requests.post("http://localhost:8000/upload",
                  files={"file": f},
                  data={"collection": "financial"})

# Query
response = requests.post("http://localhost:8000/query", json={
    "collection": "financial",
    "question": "What were the main risk factors mentioned?",
    "history": []
})
print(response.json()["answer"])
```

**Agentic CSV analysis:**
```python
with open("sensor_data.csv", "rb") as f:
    response = requests.post("http://localhost:8000/agent/analyze",
                             files={"file": f},
                             data={"question": "Are there any anomalies in pressure readings?"})
print(response.json()["answer"])
# Also returns tool_calls showing exactly what Claude computed
```

## Deployment

### Azure Container Apps (free tier)
```bash
az containerapp up \
  --name ragcore \
  --resource-group mygroup \
  --environment myenv \
  --image ghcr.io/zahraparvin/ragcore:latest \
  --env-vars ANTHROPIC_API_KEY=secretref:anthropic-key
```

### Railway
Connect your GitHub repo at [railway.app](https://railway.app) — Railway auto-detects the Dockerfile and deploys on push.

## Project Structure

```
ragcore/
├── backend/
│   ├── main.py              # FastAPI app and routes
│   ├── rag/
│   │   ├── ingestion.py     # Document loading and chunking
│   │   ├── vectorstore.py   # Chroma wrapper
│   │   └── chain.py         # RAG chain with Claude
│   └── tools/
│       └── data_tools.py    # Agentic tools (function calling)
├── frontend/
│   ├── streamlit_app.py     # Chat UI
│   └── static/
│       └── index.html       # Landing page
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Design Decisions

- **sentence-transformers over API embeddings**: Free, runs locally, no per-token cost for indexing large document sets.
- **Chroma over Pinecone/Qdrant**: Persistent local vector store with zero external dependencies — easier to run and demo offline.
- **Claude Haiku**: Fast and cost-efficient for production use; swap to Sonnet/Opus in `chain.py` for more complex reasoning.
- **Function calling for agentic analysis**: Rather than summarising CSV data as text, Claude calls real pandas operations — anomaly detection uses true 3σ outlier detection, not approximation.

---

Built by [Zahra Parvin](https://linkedin.com/in/zahraparvin) | [GitHub](https://github.com/ZahraParvin)
"# ragcore" 
