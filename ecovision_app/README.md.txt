# EcoVision

A FastAPI + SQLite backend with a Next.js dashboard for macroeconomic indicators, AI RAG search, and simple forecasting.

## Tech
- **Backend**: FastAPI, SQLAlchemy, APScheduler, requests
- **DB**: SQLite (auto-created as `backend/ecovision.db`)
- **AI**: Local Ollama (qwen2.5:7b-instruct for chat, mxbai-embed-large for embeddings)
- **Frontend**: Next.js (App Router), Recharts, Tailwind

## Local Setup

### 1) Backend
```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
# Run API
uvicorn backend.main:app --reload
# API -> http://127.0.0.1:8000
