# WaveID Platform Prototype

This repository contains a minimal prototype for the WaveID platform,
an audio identification system designed to recognise heavily
transformed music clips in short‑form media. It accompanies the
deliverable D1 and implements a scaffold ready for extension in
Deliverable 2.

## Structure

- `waveid_backend/` – FastAPI backend application with endpoints for
  ingesting reference tracks and querying short audio clips. It uses
  stub implementations for embedding extraction and vector search,
  which can be replaced with real models and FAISS indices in later
  iterations.
- `requirements.txt` – Python dependencies.

## Running the Backend

Create a virtual environment and install the requirements:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start the server using Uvicorn:

```bash
uvicorn waveid_backend.main:app --reload
```

You can then test the API endpoints:

- `GET /health` returns a health check.
- `POST /ingest-track` accepts an audio file and stores dummy
  embeddings.
- `POST /query` accepts a clip and returns dummy similarity
  results.

In the future the `services` subpackage should be extended to
implement real audio preprocessing, embedding models and FAISS
indices. The data model, storage and evaluation harness described in
Deliverable D1 can be layered on top of this scaffold.