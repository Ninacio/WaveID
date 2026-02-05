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

## Dataset Ingestion (Offline)

Reference catalogues should be ingested offline instead of uploading
thousands of tracks through the API. This is faster, more reliable,
and mirrors production systems where ingestion is a batch process and
the API handles only short query clips.

Use the CLI script:

```bash
python -m scripts.ingest_dataset
```

By default it looks for datasets under `datasets/gtzan/genres_original`
relative to the repo root. You can override the location and set limits:

```bash
python -m scripts.ingest_dataset --dataset "datasets/GTZAN/genres_original" --limit 10
```

## Dataset Layout

Place raw datasets under a top-level `datasets/` folder. Layout:

```
datasets/
  gtzan/
    genres_original/
      blues/
      classical/
      ...
  fma/
    small/
    medium/
```

In the future the `services` subpackage should be extended to
implement real audio preprocessing, embedding models and FAISS
indices. The data model, storage and evaluation harness described in D1 can be layered on top of this scaffold.