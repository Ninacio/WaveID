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

---

## Demo Workflow

End-to-end flow for demonstrating WaveID: ingest reference tracks, run the backend, and query with a short clip.

### 1. Prepare the environment

```bash
cd waveid_platform
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ingest reference tracks (choose one)

**Option A – Via API (interactive demo):** Start the server, then upload tracks with `curl` or Postman (see API Reference below).

**Option B – Via CLI (recommended for many tracks):**

```bash
python -m scripts.ingest_dataset --dataset "../datasets/GTZAN/genres_original" --limit 5
```

This ingests up to 5 tracks per genre into the catalogue. Run this *before* starting the backend so the index is populated; the backend loads the persisted catalogue on startup.

### 3. Start the backend

```bash
uvicorn waveid_backend.main:app --reload --host 0.0.0.0 --port 8000
```

- **Web UI:** `http://localhost:8000` – minimal frontend for upload and results
- **API docs:** `http://localhost:8000/docs`

### 4. Query with a clip

Upload a short audio clip (WAV, MP3, or AU) that matches one of the ingested tracks (or a transformed variant):

```bash
curl -X POST "http://localhost:8000/query" \
  -H "accept: application/json" \
  -F "file=@path/to/query_clip.wav"
```

The response lists top-k matched tracks with similarity scores.

### 5. Verify the catalogue

```bash
curl "http://localhost:8000/catalogue"
```

---

## API Reference

Base URL: `http://localhost:8000` (default). All endpoints return JSON unless noted.

### `GET /health`

Health check. Use to verify the server is running.

**Example:**

```bash
curl "http://localhost:8000/health"
```

**Response:**

```json
{"status": "ok"}
```

---

### `POST /ingest-track`

Ingest a reference track. Upload an audio file (WAV, MP3, or AU). The file is segmented, embedded, and added to the catalogue. Max file size: 50 MB; max duration: 10 minutes.

**Example:**

```bash
curl -X POST "http://localhost:8000/ingest-track" \
  -H "accept: application/json" \
  -F "file=@path/to/reference_track.wav"
```

**Response:**

```json
{
  "message": "Ingested reference_track.wav",
  "track_id": "abc123-def456-...",
  "num_segments": 12,
  "duration_seconds": 30.0
}
```

---

### `POST /query`

Identify an unknown audio clip. Upload a short clip; the server returns the top-k matched reference tracks with similarity scores.

**Example:**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "accept: application/json" \
  -F "file=@path/to/query_clip.wav"
```

**Response:**

```json
{
  "query_embedding": [0.1, -0.2, ...],
  "matches": [
    {
      "track_id": "abc123-def456-...",
      "filename": "reference_track.wav",
      "score": 0.95,
      "hits": 3
    }
  ]
}
```

- `score`: Average similarity across matched segments (higher = better match).
- `hits`: Number of query segments that matched this track.

---

### `GET /catalogue`

List all ingested tracks.

**Example:**

```bash
curl "http://localhost:8000/catalogue"
```

**Response:**

```json
[
  {
    "track_id": "abc123-def456-...",
    "filename": "blues.00000.wav",
    "num_segments": 15,
    "duration": 30.0
  }
]
```

---

### `GET /catalogue/{track_id}`

Get metadata and segment list for a specific track.

**Example:**

```bash
curl "http://localhost:8000/catalogue/abc123-def456-..."
```

**Response:**

```json
{
  "track_id": "abc123-def456-...",
  "filename": "blues.00000.wav",
  "num_segments": 15,
  "duration": 30.0,
  "segments": [
    {
      "segment_id": "seg_001",
      "start_time": 0.0,
      "end_time": 2.0,
      "embedding_id": "emb_xyz"
    }
  ]
}
```

---

### Postman / Swagger

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

Use the "Try it out" buttons to upload files and test endpoints interactively.

---

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

## Query Smoke Test

Run a quick end-to-end check without the API:

```bash
python -m scripts.query_smoke_test --reference "path/to/ref.wav" --query "path/to/query.wav"
```

For quicker checks on larger catalogues, limit query windows:

```bash
python -m scripts.query_smoke_test --reference "path/to/ref.wav" --query "path/to/query.wav" --max-query-segments 5
```

## Transformation Generator (Evaluation Prep)

Generate transformed variants (pitch, tempo, noise, crop) from one clip:

```bash
python -m scripts.evaluate_transformations --input "path/to/audio.wav" --output-dir "data/query/eval"
```

Quick sanity run on a short excerpt:

```bash
python -m scripts.evaluate_transformations --input "path/to/audio.wav" --output-dir "data/query/eval" --max-seconds 5
```

**From preloaded catalogue:** Generate transforms for all ingested reference tracks in one go:

```bash
python -m scripts.generate_transforms_from_catalogue
```

Options: `--limit 5` (process only 5 tracks), `--max-seconds 5` (shorter clips), `--dry-run` (show plan without running).

## Evaluation Runner (CSV Report)

Run transformed clips through query matching and save a CSV summary:

```bash
python -m scripts.run_evaluation --reference "path/to/ref.wav" --queries-dir "data/query/eval/blues00000_short" --output-csv "data/index/eval_results.csv" --fresh-index
```

Summarise the results (overall and by transform):

```bash
python -m scripts.summarise_evaluation --input-csv "data/index/eval_results.csv" --output-csv "data/index/eval_summary.csv" --severity-output-csv "data/index/eval_summary_severity.csv" --report-md "data/index/eval_report.md"
```

Run all three steps in one command:

```bash
python -m scripts.run_eval_pipeline --reference "path/to/ref.wav" --max-seconds 5 --max-query-segments 1 --top-k 3 --fresh-index
```

Run a multi-reference sweep:

```bash
python -m scripts.run_evaluation_sweep --references-dir "path/to/genre_folder" --limit-references 3 --max-seconds 5 --max-query-segments 1 --limit-queries 3 --top-k 3 --fresh-index
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