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
- `POST /ingest-track` accepts an audio file, segments it, stores
  baseline embeddings and writes catalogue metadata.
- `POST /query` accepts a clip, segments it, searches nearest
  neighbours and returns ranked track-level matches.

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