# WaveID Meeting Notes

## Meeting notes - Draft Entry (will be expanded on later)

**Date:** 02/02/2025 
**Agenda:**
- Ingestion pipeline progress
- Persistence and catalogue sharing
- Next evaluation steps

**Summary since last meeting:**
- I implemented a central config, added data folders, and built the decoding/segmentation checks.
- I added baseline MFCC embeddings, a dataset ingestion CLI, and disk persistence for catalogue/index.
- I verified WAV/MP3 ingestion and confirmed the API reads the stored catalogue.

**Key discussion points:**
- Keep batch ingestion offline via CLI and reserve the API for query clips.
- Start evaluation scripts for robustness testing next.

**Decisions:**
- Continue with offline ingestion workflow.

**Action items:**
- Ingest a larger GTZAN subset and sanity‑check retrieval.
- Draft evaluation script plan (pitch/tempo/noise).

---
