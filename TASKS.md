# WaveID Task Board

This file mirrors a lightweight sprint board with clear task titles
and time estimates. Will be updated as I plan each sprint.

## Backlog

- [ ] Define ingestion constraints (formats, limits) - 2h
- [ ] Add query pipeline tests - 4h
- [ ] Build evaluation scripts (pitch/tempo/noise) - 6h
- [ ] Baseline metrics report - 4h
- [ ] Add CHANGELOG.md and record milestones - 1h
- [ ] Document dataset licensing and usage notes - 1h
- [ ] Integrate pre-trained embeddings (OpenL3/VGGish) - 1–2 days
- [ ] Implement FAISS index (replace in-memory search) - 1–2 days
- [ ] Add database for metadata (SQLite/Postgres) - 1–2 days
- [ ] Add model version switching in API - 4h
- [ ] Add query window aggregation + re-ranking - 4h
- [ ] Add evaluation endpoint (/evaluation/{id}) - 4h
- [ ] Build transformation suite (pitch/tempo/noise/crop) - 1–2 days
- [ ] Generate robustness metrics + report (CSV/HTML) - 1 day
- [ ] Create contrastive training data pairs - 1–2 days
- [ ] Implement contrastive model training loop - 2–3 days
- [ ] Compare baseline vs contrastive model - 1 day
- [ ] Minimal frontend UI for upload + results - 1–2 days
- [ ] Add admin ingestion dashboard (optional) - 1 day
- [ ] Add robustness sandbox (optional) - 1–2 days
- [ ] Write unit tests for services - 1–2 days
- [ ] Add API integration tests - 1 day
- [ ] Add Dockerfile + Docker Compose - 1 day
- [ ] Update README with API usage + examples - 4h

## Sprint TODO

- [ ] Ingest 1 genre from GTZAN - 2h
- [ ] Verify query matches from ingested set - 3h
- [ ] Draft evaluation plan - 2h

## Done

- [x] Audio decoding + segmentation check - 2h
- [x] Baseline embeddings (MFCC stats) - 3h
- [x] Dataset ingest CLI improvements - 3h
- [x] Add persistence for catalogue/index - 3h
