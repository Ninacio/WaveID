# WaveID Semester Plan

This plan breaks the semester into short sprints with clear outcomes,
time estimates, and a lightweight Gantt-style view.

## Sprint Plan (2-week cadence, structural plan subject to change)

| Sprint | Dates | Focus | Deliverables |
| --- | --- | --- | --- |
| S1 | 26/01/26–08/02/26 | Pipeline baseline | Audio IO, segmentation, baseline embeddings |
| S2 | 09/02/26–22/02/26 | Ingestion + index | Dataset ingest, persistent catalogue/index |
| S3 | 23/02/26–08/03/26 | Query + search | Query pipeline, retrieval, ranking |
| S4 | 09/03/26–22/03/26 | Evaluation | Robustness scripts, metrics, reports |
| S5 | 23/02/26–01/03/26 | Model iteration | Contrastive model prototype, comparisons |
| S6 | 02/03/26–26/03/26 | Finalization | Documentation, polishing, demo |

## Week-Level Breakdown

### S1 — Pipeline Baseline

- Week 1 (26/01–01/02): Finalise ingestion parameters, verify decoding + resampling, and confirm segmentation counts on WAV/MP3.
- Week 2 (02/02–08/02): Implement baseline embeddings, wire them into ingest/query, and run a small end-to-end smoke test.

### S2 — Ingestion + Index

- Week 3 (09/02–15/02): Build dataset ingestion CLI, confirm batch ingest on GTZAN subset, and verify stored reference audio.
- Week 4 (16/02–22/02): Add persistence for catalogue/index and confirm API reads the stored catalogue across restarts.

### S3 — Query + Search

- Week 5 (23/02–01/03): Implement query pipeline (decode, embed, search) and return top‑k with confidence scores.
- Week 6 (02/03–08/03): Add simple re‑ranking/thresholding and a query smoke test script for regression checks.

### S4 — Evaluation

- Week 7 (09/03–15/03): Build transformation scripts (pitch/tempo/noise/crop) and define evaluation protocol.
- Week 8 (16/03–22/03): Run robustness sweeps and generate a baseline metrics report (top‑k, precision/recall).

### S5 — Model Iteration

- Model target: complete by end of February to allow March for polish, frontend, and documentation.
- Week 5 (23/02–01/03): Prepare contrastive training data (positive/negative pairs) and baseline training run.
- Week 6 (02/03–08/03): Compare contrastive model vs baseline embeddings and record results.

### S6 — Finalisation

- Week 9 (09/03–15/03): Frontend prototype polish and documentation updates.
- Week 10 (16/03–22/03): Demo workflow + final testing.
- Week 11 (23/03–29/03): Final checks and submission prep (Final Deliverable D3 due 26/03/26).

## Gantt-Style Task Breakdown

| Task | Start | End | Est. | Notes |
| --- | --- | --- | --- | --- |
| Audio decoding + resampling | TBD | TBD | 2 days | WAV/MP3 validation |
| Segmentation + validation | TBD | TBD | 2 days | 2s windows, 1s hop |
| Baseline embeddings | TBD | TBD | 3 days | MFCC stats |
| Dataset ingestion CLI | TBD | TBD | 2 days | Batch ingest |
| Catalogue persistence | TBD | TBD | 2 days | JSON + NPY |
| Query pipeline | TBD | TBD | 3 days | Encode, search, response |
| Evaluation scripts | TBD | TBD | 4 days | Pitch/tempo/noise |
| Metrics + reporting | TBD | TBD | 3 days | Top-k, F1 |
| Model training v1 | TBD | TBD | 5 days | Contrastive baseline |
| Documentation + demo | TBD | TBD | 3 days | README, demo notes |

## Plans to Update

- Replace TBD dates each time I plan a sprint.
- Keep estimates short and concrete.
- Move completed tasks into the logbook.
