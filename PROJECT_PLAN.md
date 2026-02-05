# WaveID Semester Plan

This plan breaks the semester into short sprints with clear outcomes,
time estimates, and a lightweight Gantt-style view.

## Sprint Plan (2-week cadence, structural plan subject to change)

| Sprint | Dates | Focus | Deliverables |
| --- | --- | --- | --- |
| S1 | TBD | Pipeline baseline | Audio IO, segmentation, baseline embeddings |
| S2 | TBD | Ingestion + index | Dataset ingest, persistent catalogue/index |
| S3 | TBD | Query + search | Query pipeline, retrieval, ranking |
| S4 | TBD | Evaluation | Robustness scripts, metrics, reports |
| S5 | TBD | Model iteration | Contrastive model prototype, comparisons |
| S6 | TBD | Finalization | Documentation, polishing, demo |

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
