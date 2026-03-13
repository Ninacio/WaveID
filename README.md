# WaveID
A logbook for WaveID
# WaveID - Final Year Project Logbook & Repository

Welcome to the GitHub repository for **WaveID**, my final-year project:  
*A robust audio-identification system for short-form and transformed media content.*

This repository serves as the central hub for:
- Planning  
- Research notes  
- Implementation code  
- Experiments  
- Supervisor meetings  
- Weekly activity logbook  

---

## 📁 Repository Structure

| Folder | Purpose |
|--------|---------|
| `01-Planning/` | Roadmaps, Gantt charts, milestones |
| `02-Research/` | Papers, annotated notes |
| `03-Design/` | Architecture, design documentation |
| `04-Implementation/` | Code, models, notebooks |
| `05-Testing/` | Test plans, evaluation results |
| `06-Meetings/` | Supervisor meeting summaries |
| `/docs` | Final deliverables, dissertation PDFs |

---

## 📘 Logbook

All weekly activity logs are kept here:

👉 **[LOGBOOK.md](LOGBOOK.md)**

Each entry includes tasks completed, issues encountered, next steps, and supervisor notes.

---

## 🧰 Demo & API

**Demo workflow and API reference:** See [waveid_platform/README.md](waveid_platform/README.md) for:
- Step-by-step demo (ingest → start backend → query)
- **Web UI** at `http://localhost:8000` – upload clips, view matches
- Full API reference with `curl` examples
- Swagger UI at `http://localhost:8000/docs`

**Quick utilities:**

- `waveid_platform/scripts/compare_audio.py` — compare two audio files by SHA‑256 + basic stats  
  ```bash
  cd waveid_platform && python -m scripts.compare_audio --file-a "path/to/a.wav" --file-b "path/to/b.wav"
  ```

- `waveid_platform/scripts/run_eval_pipeline.py` — one-command evaluation demo  
  ```bash
  cd waveid_platform && python -m scripts.run_eval_pipeline --reference "path/to/blues.00000.wav" --max-seconds 5 --max-query-segments 1 --top-k 3
  ```

---

## 🧠 Project Summary

WaveID aims to detect copyrighted audio in short-form content even when it has been  
**pitch-shifted, time-stretched, cropped, layered or mixed**.  
The system uses:
- deep audio embeddings  
- contrastive learning  
- FAISS similarity search  
- transformation robustness testing  
