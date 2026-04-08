# 📘 WaveID Project Logbook

This logbook contains a chronological record of all work completed for the WaveID  
Final-Year Project. Each week includes:

- Completed tasks  
- Challenges / blockers  
- Solutions / decisions  
- Next week's plan  
- Meeting prep summary  
- Supervisor meeting summary  

---

# 📝 Log Entry

## Week 1 - 29/09/2025 to 05/10/2025

### ✔️ Completed This Week
- Was assigned a supervisor and scheduled days for meeting
- Conducted research on potential topic to switch to
- N/A

### ⚠️ Challenges / Issues
- Dealing with external issues such as copyright
- N/A

### 💡 Solutions / Decisions Made
- Finding Free use assets
- N/A

### 📌 Next Week’s Plan
- Come up with a definitive topic
- N/A

### 🧭 Meeting Prep Summary
- Summary since last meeting:
- Decisions/questions:

### 👥 Supervisor Meeting
- Date: 30/09/2025
- Key points: Finding a more suitable topic
- Actions: Conducted more research on new topics

## Week 2 - 06/10/2025 to 12/10/2025

### ✔️ Completed This Week
- Whittled down potential options to 2
- N/A
- N/A

### ⚠️ Challenges / Issues
- Deciding which topic would be most feasable 
- N/A

### 💡 Solutions / Decisions Made
- Choose topic by next week
- N/A

### 📌 Next Week’s Plan
- Begin work on chosen topic
- N/A

### 👥 Supervisor Meeting
- Date: N/A
- Key points: N/A
- Actions: N/A

## Week 3 - 13/10/2025 to 19/10/2025

### ✔️ Completed This Week
- Chose a definitive topic
- Initiated background research
- N/A

### ⚠️ Challenges / Issues
- N/A 
- N/A

### 💡 Solutions / Decisions Made
- Topic chosen
- N/A

### 📌 Next Week’s Plan
- Begin work on chosen topic
- N/A

### 👥 Supervisor Meeting
- Date: 17/10/2025
- Key points: Starting point for dissertation
- Actions: Operation begun

## Week 4 - 20/10/2025 to 26/10/2025

### ✔️ Completed This Week
- Background research conducted
- Ethics form completed
- N/A

### ⚠️ Challenges / Issues
- N/A 
- N/A

### 💡 Solutions / Decisions Made
- N/A
- N/A

### 📌 Next Week’s Plan
- Continue background research
- Begin write-up

### 👥 Supervisor Meeting
- Date: 21/10/2025
- Key points: Guidance on ethics form
- Actions: Ethic form completed

## Week 5 - 27/10/2025 to 02/11/2025

### ✔️ Completed This Week
- First draft in the works
- More background is being conducted alongside write-up
- N/A

### ⚠️ Challenges / Issues
- Ethics form denied, more clarification was needed 
- N/A

### 💡 Solutions / Decisions Made
- Provided more information where necessary and resubmitted
- N/A

### 📌 Next Week’s Plan
- Continue background research
- Continue with write-up

### 👥 Supervisor Meeting
- Date: N/A
- Key points: N/A
- Actions: N/A

## Week 6 - 03/11/2025 to 09/11/2025

### ✔️ Completed This Week
- Ethics form accepted by supervisor
- Lit review drafted
- N/A

### ⚠️ Challenges / Issues
- N/A
- N/A

### 💡 Solutions / Decisions Made
- N/A
- N/A

### 📌 Next Week’s Plan
- Complete Structure of Dissertation
- N/A

### 👥 Supervisor Meeting
- Date: 07/11/2025
- Key points: For Literature review:
    - Compare and contrast papers, what works, what limitations there are and the introduce my own project at the end.
    - Make sure we have a strong background research & strong methodology.
    - Since we are still reading we don't need to know exactly how things should be done.
    - Make sure to look at other methodologies when doing background research.
    - Make sure to know what you are looking to assess.
- Actions: Applied advice.

## Week 7 - 10/11/2025 to 16/11/2025

### ✔️ Completed This Week
- Dissertation rounded out
- N/A
- N/A

### ⚠️ Challenges / Issues
- N/A
- N/A

### 💡 Solutions / Decisions Made
- N/A
- N/A

### 📌 Next Week’s Plan
- Complete & polish Dissertation
- N/A

### 👥 Supervisor Meeting
- Date: 14/11/2025
- Key points: Took feedback from supervisor and used it to improve dissertation

## Week 8 - 17/11/2025 to 23/11/2025

### ✔️ Completed This Week
- Dissertation Completed and polished
- Other Deliverables completed
- N/A

### ⚠️ Challenges / Issues
- N/A
- N/A

### 💡 Solutions / Decisions Made
- N/A
- N/A

### 📌 Next Week’s Plan
- Move onto implementation
- N/A

### 👥 Supervisor Meeting
- Date: 18/11/2025
- Key points: Final feedback from supervisor, finishing touches made

---

## Christmas Break (Setup / Minor Changes)
- Implemented a central `config.py` with shared paths and ingestion settings
- Added data folders for reference, query, index, and embeddings
- Built audio decoding + segmentation verification script and tested WAV/MP3

---

## Semester 2 - Week 3 (26/01/26 to 01/02/26)

### ✔️ Completed This Week
- Implemented dataset ingestion via CLI and a reusable dataset loader
- Added baseline MFCC embeddings and an in‑memory search pipeline
- Added persistence for catalogue and embeddings on disk
- Verified ingestion and catalogue listing through the API

### ⚠️ Challenges / Issues
- Initial MP3 decoding failed until ffmpeg was installed and the terminal restarted
- API and CLI used separate in‑memory catalogues before persistence was added

### 💡 Solutions / Decisions Made
- Standardised ingestion parameters in config for deterministic behaviour
- Persisted catalogue/index to disk to share state between CLI and API
- Use CLI for bulk reference ingestion and keep API for query clips

### 📌 Next Week’s Plan
- Ingest a larger GTZAN subset and validate retrieval quality
- Add a simple query smoke test using an ingested track
- Start evaluation scripts for pitch/tempo/noise transformations

### 🧭 Meeting Prep Summary
- Summary since last meeting: I built the core ingestion pipeline (decode, segment, embed, index) with CLI ingestion and disk persistence, and verified that the API can read the stored catalogue.
- Decisions/questions: I will keep batch ingestion offline and use the API only for query clips; next step is to add evaluation tooling.

### 👥 Supervisor Meeting
- Date: N/A
- Key points: N/A
- Actions: N/A

## Semester 2 - Weeks 4–8 (02/02/2026 to 08/03/2026)

### ✔️ Completed This Week (period summary)
- Query pipeline with top-k retrieval and scores; evaluation scripts for pitch, tempo, noise, crop, filtering, compound transforms, and low-bitrate MP3 (in test battery and contrastive training augmentations)
- Contrastive encoder trained on multi-genre GTZAN data (all ten genres) with evaluation-aligned augmentations; comparison vs MFCC baseline in reports
- Severity sweeps (varying strength per transform) with main summary destined for Chapter 7 and extended plots in Appendix B
- Demo case documented: heavy pitch misrank with MFCC on blues subset; contrastive model restores correct top-1 on five-track catalogue

### ⚠️ Challenges / Issues
- Balancing dissertation page limits with full sweep detail (appendix for long graphs)
- Tuning thresholds and reporting when comparing learned embeddings vs classical fingerprints

### 💡 Solutions / Decisions Made
- Keep Chapter 7 readable; put full severity figures and Chromaprint per-transform notes in appendices
- Use fixed Chromaprint threshold (0.35) consistently in tables for comparability

### 📌 Next Week’s Plan (at time)
- External benchmark write-up; cross-genre combined index; final dissertation integration

### 🧭 Meeting Prep Summary
- Summary since last meeting: End-to-end evaluation pipeline, multi-genre contrastive training, MP3 and sweeps in code and thesis
- Decisions/questions: How to present per-genre 100% vs cross-genre ~54% without overstating real-world performance

### 👥 Supervisor Meeting
- Date: N/A
- Key points: N/A
- Actions: N/A

## Semester 2 - Weeks 9–11 (09/03/2026 to 29/03/2026)

### ✔️ Completed This Week (period summary)
- Chromaprint 1.6 benchmark on same clips as WaveID (blues table and all ten genres); written up in Evaluation Chapter 7 (Tables 7.4–7.5)
- Cross-genre combined-index evaluation: 50 reference tracks, 850 queries, ~53.8% vs ~54.6% top-1 (baseline vs contrastive); score compression discussion (Table 7.8)
- Dissertation D3 aligned with department handbook: section order, references placement, short Generative AI summary, Project Proposal and D1-to-D3 changes section
- Abstract and Introduction updated to state the two evaluation scopes (small per-genre catalogue vs large mixed list)

### ⚠️ Challenges / Issues
- Explaining why contrastive scores cluster near 0.99 under cross-genre search while still reporting nuanced robustness gains in per-genre tests

### 💡 Solutions / Decisions Made
- Lead with two-scope story in abstract; point readers to Chapter 8 for interpretation
- No Dejavu benchmark in scope; Chromaprint as primary external check for the dissertation

### 📌 Next Week’s Plan
- Final PDF checks; supervisor update; submission prep

### 🧭 Meeting Prep Summary
- Summary since last meeting: Full evaluation chapter, appendices for sweeps and extended Chromaprint commentary, benchmark and cross-genre results integrated
- Decisions/questions: Optional supervisor email with PDF table/figure pointers (Chapter 7 and Appendix B)

### 👥 Supervisor Meeting
- Date: N/A
- Key points: N/A
- Actions: N/A

## April 2026 - Week 1 (01/04/2026 to 07/04/2026)

### ✔️ Completed This Week
- Final dissertation PDF assembled (WaveID_Diss.pdf, March 2026 submission build)
- Draft email to supervisor summarising Chromaprint comparison, severity sweeps, MP3 in battery, multi-genre training, 50-track cross-genre results, and handbook housekeeping; includes pointers to abstract, intro, Chapter 7, and discussion where useful
- Logbook brought up to date to Semester 2 completion
- Contacted supervisor (Babis) for advice on dissertation content; confirmed code and frontend screenshots go in appendix, and that pseudocode algorithms are the appropriate form for crucial components in the main body
- Enforced Main Body (Chapters 6-8) 20-page limit: condensed evaluation and discussion sections, moved extended Chromaprint all-genre table to Appendix B.3
- Reviewed and improved layman-friendliness of Chapters 7 and 8
- Enforced D1 Body (Chapters 1-4) ~14-page target: condensed Chapters 2, 3, and 4; moved success criteria table to Appendix A; removed duplicate and verbose content
- Added three pseudocode algorithms to Chapter 6 (track ingestion, contrastive triplet training, query and similarity search); simplified to be accessible to a non-specialist reader
- Created Appendix C with frontend screenshot placeholders and four representative Python code listings (MFCC embedding, AudioEncoder, triplet loss, cosine similarity search); all comments written in layman-friendly language
- Added layman-friendly docstrings and inline comments to all service modules: `embedding.py`, `contrastive_model.py`, `search.py`, `transforms.py`, `audio_io.py`, `segmentation.py`, `catalogue.py`, `dataset_loader.py`
- Added layman-friendly inline comments to all scripts: `train_contrastive.py`, `create_contrastive_data.py`, `run_evaluation.py`, `summarise_evaluation.py`, `benchmark_chromaprint.py`, `cross_genre_eval.py`
- Git repository configured to exclude dissertation files; large embedding `.npy` files added to `.gitignore`; code committed and pushed to GitHub

### ⚠️ Challenges / Issues
- Git push failed due to large `.npy` embedding files exceeding GitHub's per-file limit

### 💡 Solutions / Decisions Made
- Added `waveid_platform/data/embeddings/` to `.gitignore` and reset the commit before re-pushing

### 📌 Next Steps
- Replace frontend screenshot placeholders in Appendix C with real screenshots showing ingested tracks and query results
- Sign and date the Declaration page on the final compiled PDF
- Final PDF recompilation and page count verification before submission

### 👥 Supervisor Meeting
- Date: TBD (email sent requesting feedback)
- Key points: Confirmed appendix placement for screenshots and code; pseudocode only in main body
- Actions: Implemented all advice; awaiting further feedback

## April 2026 - Week 2 (08/04/2026)

### ✔️ Completed This Week
- Captured frontend screenshots and saved to `Figures/` directory
- Replaced all three placeholder `\fbox` blocks in Appendix C with `\includegraphics` pointing to actual PNG screenshots
- Fixed figure float placement in Appendix C: changed `[ht]` to `[H]` and added `\clearpage` before code listings to prevent Figure C.3 from drifting into Listing C.1
- Replaced all em dashes (`---`) used as punctuation throughout the dissertation with ` - ` per style preference
- Fixed chapter reference inconsistency in the Dissertation Structure paragraph of Chapter 1: Chapters 9, 10, and 11 now use the same `Chapter X (\ref{...})` format as Chapters 2-8
- Added a FastAPI startup event to reset the catalogue on every server boot, preventing stale ingested tracks from persisting across demo sessions

### ⚠️ Challenges / Issues
- Browser caching prevented updated JavaScript from loading after API changes
- FastAPI route `GET /catalogue/{track_id}` intercepted a reset endpoint before it could be matched, causing 405 errors

### 💡 Solutions / Decisions Made
- Used hard-refresh (`Ctrl+Shift+R`) to bypass browser cache
- Moved reset logic to a `@app.on_event("startup")` handler that physically deletes the persisted catalogue and embedding files on boot, eliminating the routing conflict

### 📌 Next Steps
- Complete demo with real audio files and replace empty-state screenshots with result screenshots
- Final PDF recompilation and submission prep

# 🏁 Project Timeline Overview

| Phase | Start | End | Status |
|-------|-------|------|--------|
| Research & Planning | Sep 2025 | Nov 2025 | Complete |
| Prototype Development | Nov 2025 | Feb 2026 | Complete |
| Evaluation | Feb 2026 | Mar 2026 | Complete |
| Final Dissertation (D3) | Mar 2026 | Apr 2026 | In Progress (final edits and submission prep) |

---

