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

# 🏁 Project Timeline Overview

| Phase | Start | End | Status |
|-------|-------|------|--------|
| Research & Planning | | | |
| Prototype Development | | | |
| Evaluation | | | |
| Final Dissertation | | | |

---

