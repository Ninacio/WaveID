# WaveID API Guide

A single reference for how the WaveID HTTP API works, how the React frontend uses it, and how that differs from internal Python scripts.

---

## Table of contents

1. [What is an API here?](#1-what-is-an-api-here)
2. [The two apps](#2-the-two-apps)
3. [Request anatomy](#3-request-anatomy)
4. [Endpoint reference](#4-endpoint-reference)
5. [Backend architecture](#5-backend-architecture)
6. [Frontend client (`api.ts`)](#6-frontend-client-apits)
7. [TanStack Query layer](#7-tanstack-query-layer)
8. [End-to-end: query flow](#8-end-to-end-query-flow)
9. [End-to-end: ingest flow](#9-end-to-end-ingest-flow)
10. [Development proxy](#10-development-proxy)
11. [Authentication](#11-authentication)
12. [What does *not* use HTTP](#12-what-does-not-use-http)
13. [Debugging checklist](#13-debugging-checklist)
14. [Hands-on exercises (curl)](#14-hands-on-exercises-curl)
15. [Key files](#15-key-files)

---

## 1. What is an API here?

Think of the API as a **waiter** between your **React app (the customer)** and the **Python backend (the kitchen)**.

The UI does not run the ML model or read `catalogue.json` directly. It sends **HTTP requests** to agreed URLs and receives **JSON** (or audio bytes) back.

Every interaction follows the same pattern:

| Piece | Meaning |
|-------|---------|
| **Method** | What you want to do (`GET` read, `POST` create/upload, `PATCH` update, `DELETE` remove) |
| **URL** | Which resource (`/catalogue`, `/query`, …) |
| **Body** | Optional payload (file upload via `FormData`, or JSON metadata) |
| **Headers** | Optional extras (`X-API-Key`, `Content-Type`) |
| **Response** | JSON object, or a streamed file |

Example in plain English:

> “Hey server, **POST** this audio file to **/query** and tell me which catalogue tracks match.”

---

## 2. The two apps

You always run **two programs** in development:

| App | Folder | Port | Role |
|-----|--------|------|------|
| **Backend (API server)** | `waveid_backend/` | 8000 | Receives requests, processes audio, returns JSON |
| **Frontend (UI)** | `waveid_frontend/` | 5173 | Pages, forms, calls the API, displays results |

```
  Browser (you)
       │
       ▼
  React app  :5173  ──HTTP──►  FastAPI  :8000  ──►  data/ (catalogue, embeddings, audio)
```

**Start the backend (PowerShell):**

```powershell
cd C:\Users\ninac\Documents\GitHub\WaveID\waveid_platform
.\.venv\Scripts\Activate.ps1
uvicorn waveid_backend.main:app --reload --port 8000
```

**Start the frontend:**

```powershell
cd C:\Users\ninac\Documents\GitHub\WaveID\waveid_platform\waveid_frontend
npm run dev
```

Open **http://localhost:5173** for the React UI. The backend also exposes interactive docs at **http://localhost:8000/docs**.

---

## 3. Request anatomy

### File uploads (`POST /query`, `POST /ingest-track`)

Browsers send **multipart form data**:

```
POST /query
Content-Type: multipart/form-data

file=<binary audio>
```

Optional ingest metadata uses additional form fields: `title`, `artist`, `isrc`, `tags`.

### JSON updates (`PATCH /catalogue/{track_id}`)

```
PATCH /catalogue/abc123
Content-Type: application/json

{ "title": "New Title", "artist": "Artist Name" }
```

### Errors

FastAPI returns JSON with a `detail` field on failure. The frontend `handle()` helper in `api.ts` reads that and throws a JavaScript `Error` with the message.

---

## 4. Endpoint reference

All routes are defined in `waveid_backend/main.py`.

| Method | Endpoint | Purpose | Auth required* |
|--------|----------|---------|----------------|
| `GET` | `/health` | Health check → `{ "status": "ok" }` | No |
| `GET` | `/docs` | Swagger UI (auto-generated) | No |
| `POST` | `/auth/verify` | Validate an API key | No (but server must have key configured) |
| `POST` | `/ingest-track` | Upload and index a reference track | Only if `WAVEID_REQUIRE_API_KEY=true` |
| `POST` | `/query` | Identify an unknown audio clip | No |
| `GET` | `/catalogue` | List all indexed tracks | No |
| `GET` | `/catalogue/{track_id}` | Track detail + segment list | No |
| `GET` | `/catalogue/{track_id}/audio` | Stream stored reference audio | No |
| `PATCH` | `/catalogue/{track_id}` | Update title, artist, ISRC, tags | Yes, when API key is configured |
| `DELETE` | `/catalogue/{track_id}` | Remove track, segments, embeddings, file | Yes, when API key is configured |
| `POST` | `/reset-catalogue` | Wipe entire catalogue | Yes, when API key is configured |

\* “Auth required when configured” means: if `WAVEID_API_KEY` is set on the server, protected routes reject requests without a valid `X-API-Key` header.

### Response shapes (frontend types in `api.ts`)

**`GET /catalogue`** → `CatalogueTrack[]`

```typescript
{
  track_id: string
  filename: string
  num_segments: number
  duration: number
  title: string
  artist: string
  isrc: string
  tags: string[]
}
```

**`GET /catalogue/{id}`** → `TrackDetail` (above + `segments[]` with times and embedding IDs)

**`POST /query`** → `QueryResponse`

```typescript
{
  query_embedding: number[]
  matches: QueryMatch[]
  confidence_gap: number
  confidence_label: "high" | "medium" | "low"
  query_duration: number
  similarity_gap: number
}
```

Each `QueryMatch` includes `similarity`, `coverage`, `match_strength` (`strong` | `moderate` | `weak`), and per-segment alignments.

**`POST /ingest-track`** → `IngestResponse`

```typescript
{
  message: string
  track_id: string
  num_segments: number
  duration_seconds: number
  duplicate_of: { track_id, filename, similarity } | null
}
```

---

## 5. Backend architecture

When a request hits an endpoint, FastAPI:

1. **Validates** input (file type, size, filename, track ID format)
2. **Calls service modules** in `waveid_backend/services/`:
   - `audio.py` — decode and segment audio
   - `embedding.py` — compute fingerprint vectors
   - `search.py` — vector similarity search
   - `catalogue.py` — metadata, segments, delete
3. **Returns** a Pydantic model serialized as JSON

The API layer in `main.py` is intentionally thin. Heavy logic lives in services.

**On disk:**

| Path | Contents |
|------|----------|
| `data/index/catalogue.json` | Track metadata |
| `data/index/` | Search index (embeddings) |
| `data/reference/` | Original uploaded audio files |
| `data/queries/` | Saved query uploads |

---

## 6. Frontend client (`api.ts`)

**All HTTP from React goes through one file:** `waveid_frontend/src/lib/api.ts`.

Stack:

```
UI component → TanStack Query → api.ts → fetch → FastAPI → JSON
```

### Base URL

```typescript
const API_BASE = import.meta.env.VITE_API_BASE ?? ""
```

Empty in dev → requests go to the same origin (`localhost:5173`) and Vite proxies them to the backend.

### Auth header helper

```typescript
function authHeaders(): Record<string, string> {
  const key = getApiKey()  // from localStorage
  return key ? { "X-API-Key": key } : {}
}
```

### Example: query a clip

```typescript
export async function queryClip(file: File): Promise<QueryResponse> {
  const form = new FormData()
  form.append("file", file)
  return handle(
    await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: authHeaders(),
      body: form,
    })
  )
}
```

Steps: wrap file in `FormData` → `POST /query` → optional API key → parse JSON or throw on error.

### API functions → endpoints

| Function | HTTP |
|----------|------|
| `getHealth()` | `GET /health` |
| `getCatalogue()` | `GET /catalogue` |
| `getTrack(id)` | `GET /catalogue/{id}` |
| `getTrackAudioUrl(id)` | URL for `GET /catalogue/{id}/audio` |
| `queryClip(file)` | `POST /query` |
| `ingestTrack(file)` | `POST /ingest-track` |
| `updateTrackMetadata(id, body)` | `PATCH /catalogue/{id}` |
| `deleteTrack(id)` | `DELETE /catalogue/{id}` |
| `resetCatalogue()` | `POST /reset-catalogue` |
| `verifyApiKey(key)` | `POST /auth/verify` |

### Which UI files call which functions

| File | API calls |
|------|-----------|
| `pages/query.tsx` | `queryClip` |
| `pages/catalogue.tsx` | `getCatalogue`, `deleteTrack`, `resetCatalogue` |
| `pages/dashboard.tsx` | `getCatalogue` |
| `pages/track-detail.tsx` | `getTrack`, `deleteTrack` |
| `pages/settings.tsx` | `getApiKey`, `setApiKey`, `verifyApiKey` |
| `components/catalogue/bulk-ingest.tsx` | `ingestTrack` |
| `components/catalogue/edit-metadata-dialog.tsx` | `updateTrackMetadata` |
| `components/query/match-card.tsx` | `getTrackAudioUrl` |
| `components/layout/health-pill.tsx` | `getHealth` |

Pages import from `@/lib/api` — they do not call `fetch` directly.

---

## 7. TanStack Query layer

Pages wrap `api.ts` with **TanStack Query** for loading state, caching, and refetch:

```typescript
const { data, isLoading } = useQuery({
  queryKey: ["catalogue"],
  queryFn: getCatalogue,
})
```

After ingest or delete, components invalidate the cache:

```typescript
queryClient.invalidateQueries({ queryKey: ["catalogue"] })
```

That triggers a fresh `GET /catalogue` without a full page reload.

---

## 8. End-to-end: query flow

When you upload a clip and click **Identify**:

### Step 1 — Frontend

`QueryPage` calls `queryClip(file)` → `POST /query` with the WAV/MP3 file.

### Step 2 — Backend

1. Validate upload (size, format, filename)
2. Decode audio → waveform
3. Split into ~2-second segments
4. Compute an **embedding** (fingerprint) per segment
5. Search the index for similar catalogue embeddings
6. Rank tracks; compute confidence, similarity gap, segment alignments
7. Save query file to `data/queries/`
8. Return JSON

Example response shape:

```json
{
  "matches": [
    {
      "track_id": "...",
      "filename": "example.mp3",
      "similarity": 0.987,
      "coverage": 1.0,
      "match_strength": "weak",
      "segments": [
        {
          "query_start": 0.0,
          "query_end": 2.0,
          "ref_start": 45.0,
          "ref_end": 47.0,
          "score": 0.98
        }
      ]
    }
  ],
  "confidence_label": "low",
  "query_duration": 21.0,
  "similarity_gap": 0.002
}
```

**Reading results:** High fingerprint similarity alone does not always mean “same song.” The UI uses `match_strength`, `confidence_label`, and `similarity_gap` to distinguish strong same-track matches from acoustically similar but different tracks.

### Step 3 — Frontend again

- Renders match cards with alignment visuals
- For playback: `getTrackAudioUrl(track_id)` → browser loads `GET /catalogue/{id}/audio`
- WaveSurfer draws the waveform from that URL

**Two API interactions for one feature:** one JSON call (match data), one binary stream (reference audio).

---

## 9. End-to-end: ingest flow

From **Catalogue** → bulk ingest → `ingestTrack(file)`:

```
POST /ingest-track
  body: multipart file
  optional form fields: title, artist, isrc, tags
```

### Backend steps

1. Validate file
2. Decode and segment audio
3. Embed each segment → add vectors to search index
4. Save metadata to `catalogue.json`
5. Save raw file to `data/reference/{track_id}.{ext}`
6. Check for duplicates against existing index → return `duplicate_of` if average similarity exceeds threshold (~99%)

### Frontend follow-up

`bulk-ingest.tsx` calls `ingestTrack` per file, shows per-file status, then invalidates `["catalogue"]` so the table refreshes.

---

## 10. Development proxy

In dev, the browser talks to **port 5173**, not **8000**:

```
fetch("/query")  →  http://localhost:5173/query
```

Vite forwards matching paths to the backend (`vite.config.ts`):

```typescript
const apiRoutes = [
  "/health",
  "/query",
  "/catalogue",
  "/ingest-track",
  "/reset-catalogue",
  "/auth",
]
```

Flow:

```
Browser  →  localhost:5173/query  →  (Vite proxy)  →  localhost:8000/query
```

**Why?** Avoids CORS (browser blocking cross-origin requests). In production, set `VITE_API_BASE` to your real API origin if frontend and backend are hosted separately.

Override backend target with `VITE_BACKEND_URL` when starting Vite.

---

## 11. Authentication

Optional API key auth:

| Env var | Effect |
|---------|--------|
| `WAVEID_API_KEY` | Secret the server accepts |
| `WAVEID_REQUIRE_API_KEY=true` | Also require key for `POST /ingest-track` |

Frontend **Settings** saves the key to `localStorage`; `api.ts` sends it as `X-API-Key`.

| Route | Default access |
|-------|----------------|
| Query, list catalogue, health, stream audio | Open |
| Delete, reset, metadata patch | Requires valid key when server has one configured |
| Ingest | Requires key only when `WAVEID_REQUIRE_API_KEY=true` |

Verify a key from Settings or via:

```
POST /auth/verify
{ "api_key": "your-key" }
```

---

## 12. What does *not* use HTTP

The **`scripts/`** folder (evaluation, training, batch ingest) imports Python services directly:

```python
from waveid_backend.services.catalogue import add_track
```

Same data on disk (`data/index/`), but **no HTTP**. Internal tooling skips the API layer — that is normal and faster for batch jobs.

Examples: `scripts/ingest_dataset.py`, `scripts/run_evaluation.py`, `scripts/query_smoke_test.py`.

---

## 13. Debugging checklist

| Question | Where to look |
|----------|----------------|
| What endpoints exist? | `waveid_backend/main.py` — search `@app.` |
| How does the UI call them? | `waveid_frontend/src/lib/api.ts` |
| Who triggers the call? | `waveid_frontend/src/pages/` and `src/components/` |
| What JSON comes back? | Types in `api.ts`; Pydantic models in `main.py` |
| Where is data stored? | `data/index/`, `data/reference/`, `data/queries/` |
| Why did a request fail? | Backend terminal log; response `detail` field |
| Interactive testing? | http://localhost:8000/docs |

---

## 14. Hands-on exercises (curl)

With the backend running on port 8000:

**Health check**

```powershell
curl http://localhost:8000/health
```

**List catalogue**

```powershell
curl http://localhost:8000/catalogue
```

**Ingest a file**

```powershell
curl -X POST http://localhost:8000/ingest-track -F "file=@C:\path\to\song.wav"
```

**Query a clip**

```powershell
curl -X POST http://localhost:8000/query -F "file=@C:\path\to\clip.wav"
```

**Delete a track (when API key is configured)**

```powershell
curl -X DELETE http://localhost:8000/catalogue/TRACK_ID_HERE -H "X-API-Key: your-key"
```

This is exactly what `api.ts` does — the browser uses `fetch` instead of curl.

---

## 15. Key files

| File | Role |
|------|------|
| `waveid_backend/main.py` | All HTTP routes and request/response models |
| `waveid_backend/services/` | Audio, embeddings, search, catalogue logic |
| `waveid_backend/security/` | Auth, upload validation |
| `waveid_frontend/src/lib/api.ts` | Typed frontend API client |
| `waveid_frontend/vite.config.ts` | Dev proxy to backend |
| `waveid_frontend/src/pages/` | Page-level API usage via TanStack Query |
| `waveid_platform/scripts/` | Direct Python access (no HTTP) |

---

## Summary

**WaveID’s API is a FastAPI REST server that accepts audio uploads and catalogue commands over HTTP. The React app is a client that calls those endpoints through `api.ts`. Python scripts talk to the same on-disk data without HTTP.**

For a deeper dive, pick one endpoint (e.g. `/query`) and trace it from the button in `query.tsx` through `api.ts` into `main.py` and down into `services/search.py`.
