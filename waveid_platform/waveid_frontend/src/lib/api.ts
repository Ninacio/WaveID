/**
 * Typed client for the WaveID FastAPI backend.
 *
 * In dev, requests are proxied to the backend by Vite (see vite.config.ts).
 * In production, set VITE_API_BASE to the backend origin if it differs.
 */

const API_BASE = import.meta.env.VITE_API_BASE ?? ""
const API_KEY_STORAGE = "waveid.apiKey"

export function getApiKey(): string {
  return localStorage.getItem(API_KEY_STORAGE) ?? ""
}

export function setApiKey(key: string): void {
  if (key) localStorage.setItem(API_KEY_STORAGE, key)
  else localStorage.removeItem(API_KEY_STORAGE)
}

function authHeaders(): Record<string, string> {
  const key = getApiKey()
  return key ? { "X-API-Key": key } : {}
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      // ignore non-JSON error bodies
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

// ── Types (mirror the backend Pydantic models) ──────────────────────────────

export interface CatalogueTrack {
  track_id: string
  filename: string
  num_segments: number
  duration: number
  title: string
  artist: string
  isrc: string
  tags: string[]
}

export interface SegmentInfo {
  segment_id: string
  start_time: number
  end_time: number
  embedding_id: string
}

export interface TrackDetail extends CatalogueTrack {
  segments: SegmentInfo[]
}

export interface TrackMetadataUpdate {
  title?: string
  artist?: string
  isrc?: string
  tags?: string[]
}

export interface SegmentMatch {
  query_start: number
  query_end: number
  ref_start: number
  ref_end: number
  score: number
}

export interface QueryMatch {
  track_id: string
  filename: string
  score: number
  similarity: number
  coverage: number
  hits: number
  segments: SegmentMatch[]
  match_strength: "strong" | "moderate" | "weak"
}

export interface QueryResponse {
  query_embedding: number[]
  matches: QueryMatch[]
  confidence_gap: number
  confidence_label: "high" | "medium" | "low"
  query_duration: number
  similarity_gap: number
}

export interface DuplicateInfo {
  track_id: string
  filename: string
  similarity: number
}

export interface IngestResponse {
  message: string
  track_id: string
  num_segments: number
  duration_seconds: number
  duplicate_of: DuplicateInfo | null
}

// ── Endpoints ───────────────────────────────────────────────────────────────

export async function getHealth(): Promise<{ status: string }> {
  return handle(await fetch(`${API_BASE}/health`))
}

export async function getCatalogue(): Promise<CatalogueTrack[]> {
  return handle(await fetch(`${API_BASE}/catalogue`))
}

export async function getTrack(trackId: string): Promise<TrackDetail> {
  return handle(await fetch(`${API_BASE}/catalogue/${trackId}`))
}

export function getTrackAudioUrl(trackId: string): string {
  return `${API_BASE}/catalogue/${trackId}/audio`
}

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

export async function ingestTrack(file: File): Promise<IngestResponse> {
  const form = new FormData()
  form.append("file", file)
  return handle(
    await fetch(`${API_BASE}/ingest-track`, {
      method: "POST",
      headers: authHeaders(),
      body: form,
    })
  )
}

export async function updateTrackMetadata(
  trackId: string,
  body: TrackMetadataUpdate
): Promise<TrackDetail> {
  return handle(
    await fetch(`${API_BASE}/catalogue/${trackId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify(body),
    })
  )
}

export async function deleteTrack(
  trackId: string
): Promise<{ message: string; track_id: string; removed_embeddings: number }> {
  return handle(
    await fetch(`${API_BASE}/catalogue/${trackId}`, {
      method: "DELETE",
      headers: authHeaders(),
    })
  )
}

export async function resetCatalogue(): Promise<{ message: string }> {
  return handle(
    await fetch(`${API_BASE}/reset-catalogue`, {
      method: "POST",
      headers: authHeaders(),
    })
  )
}

export async function verifyApiKey(
  apiKey: string
): Promise<{ authenticated: boolean; message: string }> {
  return handle(
    await fetch(`${API_BASE}/auth/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: apiKey }),
    })
  )
}
