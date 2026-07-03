import type { QueryResponse } from "@/lib/api"

const STORAGE_KEY = "waveid.queryHistory"
const MAX_ENTRIES = 20

export interface QueryHistoryEntry {
  id: string
  filename: string
  timestamp: number
  confidenceLabel: QueryResponse["confidence_label"]
  matchCount: number
  topMatch: { filename: string; score: number } | null
}

export function loadHistory(): QueryHistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? (parsed as QueryHistoryEntry[]) : []
  } catch {
    return []
  }
}

export function addHistoryEntry(
  filename: string,
  result: QueryResponse
): QueryHistoryEntry[] {
  const top = result.matches[0]
  const entry: QueryHistoryEntry = {
    id: crypto.randomUUID(),
    filename,
    timestamp: Date.now(),
    confidenceLabel: result.confidence_label,
    matchCount: result.matches.length,
    topMatch: top ? { filename: top.filename, score: top.score } : null,
  }
  const next = [entry, ...loadHistory()].slice(0, MAX_ENTRIES)
  localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
  return next
}

export function clearHistory(): void {
  localStorage.removeItem(STORAGE_KEY)
}
