import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { Radar, Loader2, SearchX, Copy, History, Trash2 } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { toast } from "sonner"

import { queryClip, type QueryResponse } from "@/lib/api"
import {
  addHistoryEntry,
  clearHistory,
  loadHistory,
  type QueryHistoryEntry,
} from "@/lib/query-history"
import { PageHeader } from "@/components/common/page-header"
import { FileDropzone } from "@/components/common/file-dropzone"
import { WaveformPlayer } from "@/components/common/waveform-player"
import { MatchCard } from "@/components/query/match-card"
import { ConfidenceBadge } from "@/components/common/confidence-badge"
import { EmptyState } from "@/components/common/empty-state"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"

function timeAgo(ts: number): string {
  const diff = Date.now() - ts
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return "just now"
  if (mins < 60) return `${mins}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  return new Date(ts).toLocaleDateString()
}

export function QueryPage() {
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [history, setHistory] = useState<QueryHistoryEntry[]>(loadHistory)

  const mutation = useMutation({
    mutationFn: (f: File) => queryClip(f),
    onSuccess: (data, f) => {
      setResult(data)
      setHistory(addHistoryEntry(f.name, data))
      if (data.matches.length === 0) {
        toast.info("No matching tracks found in the catalogue.")
      } else {
        toast.success(`Found ${data.matches.length} candidate match(es).`)
      }
    },
    onError: (err: Error) => toast.error(err.message),
  })

  const copyResult = async () => {
    if (!result) return
    try {
      await navigator.clipboard.writeText(JSON.stringify(result, null, 2))
      toast.success("Result copied to clipboard.")
    } catch {
      toast.error("Could not copy to clipboard.")
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Identify a clip"
        description="Upload a short audio clip to match it against your reference catalogue."
      />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <div className="space-y-6">
          <Card className="h-fit">
            <CardHeader>
              <CardTitle>Query clip</CardTitle>
              <CardDescription>
                WAV, MP3, or AU. Works even on pitch-shifted or time-stretched
                audio.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <FileDropzone
                file={file}
                onFile={setFile}
                disabled={mutation.isPending}
              />
              {file && <WaveformPlayer file={file} />}
              <Button
                className="w-full"
                disabled={!file || mutation.isPending}
                onClick={() => file && mutation.mutate(file)}
              >
                {mutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Identifying…
                  </>
                ) : (
                  <>
                    <Radar className="h-4 w-4" />
                    Identify
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {history.length > 0 && (
            <Card>
              <CardHeader className="flex-row items-center justify-between">
                <CardTitle className="flex items-center gap-2 text-base">
                  <History className="h-4 w-4" />
                  Recent queries
                </CardTitle>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    clearHistory()
                    setHistory([])
                  }}
                >
                  <Trash2 className="h-4 w-4" />
                  Clear
                </Button>
              </CardHeader>
              <CardContent>
                <ul className="divide-y divide-border">
                  {history.map((h) => (
                    <li
                      key={h.id}
                      className="flex items-center justify-between gap-3 py-2.5 text-sm"
                    >
                      <div className="min-w-0">
                        <div className="truncate font-medium">{h.filename}</div>
                        <div className="text-xs text-muted-foreground">
                          {h.topMatch
                            ? `→ ${h.topMatch.filename} (${(h.topMatch.score * 100).toFixed(0)}%)`
                            : "No match"}{" "}
                          · {timeAgo(h.timestamp)}
                        </div>
                      </div>
                      <ConfidenceBadge label={h.confidenceLabel} />
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}
        </div>

        <Card>
          <CardHeader className="flex-row items-center justify-between">
            <div>
              <CardTitle>Matches</CardTitle>
              <CardDescription>Ranked by confidence.</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              {result && result.matches.length > 0 && (
                <>
                  <ConfidenceBadge label={result.confidence_label} />
                  <Button variant="ghost" size="icon" onClick={copyResult}>
                    <Copy className="h-4 w-4" />
                    <span className="sr-only">Copy result JSON</span>
                  </Button>
                </>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {mutation.isPending ? (
              <div className="space-y-3">
                {Array.from({ length: 3 }).map((_, i) => (
                  <Skeleton key={i} className="h-28 w-full" />
                ))}
              </div>
            ) : !result ? (
              <EmptyState
                icon={Radar}
                title="No results yet"
                description="Upload a clip and click Identify to see ranked matches with similarity and coverage."
              />
            ) : result.matches.length === 0 ? (
              <EmptyState
                icon={SearchX}
                title="No matches found"
                description="The clip didn't match any reference track. Try ingesting the source track first."
              />
            ) : (
              <AnimatePresence mode="wait">
                <motion.div
                  key={result.matches.map((m) => m.track_id).join(",")}
                  className="space-y-3"
                  initial="hidden"
                  animate="visible"
                  variants={{ hidden: {}, visible: { transition: { staggerChildren: 0.08 } } }}
                >
                  {result.confidence_label === "low" && result.matches.length > 0 && (
                    <motion.div
                      variants={{
                        hidden: { opacity: 0, y: 8 },
                        visible: { opacity: 1, y: 0, transition: { duration: 0.25 } },
                      }}
                      className="rounded-lg border border-accent/30 bg-accent/10 px-3 py-2.5 text-xs text-foreground/90"
                    >
                      Multiple tracks scored similarly ({(result.similarity_gap * 100).toFixed(1)}%
                      gap). High fingerprint scores can reflect shared genre or production —
                      use the players below to verify whether it&apos;s truly the same song.
                    </motion.div>
                  )}
                  {result.matches.map((m, i) => (
                    <motion.div
                      key={m.track_id}
                      variants={{
                        hidden: { opacity: 0, y: 16 },
                        visible: { opacity: 1, y: 0, transition: { duration: 0.28, ease: "easeOut" } },
                      }}
                    >
                      <MatchCard
                        match={m}
                        rank={i}
                        queryDuration={result.query_duration}
                        queryFile={file}
                        similarityGap={result.similarity_gap}
                      />
                    </motion.div>
                  ))}
                </motion.div>
              </AnimatePresence>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
