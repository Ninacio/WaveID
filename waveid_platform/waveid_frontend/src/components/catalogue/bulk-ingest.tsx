import { useCallback, useRef, useState } from "react"
import { useQueryClient } from "@tanstack/react-query"
import {
  UploadCloud,
  Plus,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  Trash2,
  AlertTriangle,
} from "lucide-react"
import { toast } from "sonner"

import { cn } from "@/lib/utils"
import { ingestTrack } from "@/lib/api"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

const ACCEPTED = [".wav", ".mp3", ".au"]
const MAX_BYTES = 50 * 1024 * 1024

type Status = "queued" | "uploading" | "done" | "error"

interface QueueItem {
  id: string
  file: File
  status: Status
  message?: string
  duplicate?: boolean
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function validate(file: File): string | null {
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase()
  if (!ACCEPTED.includes(ext)) return `Unsupported format (${ext || "none"})`
  if (file.size === 0) return "File is empty"
  if (file.size > MAX_BYTES) return "Exceeds 50 MB limit"
  return null
}

const StatusIcon = ({
  status,
  duplicate,
}: {
  status: Status
  duplicate?: boolean
}) => {
  switch (status) {
    case "uploading":
      return <Loader2 className="h-4 w-4 animate-spin text-primary" />
    case "done":
      return duplicate ? (
        <AlertTriangle className="h-4 w-4 text-accent" />
      ) : (
        <CheckCircle2 className="h-4 w-4 text-emerald-400" />
      )
    case "error":
      return <XCircle className="h-4 w-4 text-destructive" />
    default:
      return <Clock className="h-4 w-4 text-muted-foreground" />
  }
}

export function BulkIngest() {
  const queryClient = useQueryClient()
  const inputRef = useRef<HTMLInputElement>(null)
  const [items, setItems] = useState<QueueItem[]>([])
  const [dragActive, setDragActive] = useState(false)
  const [running, setRunning] = useState(false)

  const addFiles = useCallback((files: FileList | null) => {
    if (!files) return
    const next: QueueItem[] = Array.from(files).map((file) => {
      const error = validate(file)
      return {
        id: `${file.name}-${file.size}-${crypto.randomUUID()}`,
        file,
        status: error ? "error" : "queued",
        message: error ?? undefined,
      }
    })
    setItems((prev) => [...prev, ...next])
  }, [])

  const update = (id: string, patch: Partial<QueueItem>) =>
    setItems((prev) =>
      prev.map((it) => (it.id === id ? { ...it, ...patch } : it))
    )

  const runQueue = async () => {
    setRunning(true)
    let ok = 0
    let failed = 0
    // Snapshot queued items at run time.
    const queued = items.filter((it) => it.status === "queued")
    for (const item of queued) {
      update(item.id, { status: "uploading", message: undefined })
      try {
        const res = await ingestTrack(item.file)
        const dup = res.duplicate_of
        update(item.id, {
          status: "done",
          duplicate: Boolean(dup),
          message: dup
            ? `${res.num_segments} segments · possible duplicate of ${dup.filename} (${(dup.similarity * 100).toFixed(0)}%)`
            : `${res.num_segments} segments`,
        })
        ok += 1
        queryClient.invalidateQueries({ queryKey: ["catalogue"] })
      } catch (err) {
        update(item.id, {
          status: "error",
          message: (err as Error).message,
        })
        failed += 1
      }
    }
    setRunning(false)
    if (ok) toast.success(`Ingested ${ok} track${ok === 1 ? "" : "s"}.`)
    if (failed) toast.error(`${failed} file${failed === 1 ? "" : "s"} failed.`)
  }

  const queuedCount = items.filter((it) => it.status === "queued").length

  return (
    <Card>
      <CardHeader>
        <CardTitle>Add reference tracks</CardTitle>
        <CardDescription>
          Drop one or many files. Each is segmented, embedded, and indexed.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div
          role="button"
          tabIndex={0}
          onClick={() => !running && inputRef.current?.click()}
          onKeyDown={(e) => {
            if ((e.key === "Enter" || e.key === " ") && !running)
              inputRef.current?.click()
          }}
          onDragOver={(e) => {
            e.preventDefault()
            if (!running) setDragActive(true)
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragActive(false)
            if (!running) addFiles(e.dataTransfer.files)
          }}
          className={cn(
            "flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed border-border bg-muted/20 px-6 py-8 text-center transition-colors",
            dragActive && "border-primary bg-primary/5",
            running && "pointer-events-none opacity-60"
          )}
        >
          <input
            ref={inputRef}
            type="file"
            multiple
            accept={ACCEPTED.join(",")}
            className="hidden"
            onChange={(e) => addFiles(e.target.files)}
          />
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary">
            <UploadCloud className="h-6 w-6" />
          </div>
          <div>
            <p className="text-sm font-medium">
              <span className="text-primary">Drop files</span> or click to browse
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              WAV, MP3, or AU · max 50 MB each
            </p>
          </div>
        </div>

        {items.length > 0 && (
          <ul className="divide-y divide-border rounded-lg border border-border">
            {items.map((item) => (
              <li
                key={item.id}
                className="flex items-center gap-3 px-3 py-2.5 text-sm"
              >
                <StatusIcon status={item.status} duplicate={item.duplicate} />
                <div className="min-w-0 flex-1">
                  <div className="truncate font-medium">{item.file.name}</div>
                  <div className="text-xs text-muted-foreground">
                    {formatBytes(item.file.size)}
                    {item.message && (
                      <span
                        className={cn(
                          "ml-2",
                          item.status === "error" && "text-destructive",
                          item.duplicate && "text-accent"
                        )}
                      >
                        · {item.message}
                      </span>
                    )}
                  </div>
                </div>
                {!running && item.status !== "uploading" && (
                  <button
                    onClick={() =>
                      setItems((prev) =>
                        prev.filter((it) => it.id !== item.id)
                      )
                    }
                    className="text-muted-foreground hover:text-destructive"
                    aria-label="Remove from queue"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}

        <div className="flex flex-wrap gap-2">
          <Button onClick={runQueue} disabled={running || queuedCount === 0}>
            {running ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Ingesting…
              </>
            ) : (
              <>
                <Plus className="h-4 w-4" />
                Ingest {queuedCount > 0 ? `${queuedCount} ` : ""}track
                {queuedCount === 1 ? "" : "s"}
              </>
            )}
          </Button>
          {items.length > 0 && !running && (
            <Button variant="ghost" onClick={() => setItems([])}>
              Clear list
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
