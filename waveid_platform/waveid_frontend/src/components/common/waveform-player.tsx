import { useEffect, useRef, useState } from "react"
import WaveSurfer from "wavesurfer.js"
import { Loader2, Pause, Play } from "lucide-react"

import { Button } from "@/components/ui/button"

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return "0:00"
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}

export function WaveformPlayer({
  file,
  src,
  height = 64,
  label,
}: {
  file?: File
  src?: string
  height?: number
  label?: string
}) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const wsRef = useRef<WaveSurfer | null>(null)
  const [playing, setPlaying] = useState(false)
  const [current, setCurrent] = useState(0)
  const [duration, setDuration] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!containerRef.current) return

    let objectUrl: string | null = null
    let audioUrl = src ?? ""
    if (file) {
      objectUrl = URL.createObjectURL(file)
      audioUrl = objectUrl
    }
    if (!audioUrl) return

    setLoading(true)
    setError(null)

    const ws = WaveSurfer.create({
      container: containerRef.current,
      height,
      waveColor: "#52525b",
      progressColor: "#8b45c8",
      cursorColor: "#d4a843",
      barWidth: 2,
      barGap: 2,
      barRadius: 2,
      url: audioUrl,
    })
    wsRef.current = ws

    ws.on("ready", () => {
      setDuration(ws.getDuration())
      setLoading(false)
    })
    ws.on("error", () => {
      setError("Could not load audio.")
      setLoading(false)
    })
    ws.on("audioprocess", (t) => setCurrent(t))
    ws.on("interaction", () => setCurrent(ws.getCurrentTime()))
    ws.on("play", () => setPlaying(true))
    ws.on("pause", () => setPlaying(false))
    ws.on("finish", () => setPlaying(false))

    return () => {
      ws.destroy()
      wsRef.current = null
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [file, src, height])

  return (
    <div className="space-y-1.5">
      {label && (
        <div className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </div>
      )}
      <div className="flex items-center gap-3 rounded-lg border border-border bg-muted/30 p-3">
        <Button
          type="button"
          variant="secondary"
          size="icon"
          className="h-9 w-9 shrink-0 rounded-full"
          onClick={() => wsRef.current?.playPause()}
          disabled={loading || Boolean(error)}
          aria-label={playing ? "Pause" : "Play"}
        >
          {loading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : playing ? (
            <Pause className="h-4 w-4" />
          ) : (
            <Play className="h-4 w-4" />
          )}
        </Button>
        <div className="min-w-0 flex-1">
          <div ref={containerRef} />
          {error && (
            <p className="mt-1 text-xs text-destructive">{error}</p>
          )}
        </div>
        <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
          {formatTime(current)} / {formatTime(duration)}
        </span>
      </div>
    </div>
  )
}
