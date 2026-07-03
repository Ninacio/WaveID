import type { SegmentMatch } from "@/lib/api"

function fmt(t: number): string {
  const m = Math.floor(t / 60)
  const s = Math.floor(t % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}

function TimelineRow({
  label,
  duration,
  segments,
  mode,
}: {
  label: string
  duration: number
  segments: SegmentMatch[]
  mode: "query" | "reference"
}) {
  if (duration <= 0) return null

  return (
    <div className="space-y-1">
      <div className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      <div className="relative h-6 w-full overflow-hidden rounded-md bg-muted/80">
        {segments.map((seg, i) => {
          const start = mode === "query" ? seg.query_start : seg.ref_start
          const end = mode === "query" ? seg.query_end : seg.ref_end
          const left = (start / duration) * 100
          const width = Math.max(((end - start) / duration) * 100, 1.5)
          return (
            <div
              key={i}
              className="absolute top-0.5 bottom-0.5 rounded-sm bg-gradient-to-r from-primary to-accent"
              style={{
                left: `${left}%`,
                width: `${width}%`,
                opacity: 0.45 + Math.min(Math.max(seg.score, 0), 1) * 0.55,
              }}
              title={`${fmt(start)}–${fmt(end)} (${(seg.score * 100).toFixed(0)}% match)`}
            />
          )
        })}
      </div>
    </div>
  )
}

export function AlignmentVisual({
  segments,
  queryDuration,
  referenceFilename,
}: {
  segments: SegmentMatch[]
  queryDuration: number
  referenceFilename: string
}) {
  if (!segments.length || queryDuration <= 0) return null

  const refDuration = Math.max(
    ...segments.map((s) => s.ref_end),
    queryDuration
  )

  return (
    <div className="space-y-3 rounded-lg border border-border/60 bg-background/40 p-3">
      <p className="text-xs text-muted-foreground">
        The highlighted bars show where your clip and{" "}
        <span className="font-medium text-foreground">{referenceFilename}</span>{" "}
        have similar-sounding moments. Brighter bars = stronger overlap.
      </p>
      <TimelineRow
        label="Your clip"
        duration={queryDuration}
        segments={segments}
        mode="query"
      />
      <TimelineRow
        label="Reference track"
        duration={refDuration}
        segments={segments}
        mode="reference"
      />
      <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
        <span className="inline-flex items-center gap-1">
          <span className="h-2 w-6 rounded-sm bg-primary/60" />
          Matched section
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="h-2 w-6 rounded-sm bg-muted" />
          No match
        </span>
      </div>
    </div>
  )
}
