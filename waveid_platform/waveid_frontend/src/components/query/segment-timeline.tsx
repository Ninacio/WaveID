import type { SegmentMatch } from "@/lib/api"

function fmt(t: number): string {
  const m = Math.floor(t / 60)
  const s = Math.floor(t % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}

export function SegmentTimeline({
  segments,
  duration,
}: {
  segments: SegmentMatch[]
  duration: number
}) {
  if (!segments.length || duration <= 0) return null
  const total = Math.max(duration, segments[segments.length - 1]?.query_end ?? 0)

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
        <span>Query timeline</span>
        <span>{fmt(total)}</span>
      </div>
      <div className="relative h-7 w-full overflow-hidden rounded-md bg-muted">
        {segments.map((seg, i) => {
          const left = (seg.query_start / total) * 100
          const width = Math.max(
            ((seg.query_end - seg.query_start) / total) * 100,
            1.2
          )
          return (
            <div
              key={i}
              className="absolute top-0 h-full rounded-sm bg-gradient-to-r from-primary to-accent"
              style={{
                left: `${left}%`,
                width: `${width}%`,
                opacity: 0.4 + Math.min(Math.max(seg.score, 0), 1) * 0.6,
              }}
              title={`Query ${fmt(seg.query_start)}–${fmt(seg.query_end)} → reference ${fmt(seg.ref_start)}–${fmt(seg.ref_end)} (${(seg.score * 100).toFixed(0)}%)`}
            />
          )
        })}
      </div>
      <p className="text-[11px] text-muted-foreground">
        {segments.length} aligned segment{segments.length === 1 ? "" : "s"} ·
        brighter = stronger match
      </p>
    </div>
  )
}
