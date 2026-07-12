import { useEffect, useRef, useState } from "react"
import { motion, useInView } from "framer-motion"

type Row = {
  label: string
  value: string
  tone?: "green" | "amber" | "default"
}

const ROWS: Row[] = [
  { label: "REF_TRACK", value: "james_savage__on_time.mp3" },
  { label: "SEGMENTS", value: "157 / 2.0s window" },
  { label: "SIMILARITY", value: "0.9871", tone: "green" },
  { label: "COVERAGE", value: "1.000", tone: "green" },
  { label: "ALIGNMENT", value: "00:45.0 -> 01:06.2" },
  { label: "VERDICT", value: "STRONG MATCH", tone: "green" },
]

function toneClass(tone: Row["tone"]) {
  if (tone === "green") return "sl-glow font-semibold text-[var(--sl-green)]"
  if (tone === "amber") return "text-[var(--sl-amber)]"
  return "text-[var(--sl-text)]"
}

/**
 * Terminal-style live match report. On first view, rows stream in one by one
 * (staggered), and the similarity value ticks up to its final reading before
 * the verdict locks. Reduced-motion users get the fully-settled report.
 */
export function MatchReport({ className }: { className?: string }) {
  const ref = useRef<HTMLDivElement | null>(null)
  const inView = useInView(ref, { once: true, margin: "-40px" })
  const reduced =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches

  const [visibleRows, setVisibleRows] = useState(reduced ? ROWS.length : 0)
  const [sim, setSim] = useState(reduced ? 0.9871 : 0)

  useEffect(() => {
    if (reduced || !inView) return
    let cancelled = false

    // Stream rows in
    const timers: number[] = []
    ROWS.forEach((_, i) => {
      timers.push(
        window.setTimeout(() => {
          if (!cancelled) setVisibleRows(i + 1)
        }, 260 * i)
      )
    })

    // Count the similarity reading up once its row is due
    const simStart = 260 * 2
    timers.push(
      window.setTimeout(() => {
        const target = 0.9871
        const dur = 900
        const t0 = performance.now()
        const tick = (now: number) => {
          const p = Math.min(1, (now - t0) / dur)
          const eased = 1 - Math.pow(1 - p, 3)
          if (!cancelled) setSim(target * eased)
          if (p < 1) requestAnimationFrame(tick)
        }
        requestAnimationFrame(tick)
      }, simStart)
    )

    return () => {
      cancelled = true
      timers.forEach(clearTimeout)
    }
  }, [inView, reduced])

  return (
    <div
      ref={ref}
      className={`sl-card sl-corners rounded-lg text-left ${className ?? ""}`}
    >
      <div className="flex items-center justify-between border-b border-[var(--sl-border)] px-4 py-2.5">
        <span className="sl-mono text-[11px] uppercase tracking-[0.22em] text-[var(--sl-text-dim)]">
          match_report.log
        </span>
        <span className="sl-mono flex items-center gap-2 text-[11px] text-[var(--sl-green)]">
          <span className="sl-blink inline-block h-1.5 w-1.5 rounded-full bg-[var(--sl-green)]" />
          LIVE
        </span>
      </div>
      <div className="sl-mono space-y-2 px-5 py-4 text-[13px]">
        {ROWS.map((row, i) => {
          const shown = i < visibleRows
          const value =
            row.label === "SIMILARITY" ? sim.toFixed(4) : row.value
          return (
            <motion.div
              key={row.label}
              initial={false}
              animate={{
                opacity: shown ? 1 : 0,
                x: shown ? 0 : -6,
              }}
              transition={{ duration: 0.28, ease: "easeOut" }}
              className="flex items-baseline justify-between gap-4"
            >
              <span className="shrink-0 text-[var(--sl-text-dim)]">
                <span className="text-[var(--sl-green-dim)]">&gt;</span>{" "}
                {row.label}
              </span>
              <span className={`truncate text-right ${toneClass(row.tone)}`}>
                {value}
              </span>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}
