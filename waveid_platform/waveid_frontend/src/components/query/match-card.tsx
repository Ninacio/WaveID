import { useState } from "react"
import { ChevronDown, FileAudio } from "lucide-react"
import { Link } from "react-router-dom"
import { motion, AnimatePresence } from "framer-motion"

import { cn } from "@/lib/utils"
import { getTrackAudioUrl, type QueryMatch } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { WaveformPlayer } from "@/components/common/waveform-player"
import { AlignmentVisual } from "@/components/query/alignment-visual"
import { buildCasualExplanation } from "@/components/query/match-explainer"

function MeterRow({ label, value, delay = 0 }: { label: string; value: number; delay?: number }) {
  const pct = Math.max(0, Math.min(100, value * 100))
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium tabular-nums">{pct.toFixed(1)}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <motion.div
          className="h-full rounded-full bg-gradient-to-r from-primary to-accent"
          initial={{ width: "0%" }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.65, ease: "easeOut", delay }}
        />
      </div>
    </div>
  )
}

const STRENGTH_BADGE: Record<
  QueryMatch["match_strength"],
  { variant: "success" | "warning" | "secondary"; label: string }
> = {
  strong: { variant: "success", label: "Likely same song" },
  moderate: { variant: "warning", label: "Listen & compare" },
  weak: { variant: "secondary", label: "Probably different" },
}

export function MatchCard({
  match,
  rank,
  queryDuration,
  queryFile,
  similarityGap,
}: {
  match: QueryMatch
  rank: number
  queryDuration: number
  queryFile?: File | null
  similarityGap: number
}) {
  const [open, setOpen] = useState(rank === 0)
  const confidencePct = (match.score * 100).toFixed(1)
  const strength = STRENGTH_BADGE[match.match_strength] ?? STRENGTH_BADGE.moderate
  const explanation = buildCasualExplanation(match, rank, similarityGap)

  return (
    <motion.div
      className={cn(
        "rounded-xl border bg-card p-4",
        rank === 0
          ? "border-primary/50 shadow-[0_0_24px_-12px_var(--color-primary)]"
          : "border-border"
      )}
      whileHover={{
        y: -2,
        boxShadow:
          rank === 0
            ? "0 0 36px -6px var(--color-primary)"
            : "0 6px 24px -6px rgba(0,0,0,0.18)",
      }}
      transition={{ type: "spring", stiffness: 400, damping: 30 }}
    >
      <div className="flex items-start gap-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-primary/15 text-sm font-semibold text-primary">
          {rank + 1}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <FileAudio className="h-4 w-4 shrink-0 text-muted-foreground" />
            <span className="truncate font-medium">{match.filename}</span>
            <Badge variant={strength.variant}>{strength.label}</Badge>
            {rank === 0 && <Badge>Top match</Badge>}
          </div>
          <div className="mt-0.5 text-xs text-muted-foreground">
            {match.hits} matched moment{match.hits === 1 ? "" : "s"} in your clip
          </div>
        </div>
      </div>

      <motion.div
        className="mt-3 flex items-baseline gap-2"
        initial={{ opacity: 0, scale: 0.92 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.28, ease: "easeOut", delay: 0.06 }}
      >
        <span className="text-2xl font-semibold tabular-nums">
          {confidencePct}%
        </span>
        <span className="text-xs text-muted-foreground">
          relative ranking (not "same song" certainty)
        </span>
      </motion.div>

      <div className="mt-3 grid gap-2.5">
        <MeterRow label="Fingerprint similarity" value={match.similarity} delay={0.18} />
        <MeterRow label="Clip coverage" value={match.coverage} delay={0.28} />
      </div>

      <motion.div
        className="mt-4 space-y-3"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: "easeOut", delay: 0.38 }}
      >
        {queryFile && (
          <WaveformPlayer file={queryFile} label="Your clip" height={56} />
        )}
        <WaveformPlayer
          src={getTrackAudioUrl(match.track_id)}
          label={`Reference: ${match.filename}`}
          height={56}
        />
      </motion.div>

      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="mt-3 flex items-center gap-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
        aria-expanded={open}
      >
        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 transition-transform",
            open && "rotate-180"
          )}
        />
        Why this match?
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: [0.4, 0, 0.2, 1] }}
            className="overflow-hidden"
          >
            <div className="mt-2 space-y-3 rounded-lg border border-border bg-muted/30 p-3 text-sm">
              <div>
                <p className="font-medium text-foreground">{explanation.headline}</p>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                  {explanation.body}
                </p>
                {explanation.caution && (
                  <p className="mt-2 rounded-md border border-accent/30 bg-accent/10 px-2.5 py-2 text-xs text-foreground/90">
                    {explanation.caution}
                  </p>
                )}
              </div>

              {match.segments.length > 0 && (
                <AlignmentVisual
                  segments={match.segments}
                  queryDuration={queryDuration}
                  referenceFilename={match.filename}
                />
              )}

              <Link
                to={`/catalogue/${match.track_id}`}
                className="inline-block text-xs font-medium text-primary hover:underline"
              >
                View track details →
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
