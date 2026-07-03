import { useEffect, useRef } from "react"
import { Link } from "react-router-dom"
import {
  ArrowRight,
  Crosshair,
  Fingerprint,
  ScanLine,
  Timer,
  Waves,
} from "lucide-react"

import { WaveMark } from "@/components/brand/logo"
import "./signal-lab.css"

/**
 * Signal Lab — rebrand prototype.
 * Forensic / oscilloscope identity: near-black, phosphor green, monospace data.
 * Lives on its own route (/signal-lab) so it can be compared against
 * the current landing page without touching it.
 */

const READOUT_ROWS = [
  { label: "REF_TRACK", value: "james_savage__on_time.mp3" },
  { label: "SIMILARITY", value: "0.9871", highlight: true },
  { label: "COVERAGE", value: "1.000" },
  { label: "ALIGNMENT", value: "00:45.0 → 01:06.2" },
  { label: "VERDICT", value: "STRONG MATCH", highlight: true },
] as const

const FEATURES = [
  {
    icon: Fingerprint,
    title: "Contrastive fingerprints",
    body: "Every 2-second segment becomes an embedding trained to survive pitch-shift, time-stretch, and re-encoding.",
    metric: "8+ transforms",
  },
  {
    icon: Crosshair,
    title: "Segment alignment",
    body: "Matches are pinned to exact timestamps in the reference track, so you can see where the clip came from.",
    metric: "2s resolution",
  },
  {
    icon: Timer,
    title: "Sub-3s identification",
    body: "Vector similarity search over the whole catalogue returns ranked candidates with confidence labels in seconds.",
    metric: "<3s latency",
  },
] as const

function SpectrogramCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return undefined
    const ctx = canvas.getContext("2d")
    if (!ctx) return undefined

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches

    let raf = 0
    let t = 0
    const COL_W = 5
    const GAP = 2

    const resize = () => {
      const rect = canvas.getBoundingClientRect()
      canvas.width = rect.width
      canvas.height = rect.height
    }
    resize()
    window.addEventListener("resize", resize)

    // Pseudo-spectrogram: vertical bars whose heights drift like audio energy bands.
    const draw = () => {
      const { width, height } = canvas
      ctx.clearRect(0, 0, width, height)

      const cols = Math.ceil(width / (COL_W + GAP))
      for (let i = 0; i < cols; i++) {
        const x = i * (COL_W + GAP)
        const phase = i * 0.35
        const energy =
          0.28 +
          0.22 * Math.sin(t * 0.021 + phase) +
          0.16 * Math.sin(t * 0.043 + phase * 1.7) +
          0.1 * Math.sin(t * 0.011 + phase * 0.6)
        const h = Math.max(4, energy * height * 0.7)
        const y = height - h

        const grad = ctx.createLinearGradient(0, y, 0, height)
        grad.addColorStop(0, "rgba(0, 229, 160, 0.55)")
        grad.addColorStop(0.5, "rgba(94, 234, 212, 0.22)")
        grad.addColorStop(1, "rgba(0, 229, 160, 0.04)")
        ctx.fillStyle = grad
        ctx.fillRect(x, y, COL_W, h)

        // Peak cap
        ctx.fillStyle = "rgba(0, 229, 160, 0.8)"
        ctx.fillRect(x, y, COL_W, 2)
      }

      t += reduced ? 0 : 1
      raf = window.requestAnimationFrame(draw)
    }
    raf = window.requestAnimationFrame(draw)

    return () => {
      window.removeEventListener("resize", resize)
      cancelAnimationFrame(raf)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-x-0 bottom-0 h-[46%] w-full opacity-70"
      aria-hidden="true"
    />
  )
}

export function SignalLabPage() {
  return (
    <div className="signal-lab sl-grid-bg">
      {/* Header */}
      <header className="sticky top-0 z-30 border-b border-[var(--sl-border)] bg-[var(--sl-bg)]/85 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-5">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center border border-[var(--sl-green)]/60 bg-[var(--sl-surface)] text-[var(--sl-green)]">
              <WaveMark />
            </div>
            <div className="leading-tight">
              <div className="text-sm font-semibold tracking-tight">WaveID</div>
              <div className="sl-mono text-[10px] uppercase tracking-[0.22em] text-[var(--sl-text-dim)]">
                Signal Lab
              </div>
            </div>
          </div>
          <nav className="flex items-center gap-2">
            <span className="sl-mono mr-2 hidden items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-[var(--sl-text-dim)] sm:flex">
              <span className="sl-blink inline-block h-1.5 w-1.5 rounded-full bg-[var(--sl-green)]" />
              sys.online
            </span>
            <a
              href="/docs"
              target="_blank"
              rel="noreferrer"
              className="sl-btn-ghost cursor-pointer rounded-md px-4 py-2 text-sm"
            >
              API docs
            </a>
            <Link
              to="/dashboard"
              className="sl-btn-primary cursor-pointer rounded-md px-4 py-2 text-sm"
            >
              Open app
            </Link>
          </nav>
        </div>
      </header>

      {/* Hero */}
      <section className="relative overflow-hidden">
        <SpectrogramCanvas />
        <div className="relative z-10 mx-auto flex max-w-6xl flex-col items-center px-6 pb-64 pt-24 text-center md:pt-32">
          <div className="sl-mono mb-8 inline-flex items-center gap-2 border border-[var(--sl-border)] bg-[var(--sl-surface)]/80 px-4 py-2 text-[11px] uppercase tracking-[0.28em] text-[var(--sl-cyan)]">
            <ScanLine className="h-3.5 w-3.5" aria-hidden="true" />
            Forensic audio identification
          </div>

          <h1 className="mb-6 max-w-4xl text-4xl font-bold leading-tight tracking-tight md:text-6xl">
            Prove the clip is <span className="sl-glow text-[var(--sl-green)]">the song</span>.
          </h1>

          <p className="mb-10 max-w-2xl text-lg leading-relaxed text-[var(--sl-text-dim)] md:text-xl">
            WaveID fingerprints reference tracks and pins short-form clips back
            to them — with timestamps, similarity scores, and confidence
            verdicts. Even when the audio has been pitch-shifted, stretched, or
            buried in a mix.
          </p>

          <div className="mb-16 flex flex-col items-center gap-4 sm:flex-row">
            <Link
              to="/query"
              className="sl-btn-primary group inline-flex cursor-pointer items-center gap-2 rounded-md px-8 py-3.5 text-base"
            >
              Identify a clip
              <ArrowRight
                className="h-4 w-4 transition-transform group-hover:translate-x-1"
                aria-hidden="true"
              />
            </Link>
            <Link
              to="/catalogue"
              className="sl-btn-ghost inline-flex cursor-pointer items-center gap-2 rounded-md px-8 py-3.5 text-base"
            >
              <Waves className="h-4 w-4" aria-hidden="true" />
              Browse catalogue
            </Link>
          </div>

          {/* Terminal-style match readout */}
          <div className="sl-card sl-corners w-full max-w-xl rounded-lg text-left">
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
              {READOUT_ROWS.map((row) => (
                <div key={row.label} className="flex items-baseline justify-between gap-4">
                  <span className="shrink-0 text-[var(--sl-text-dim)]">{row.label}</span>
                  <span
                    className={
                      "truncate text-right " +
                      ("highlight" in row && row.highlight
                        ? "sl-glow font-semibold text-[var(--sl-green)]"
                        : "text-[var(--sl-text)]")
                    }
                  >
                    {row.value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="relative border-t border-[var(--sl-border)] bg-[var(--sl-bg)]">
        <div className="mx-auto max-w-6xl px-6 py-20">
          <div className="mb-12 flex items-end justify-between gap-4">
            <div>
              <div className="sl-mono mb-2 text-[11px] uppercase tracking-[0.28em] text-[var(--sl-cyan)]">
                01 / capabilities
              </div>
              <h2 className="text-2xl font-bold tracking-tight md:text-3xl">
                Built like lab equipment
              </h2>
            </div>
            <div className="sl-mono hidden text-[11px] uppercase tracking-[0.2em] text-[var(--sl-text-dim)] md:block">
              precision · repeatability · evidence
            </div>
          </div>

          <div className="grid gap-5 md:grid-cols-3">
            {FEATURES.map((feature) => (
              <div key={feature.title} className="sl-card rounded-lg p-6">
                <feature.icon
                  className="mb-5 h-6 w-6 text-[var(--sl-green)]"
                  aria-hidden="true"
                />
                <h3 className="mb-2 text-lg font-semibold">{feature.title}</h3>
                <p className="mb-6 text-sm leading-relaxed text-[var(--sl-text-dim)]">
                  {feature.body}
                </p>
                <div className="sl-mono border-t border-[var(--sl-border)] pt-4 text-[12px] uppercase tracking-[0.18em] text-[var(--sl-cyan)]">
                  {feature.metric}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA strip */}
      <section className="border-t border-[var(--sl-border)]">
        <div className="mx-auto flex max-w-6xl flex-col items-center gap-6 px-6 py-16 text-center">
          <div className="sl-mono text-[11px] uppercase tracking-[0.28em] text-[var(--sl-cyan)]">
            02 / run a test
          </div>
          <h2 className="max-w-2xl text-2xl font-bold tracking-tight md:text-3xl">
            Drop a clip in. Get a verdict with the receipts.
          </h2>
          <Link
            to="/query"
            className="sl-btn-primary inline-flex cursor-pointer items-center gap-2 rounded-md px-8 py-3.5 text-base"
          >
            Start identifying
            <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-[var(--sl-border)]">
        <div className="sl-mono mx-auto flex max-w-6xl flex-col items-center justify-between gap-2 px-6 py-6 text-[11px] uppercase tracking-[0.18em] text-[var(--sl-text-dim)] sm:flex-row">
          <span>WaveID — Signal Lab prototype</span>
          <span>
            <Link to="/" className="cursor-pointer underline-offset-4 hover:text-[var(--sl-green)] hover:underline">
              view current landing
            </Link>
          </span>
        </div>
      </footer>
    </div>
  )
}
