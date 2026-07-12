import { useRef } from "react"
import { Link } from "react-router-dom"
import { motion, useScroll, useTransform, useReducedMotion } from "framer-motion"
import {
  ArrowRight,
  ScanLine,
  Crosshair,
  Fingerprint,
  Timer,
  Waves,
  Upload,
  Layers,
  Radar,
} from "lucide-react"

import "@/styles/signal.css"
import { Logo } from "@/components/brand/logo"
import { Oscilloscope } from "@/components/landing/oscilloscope"
import { MatchReport } from "@/components/landing/match-report"
import {
  Reveal,
  RevealItem,
  CountUp,
} from "@/components/motion/primitives"

const PIPELINE = [
  {
    icon: Upload,
    step: "01",
    title: "Ingest",
    body: "Reference tracks decoded, resampled to 16 kHz mono.",
  },
  {
    icon: Layers,
    step: "02",
    title: "Segment",
    body: "Sliced into 2s windows at 1s hop for overlap.",
  },
  {
    icon: Fingerprint,
    step: "03",
    title: "Embed",
    body: "Each window becomes a 128-d contrastive fingerprint.",
  },
  {
    icon: Radar,
    step: "04",
    title: "Match",
    body: "Cosine search ranks candidates with a verdict.",
  },
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
    body: "Matches pin to exact timestamps in the reference track, so you can see precisely where a clip came from.",
    metric: "2s resolution",
  },
  {
    icon: Timer,
    title: "Sub-3s identification",
    body: "Vector similarity search across the whole catalogue returns ranked candidates with confidence labels in seconds.",
    metric: "<3s latency",
  },
] as const

const STATS = [
  { to: 8, suffix: "+", label: "transforms survived" },
  { to: 3, prefix: "<", suffix: "s", label: "query latency" },
  { to: 128, suffix: "-d", label: "fingerprint vector" },
  { to: 2, suffix: "s", label: "segment resolution" },
] as const

function HeroBackdrop() {
  const ref = useRef<HTMLDivElement | null>(null)
  const reduced = useReducedMotion()
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end start"],
  })
  const gridY = useTransform(scrollYProgress, [0, 1], [0, reduced ? 0 : 120])
  const scopeY = useTransform(scrollYProgress, [0, 1], [0, reduced ? 0 : 200])
  const fade = useTransform(scrollYProgress, [0, 0.8], [1, 0])

  return (
    <div ref={ref} className="pointer-events-none absolute inset-0 overflow-hidden">
      <motion.div
        style={{ y: gridY }}
        className="sl-grid-bg absolute inset-0 -top-24 h-[140%]"
      />
      <motion.div
        style={{ y: scopeY, opacity: fade }}
        className="absolute inset-x-0 top-[38%] h-[42%]"
      >
        <Oscilloscope className="h-full w-full" />
      </motion.div>
      <div className="sl-vignette absolute inset-0" />
    </div>
  )
}

export function LandingPage() {
  return (
    <div className="sl-root sl-scanlines relative min-h-screen">
      {/* Header */}
      <header className="sticky top-0 z-30 border-b border-[var(--sl-border)] bg-[var(--sl-bg)]/85 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-5 lg:px-8">
          <Logo />
          <nav className="flex items-center gap-2">
            <span className="sl-mono mr-2 hidden items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-[var(--sl-text-dim)] sm:flex">
              <span className="sl-blink inline-block h-1.5 w-1.5 rounded-full bg-[var(--sl-green)]" />
              sys.online
            </span>
            <a
              href="/docs"
              target="_blank"
              rel="noreferrer"
              className="sl-btn-ghost rounded-md px-4 py-2 text-sm"
            >
              API docs
            </a>
            <Link to="/dashboard" className="sl-btn-primary rounded-md px-4 py-2 text-sm">
              Open app
            </Link>
          </nav>
        </div>
      </header>

      {/* Hero */}
      <section className="relative overflow-hidden">
        <HeroBackdrop />
        <div className="relative z-10 mx-auto flex max-w-6xl flex-col items-center px-6 pb-28 pt-24 text-center md:pt-32">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="sl-chip mb-8 rounded-full px-4 py-2 text-[11px]"
          >
            <ScanLine className="h-3.5 w-3.5" aria-hidden="true" />
            Forensic audio identification
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.05 }}
            className="mb-6 max-w-4xl text-4xl font-bold leading-[1.05] tracking-tight md:text-6xl"
          >
            Prove the clip is{" "}
            <span className="sl-glow text-[var(--sl-green)]">the song</span>.
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.12 }}
            className="mb-10 max-w-2xl text-lg leading-relaxed text-[var(--sl-text-dim)] md:text-xl"
          >
            WaveID fingerprints reference tracks and pins short-form clips back
            to them - with timestamps, similarity scores, and confidence
            verdicts. Even when the audio has been pitch-shifted, stretched, or
            buried in a mix.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.18 }}
            className="mb-16 flex flex-col items-center gap-4 sm:flex-row"
          >
            <Link
              to="/query"
              className="sl-btn-primary group rounded-md px-8 py-3.5 text-base"
            >
              Identify a clip
              <ArrowRight
                className="h-4 w-4 transition-transform group-hover:translate-x-1"
                aria-hidden="true"
              />
            </Link>
            <Link
              to="/catalogue"
              className="sl-btn-ghost rounded-md px-8 py-3.5 text-base"
            >
              <Waves className="h-4 w-4" aria-hidden="true" />
              Browse catalogue
            </Link>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.28 }}
            className="w-full max-w-xl"
          >
            <MatchReport />
          </motion.div>
        </div>
      </section>

      {/* Pipeline */}
      <section className="relative border-t border-[var(--sl-border)] bg-[var(--sl-bg-2)]">
        <div className="mx-auto max-w-6xl px-6 py-20">
          <Reveal className="mb-12">
            <div className="sl-mono mb-2 text-[11px] uppercase tracking-[0.28em] text-[var(--sl-cyan)]">
              01 / signal path
            </div>
            <h2 className="text-2xl font-bold tracking-tight md:text-3xl">
              From upload to verdict
            </h2>
          </Reveal>

          <div className="relative">
            {/* Connecting line */}
            <motion.div
              initial={{ scaleX: 0 }}
              whileInView={{ scaleX: 1 }}
              viewport={{ once: true, margin: "-80px" }}
              transition={{ duration: 1, ease: "easeInOut" }}
              className="absolute left-0 right-0 top-[34px] hidden h-px origin-left bg-gradient-to-r from-[var(--sl-green)]/60 via-[var(--sl-cyan)]/30 to-transparent md:block"
            />
            <Reveal
              stagger={0.14}
              className="grid gap-6 md:grid-cols-4"
            >
              {PIPELINE.map((p) => (
                <RevealItem key={p.step}>
                  <div className="relative">
                    <div className="mb-5 flex h-[68px] w-[68px] items-center justify-center border border-[var(--sl-border)] bg-[var(--sl-surface)] text-[var(--sl-green)]">
                      <p.icon className="h-6 w-6" aria-hidden="true" />
                    </div>
                    <div className="sl-mono mb-1 text-[11px] tracking-[0.2em] text-[var(--sl-text-faint)]">
                      {p.step}
                    </div>
                    <h3 className="mb-1.5 text-lg font-semibold">{p.title}</h3>
                    <p className="text-sm leading-relaxed text-[var(--sl-text-dim)]">
                      {p.body}
                    </p>
                  </div>
                </RevealItem>
              ))}
            </Reveal>
          </div>
        </div>
      </section>

      {/* Capabilities */}
      <section className="relative border-t border-[var(--sl-border)]">
        <div className="mx-auto max-w-6xl px-6 py-20">
          <Reveal className="mb-12 flex items-end justify-between gap-4">
            <div>
              <div className="sl-mono mb-2 text-[11px] uppercase tracking-[0.28em] text-[var(--sl-cyan)]">
                02 / capabilities
              </div>
              <h2 className="text-2xl font-bold tracking-tight md:text-3xl">
                Built like lab equipment
              </h2>
            </div>
            <div className="sl-mono hidden text-[11px] uppercase tracking-[0.2em] text-[var(--sl-text-dim)] md:block">
              precision · repeatability · evidence
            </div>
          </Reveal>

          <Reveal stagger={0.12} className="grid gap-5 md:grid-cols-3">
            {FEATURES.map((f) => (
              <RevealItem key={f.title}>
                <div className="sl-card h-full rounded-lg p-6">
                  <f.icon
                    className="mb-5 h-6 w-6 text-[var(--sl-green)]"
                    aria-hidden="true"
                  />
                  <h3 className="mb-2 text-lg font-semibold">{f.title}</h3>
                  <p className="mb-6 text-sm leading-relaxed text-[var(--sl-text-dim)]">
                    {f.body}
                  </p>
                  <div className="sl-mono border-t border-[var(--sl-border)] pt-4 text-[12px] uppercase tracking-[0.18em] text-[var(--sl-cyan)]">
                    {f.metric}
                  </div>
                </div>
              </RevealItem>
            ))}
          </Reveal>
        </div>
      </section>

      {/* Stats band */}
      <section className="relative border-t border-[var(--sl-border)] bg-[var(--sl-bg-2)]">
        <Reveal
          stagger={0.1}
          className="mx-auto grid max-w-6xl grid-cols-2 gap-px overflow-hidden px-6 py-16 md:grid-cols-4"
        >
          {STATS.map((s) => (
            <RevealItem key={s.label} className="px-4 text-center">
              <div className="sl-mono text-4xl font-bold tracking-tight text-[var(--sl-green)] md:text-5xl">
                <CountUp
                  to={s.to}
                  prefix={"prefix" in s ? s.prefix : ""}
                  suffix={s.suffix}
                />
              </div>
              <div className="sl-mono mt-2 text-[11px] uppercase tracking-[0.18em] text-[var(--sl-text-dim)]">
                {s.label}
              </div>
            </RevealItem>
          ))}
        </Reveal>
      </section>

      {/* CTA */}
      <section className="relative border-t border-[var(--sl-border)]">
        <Reveal className="mx-auto flex max-w-6xl flex-col items-center gap-6 px-6 py-20 text-center">
          <div className="sl-mono text-[11px] uppercase tracking-[0.28em] text-[var(--sl-cyan)]">
            03 / run a test
          </div>
          <h2 className="max-w-2xl text-2xl font-bold tracking-tight md:text-4xl">
            Drop a clip in. Get a verdict with the receipts.
          </h2>
          <Link
            to="/query"
            className="sl-btn-primary group rounded-md px-8 py-3.5 text-base"
          >
            Start identifying
            <ArrowRight
              className="h-4 w-4 transition-transform group-hover:translate-x-1"
              aria-hidden="true"
            />
          </Link>
        </Reveal>
      </section>

      {/* Footer */}
      <footer className="border-t border-[var(--sl-border)]">
        <div className="sl-mono mx-auto flex max-w-6xl flex-col items-center justify-between gap-2 px-6 py-6 text-[11px] uppercase tracking-[0.18em] text-[var(--sl-text-dim)] sm:flex-row">
          <span>WaveID - Signal Lab</span>
          <span>Audio fingerprinting · contrastive embeddings</span>
        </div>
      </footer>
    </div>
  )
}
