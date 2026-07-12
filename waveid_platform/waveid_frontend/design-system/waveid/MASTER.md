# Design System Master File

> **LOGIC:** When building a specific page, first check `design-system/pages/[page-name].md`.
> If that file exists, its rules **override** this Master file.
> If not, strictly follow the rules below.

---

**Project:** WaveID
**Identity:** Signal Lab - forensic audio instrument
**Updated:** 2026-07-03

---

## Concept

WaveID proves that a transformed clip came from a reference track. The UI
reads like **lab equipment**: near-black canvas, phosphor-green readouts,
amber alert channel, monospace data type, crosshair/reticle details.
Precision, repeatability, evidence - never "vibey" gradients or decorative
glassmorphism.

## Global Rules

### Color Palette (dark = native mode)

| Role | Dark | Light ("lab daylight") | CSS Variable |
|------|------|------------------------|--------------|
| Background | `#07090B` | `#F5F8F7` | `--background` |
| Card / surface | `#0E131B` | `#FFFFFF` | `--card` |
| Primary (phosphor) | `#00E5A0` | `#047857` | `--primary` |
| Accent (amber alert) | `#FFB020` | `#B45309` | `--accent` |
| Destructive | `#FF4D6D` | `#DC2626` | `--destructive` |
| Muted text | `#7F95A9` | `#566761` | `--muted-foreground` |
| Border | `#1C2836` | `#D7E0DD` | `--border` |

Semantics: **primary/green = signal, confirmation, match**. **accent/amber =
caution, peak-hold, "listen & compare"**. **destructive/red = mismatch, error**.
Never use accent as a decorative tint - it is an alert channel.

### Typography

- **Display/body:** Space Grotesk (400–700)
- **Data/labels:** JetBrains Mono (400–600) - all numeric readouts, labels,
  timestamps, file names
- Labels use the `.label-mono` utility: mono + uppercase + `0.18em` tracking
- Numbers are always `tabular-nums`

### Instrument Utilities (in `src/index.css`)

| Utility | Effect |
|---------|--------|
| `.grid-bg` | Blueprint grid backdrop tinted by `--primary` |
| `.corner-ticks` | Crosshair corner brackets on cards (needs `position` set) |
| `.label-mono` | Monospace uppercase tracked label |
| `.glow-primary` | Phosphor text glow (dark mode only) |

The landing page additionally uses the scoped `.sl-root` system in
`src/styles/signal.css` (scanlines, vignette, scan-sweep cards, sl buttons).

### Motion

Primitives live in `src/components/motion/primitives.tsx`:

- `Reveal` / `RevealItem` - fade-rise on scroll, staggered lists
- `CountUp` - numeric readouts tick up when scrolled into view

Rules: motion communicates *measurement* (sweeps, counts, streams), not
decoration. Everything honours `prefers-reduced-motion` with a settled
static state. Durations 150–700ms; easing `[0.16, 1, 0.3, 1]` for reveals.

### Radius

Small and technical: `--radius: 0.375rem`. Instrument surfaces are square-ish;
avoid pill shapes except status chips.

---

## Anti-Patterns (Do NOT Use)

- ❌ Purple/indigo gradients, glassmorphism, "AI startup" glow blobs
- ❌ Decorative emoji, emojis as icons (use Lucide)
- ❌ Amber/accent as decoration (it means *caution*)
- ❌ Layout-shifting hovers; instant state changes (use 150–300ms)
- ❌ Low contrast text (4.5:1 minimum); invisible focus states
- ❌ Motion without a reduced-motion fallback

## Pre-Delivery Checklist

- [ ] Numeric readouts are mono + tabular
- [ ] Labels use `.label-mono`
- [ ] Green = match/confirm, amber = caution, red = error (never mixed)
- [ ] `prefers-reduced-motion` respected
- [ ] Focus states visible; contrast ≥ 4.5:1
- [ ] Responsive: 375px, 768px, 1024px, 1440px
