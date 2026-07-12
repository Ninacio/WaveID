import { useEffect, useRef } from "react"

/**
 * Live oscilloscope trace — the brand centerpiece.
 * A phosphor waveform sweeps across a transparent canvas (grid shows behind),
 * layered with a moving scan reticle and a peak-hold cap. Honours
 * prefers-reduced-motion by rendering a single static frame.
 */
export function Oscilloscope({ className }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const reduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches

    let raf = 0
    let t = reduced ? 40 : 0
    let dpr = Math.min(window.devicePixelRatio || 1, 2)

    const resize = () => {
      const rect = canvas.getBoundingClientRect()
      dpr = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = Math.max(1, Math.floor(rect.width * dpr))
      canvas.height = Math.max(1, Math.floor(rect.height * dpr))
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    window.addEventListener("resize", resize)

    // A quasi-audio waveform: a few detuned sines + an envelope that drifts,
    // so the trace looks like a living signal rather than a clean sine.
    const sample = (x: number, phase: number) => {
      const env =
        0.55 +
        0.35 * Math.sin(x * 0.6 + phase * 0.4) +
        0.1 * Math.sin(x * 2.3 - phase * 0.7)
      return (
        env *
        (0.6 * Math.sin(x * 3.1 + phase) +
          0.28 * Math.sin(x * 6.7 - phase * 1.3) +
          0.16 * Math.sin(x * 11.3 + phase * 0.6))
      )
    }

    const draw = () => {
      const rect = canvas.getBoundingClientRect()
      const w = rect.width
      const h = rect.height
      const midY = h * 0.5
      const amp = h * 0.3
      ctx.clearRect(0, 0, w, h)

      const phase = t * 0.03
      const steps = Math.max(64, Math.floor(w / 3))

      // Faint pre-echo trace (previous phase) for a motion-blur feel
      ctx.beginPath()
      for (let i = 0; i <= steps; i++) {
        const px = (i / steps) * w
        const xv = (i / steps) * 10
        const y = midY + sample(xv, phase - 0.5) * amp * 0.7
        if (i === 0) ctx.moveTo(px, y)
        else ctx.lineTo(px, y)
      }
      ctx.strokeStyle = "rgba(94, 234, 212, 0.14)"
      ctx.lineWidth = 1
      ctx.stroke()

      // Main phosphor trace with glow
      ctx.beginPath()
      let peakX = 0
      let peakY = midY
      let peakMag = 0
      for (let i = 0; i <= steps; i++) {
        const px = (i / steps) * w
        const xv = (i / steps) * 10
        const s = sample(xv, phase)
        const y = midY + s * amp
        if (Math.abs(s) > peakMag) {
          peakMag = Math.abs(s)
          peakX = px
          peakY = y
        }
        if (i === 0) ctx.moveTo(px, y)
        else ctx.lineTo(px, y)
      }
      ctx.shadowColor = "rgba(0, 229, 160, 0.7)"
      ctx.shadowBlur = 16
      ctx.strokeStyle = "rgba(0, 229, 160, 0.9)"
      ctx.lineWidth = 2
      ctx.stroke()
      ctx.shadowBlur = 0

      // Peak-hold marker + crosshair
      ctx.fillStyle = "rgba(255, 176, 32, 0.95)"
      ctx.fillRect(peakX - 2, peakY - 2, 4, 4)
      ctx.strokeStyle = "rgba(255, 176, 32, 0.28)"
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(peakX, 0)
      ctx.lineTo(peakX, h)
      ctx.stroke()

      // Sweeping scan reticle (left→right)
      if (!reduced) {
        const scanX = ((t * 1.6) % (steps + 40)) / steps * w
        const grad = ctx.createLinearGradient(scanX - 40, 0, scanX, 0)
        grad.addColorStop(0, "rgba(0, 229, 160, 0)")
        grad.addColorStop(1, "rgba(0, 229, 160, 0.35)")
        ctx.fillStyle = grad
        ctx.fillRect(scanX - 40, 0, 40, h)
        ctx.fillStyle = "rgba(0, 229, 160, 0.9)"
        ctx.fillRect(scanX - 1, 0, 1.5, h)
      }

      // Center baseline
      ctx.strokeStyle = "rgba(127, 149, 169, 0.18)"
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(0, midY)
      ctx.lineTo(w, midY)
      ctx.stroke()

      if (!reduced) {
        t += 1
        raf = window.requestAnimationFrame(draw)
      }
    }

    draw()

    return () => {
      window.removeEventListener("resize", resize)
      cancelAnimationFrame(raf)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className={className}
      aria-hidden="true"
      role="presentation"
    />
  )
}
