import { useEffect, useRef, useState, type ReactNode } from "react"
import {
  motion,
  useInView,
  useReducedMotion,
  type Variants,
} from "framer-motion"

const revealVariants: Variants = {
  hidden: { opacity: 0, y: 24 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1] },
  },
}

/**
 * Fade + rise on scroll into view. When staggering children, pass
 * `stagger` on the parent Reveal and use `RevealItem` for each child.
 */
export function Reveal({
  children,
  className,
  delay = 0,
  stagger,
  as = "div",
}: {
  children: ReactNode
  className?: string
  delay?: number
  stagger?: number
  as?: "div" | "section" | "ul" | "li"
}) {
  const reduced = useReducedMotion()
  const MotionTag = motion[as]

  if (reduced) {
    const Tag = as
    return <Tag className={className}>{children}</Tag>
  }

  return (
    <MotionTag
      className={className}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, margin: "-60px" }}
      variants={
        stagger
          ? {
              hidden: {},
              show: {
                transition: { staggerChildren: stagger, delayChildren: delay },
              },
            }
          : {
              hidden: revealVariants.hidden,
              show: {
                ...revealVariants.show,
                transition: {
                  ...(revealVariants.show as { transition: object }).transition,
                  delay,
                },
              },
            }
      }
    >
      {children}
    </MotionTag>
  )
}

/** Child of a staggered Reveal. */
export function RevealItem({
  children,
  className,
}: {
  children: ReactNode
  className?: string
}) {
  const reduced = useReducedMotion()
  if (reduced) return <div className={className}>{children}</div>
  return (
    <motion.div className={className} variants={revealVariants}>
      {children}
    </motion.div>
  )
}

/** Animated number that counts up when scrolled into view. */
export function CountUp({
  to,
  suffix = "",
  prefix = "",
  decimals = 0,
  duration = 1200,
  className,
}: {
  to: number
  suffix?: string
  prefix?: string
  decimals?: number
  duration?: number
  className?: string
}) {
  const ref = useRef<HTMLSpanElement | null>(null)
  const inView = useInView(ref, { once: true, margin: "-40px" })
  const reduced = useReducedMotion()
  const [value, setValue] = useState(reduced ? to : 0)

  useEffect(() => {
    if (reduced || !inView) return
    let raf = 0
    const t0 = performance.now()
    const tick = (now: number) => {
      const p = Math.min(1, (now - t0) / duration)
      const eased = 1 - Math.pow(1 - p, 3)
      setValue(to * eased)
      if (p < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [inView, reduced, to, duration])

  return (
    <span ref={ref} className={className}>
      {prefix}
      {value.toFixed(decimals)}
      {suffix}
    </span>
  )
}
