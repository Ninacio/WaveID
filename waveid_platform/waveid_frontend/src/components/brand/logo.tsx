import { cn } from "@/lib/utils"

/**
 * WaveID mark - an oscilloscope trace framed by instrument crosshair brackets.
 * Stroke uses currentColor so it adapts to context (phosphor green on the
 * landing, foreground in the app shell).
 */
export function WaveMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      className={cn("h-[18px] w-[18px]", className)}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {/* Corner brackets - instrument reticle */}
      <path d="M2 6.5V3.5A1.5 1.5 0 0 1 3.5 2H6.5" opacity={0.85} />
      <path d="M17.5 2H20.5A1.5 1.5 0 0 1 22 3.5V6.5" opacity={0.85} />
      <path d="M22 17.5V20.5A1.5 1.5 0 0 1 20.5 22H17.5" opacity={0.85} />
      <path d="M6.5 22H3.5A1.5 1.5 0 0 1 2 20.5V17.5" opacity={0.85} />
      {/* Oscilloscope trace with a peak spike */}
      <path d="M4 12h2.2l1.4-5 2 10 2.2-8 1.6 4.2 1.2-1.2H20" />
    </svg>
  )
}

export function Logo({
  className,
  showTag = true,
}: {
  className?: string
  showTag?: boolean
}) {
  return (
    <div className={cn("flex items-center gap-2.5", className)}>
      <div className="flex h-8 w-8 shrink-0 items-center justify-center border border-current/25 bg-current/[0.06] text-primary">
        <WaveMark />
      </div>
      <div className="leading-tight">
        <div className="font-mono text-sm font-semibold tracking-tight text-foreground">
          Wave<span className="text-primary">ID</span>
        </div>
        {showTag && (
          <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
            Signal Lab
          </div>
        )}
      </div>
    </div>
  )
}
