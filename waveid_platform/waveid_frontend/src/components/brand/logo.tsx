import { cn } from "@/lib/utils"

export function WaveMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      className={cn("h-[18px] w-[18px]", className)}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      aria-hidden="true"
    >
      <path d="M3 12h2.5M18.5 12H21" />
      <path d="M5.5 12c1.2 0 1.2-3 2.4-3s1.2 6 2.4 6 1.2-9 2.4-9 1.2 12 2.4 12 1.2-6 2.4-6" />
      <path d="M3 16h2M19 8h2" />
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
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-accent text-white shadow-[0_0_20px_-4px_var(--color-primary)]">
        <WaveMark />
      </div>
      <div className="leading-tight">
        <div className="text-sm font-semibold tracking-tight text-foreground">
          WaveID
        </div>
        {showTag && (
          <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            Audio Identification
          </div>
        )}
      </div>
    </div>
  )
}
