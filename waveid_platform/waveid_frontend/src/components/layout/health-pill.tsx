import { useQuery } from "@tanstack/react-query"

import { cn } from "@/lib/utils"
import { getHealth } from "@/lib/api"

export function HealthPill() {
  const { data, isError, isLoading } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
    retry: false,
  })

  const ok = !isError && data?.status === "ok"
  const label = isLoading ? "Checking…" : ok ? "API online" : "API offline"

  return (
    <div className="flex items-center gap-2 rounded-lg border border-border bg-muted/30 px-3 py-2 text-xs">
      <span
        className={cn(
          "h-2 w-2 rounded-full",
          isLoading
            ? "bg-muted-foreground animate-pulse"
            : ok
              ? "bg-emerald-400 shadow-[0_0_8px] shadow-emerald-400/60"
              : "bg-destructive"
        )}
      />
      <span className="text-muted-foreground">{label}</span>
    </div>
  )
}
