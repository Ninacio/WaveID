import { Badge } from "@/components/ui/badge"

const MAP = {
  high: { variant: "success" as const, label: "High confidence" },
  medium: { variant: "warning" as const, label: "Medium confidence" },
  low: { variant: "secondary" as const, label: "Low confidence" },
}

export function ConfidenceBadge({
  label,
}: {
  label: "high" | "medium" | "low"
}) {
  const cfg = MAP[label] ?? MAP.low
  return <Badge variant={cfg.variant}>{cfg.label}</Badge>
}
