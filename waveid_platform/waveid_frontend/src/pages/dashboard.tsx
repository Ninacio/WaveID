import { useMemo } from "react"
import { Link } from "react-router-dom"
import { useQuery } from "@tanstack/react-query"
import {
  Library,
  Radar,
  Layers,
  Clock,
  ArrowRight,
  type LucideProps,
} from "lucide-react"
import type { ComponentType } from "react"

import { getCatalogue, type CatalogueTrack } from "@/lib/api"
import { PageHeader } from "@/components/common/page-header"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { EmptyState } from "@/components/common/empty-state"

function StatCard({
  icon: Icon,
  label,
  value,
  loading,
}: {
  icon: ComponentType<LucideProps>
  label: string
  value: string | number
  loading?: boolean
}) {
  return (
    <Card>
      <CardContent className="flex items-center gap-4">
        <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-primary/15 text-primary">
          <Icon className="h-5 w-5" />
        </div>
        <div className="space-y-1">
          <div className="text-xs uppercase tracking-wide text-muted-foreground">
            {label}
          </div>
          {loading ? (
            <Skeleton className="h-7 w-16" />
          ) : (
            <div className="text-2xl font-semibold tabular-nums">{value}</div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}

export function DashboardPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["catalogue"],
    queryFn: getCatalogue,
  })

  const tracks: CatalogueTrack[] = data ?? []

  const stats = useMemo(() => {
    const totalSegments = tracks.reduce((sum, t) => sum + t.num_segments, 0)
    const totalDuration = tracks.reduce((sum, t) => sum + t.duration, 0)
    return {
      tracks: tracks.length,
      segments: totalSegments,
      duration: totalDuration,
    }
  }, [tracks])

  const recent = tracks.slice(-5).reverse()

  return (
    <div className="space-y-6">
      <PageHeader
        title="Dashboard"
        description="Overview of your reference catalogue and identification activity."
        actions={
          <Button asChild>
            <Link to="/query">
              <Radar className="h-4 w-4" />
              Identify a clip
            </Link>
          </Button>
        }
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          icon={Library}
          label="Tracks"
          value={stats.tracks}
          loading={isLoading}
        />
        <StatCard
          icon={Layers}
          label="Segments"
          value={stats.segments}
          loading={isLoading}
        />
        <StatCard
          icon={Clock}
          label="Indexed audio"
          value={formatDuration(stats.duration)}
          loading={isLoading}
        />
        <StatCard
          icon={Radar}
          label="Avg segments/track"
          value={stats.tracks ? Math.round(stats.segments / stats.tracks) : 0}
          loading={isLoading}
        />
      </div>

      <Card>
        <CardHeader className="flex-row items-center justify-between">
          <div>
            <CardTitle>Recently ingested</CardTitle>
            <CardDescription>The latest reference tracks added.</CardDescription>
          </div>
          <Button variant="ghost" size="sm" asChild>
            <Link to="/catalogue">
              View all
              <ArrowRight className="h-4 w-4" />
            </Link>
          </Button>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : recent.length === 0 ? (
            <EmptyState
              icon={Library}
              title="No tracks yet"
              description="Ingest reference tracks to build your catalogue, then identify clips against it."
              action={
                <Button asChild>
                  <Link to="/catalogue">Go to catalogue</Link>
                </Button>
              }
            />
          ) : (
            <ul className="divide-y divide-border">
              {recent.map((t) => (
                <li key={t.track_id}>
                  <Link
                    to={`/catalogue/${t.track_id}`}
                    className="flex items-center justify-between gap-3 py-3 transition-colors hover:text-primary"
                  >
                    <span className="truncate font-medium">{t.filename}</span>
                    <span className="shrink-0 text-xs text-muted-foreground">
                      {t.num_segments} segments · {formatDuration(t.duration)}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
