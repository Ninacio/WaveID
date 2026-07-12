import { Link, useNavigate, useParams } from "react-router-dom"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  ArrowLeft,
  Layers,
  Clock,
  Hash,
  Trash2,
  Loader2,
  Pencil,
} from "lucide-react"
import { toast } from "sonner"

import { deleteTrack, getTrack } from "@/lib/api"
import { PageHeader } from "@/components/common/page-header"
import { ConfirmDialog } from "@/components/common/confirm-dialog"
import { EditMetadataDialog } from "@/components/catalogue/edit-metadata-dialog"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Skeleton } from "@/components/ui/skeleton"
import { EmptyState } from "@/components/common/empty-state"

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}

export function TrackDetailPage() {
  const { trackId = "" } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["track", trackId],
    queryFn: () => getTrack(trackId),
    retry: false,
  })

  const remove = useMutation({
    mutationFn: () => deleteTrack(trackId),
    onSuccess: (res) => {
      toast.success(res.message)
      queryClient.invalidateQueries({ queryKey: ["catalogue"] })
      navigate("/catalogue")
    },
    onError: (err: Error) => toast.error(err.message),
  })

  return (
    <div className="space-y-6">
      <Button variant="ghost" size="sm" asChild className="-ml-2 w-fit">
        <Link to="/catalogue">
          <ArrowLeft className="h-4 w-4" />
          Back to catalogue
        </Link>
      </Button>

      {isLoading ? (
        <div className="space-y-4">
          <Skeleton className="h-9 w-72" />
          <div className="grid gap-4 sm:grid-cols-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-20 w-full" />
            ))}
          </div>
          <Skeleton className="h-64 w-full" />
        </div>
      ) : isError || !data ? (
        <EmptyState
          icon={Hash}
          title="Track not found"
          description={(error as Error)?.message ?? "This track does not exist."}
          action={
            <Button asChild>
              <Link to="/catalogue">Back to catalogue</Link>
            </Button>
          }
        />
      ) : (
        <>
          <PageHeader
            title={data.title || data.filename}
            description={data.title ? data.filename : `Track ID: ${data.track_id}`}
            actions={
              <>
                <EditMetadataDialog
                  track={data}
                  trigger={
                    <Button variant="outline">
                      <Pencil className="h-4 w-4" />
                      Edit metadata
                    </Button>
                  }
                />
                <ConfirmDialog
                  title="Delete this track?"
                  description={`"${data.filename}" and its embeddings will be removed from the index. This cannot be undone.`}
                  confirmLabel="Delete"
                  destructive
                  onConfirm={() => remove.mutate()}
                  trigger={
                    <Button variant="outline" disabled={remove.isPending}>
                      {remove.isPending ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Trash2 className="h-4 w-4" />
                      )}
                      Delete
                    </Button>
                  }
                />
              </>
            }
          />

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Metadata</CardTitle>
            </CardHeader>
            <CardContent>
              <dl className="grid gap-4 sm:grid-cols-3">
                <div>
                  <dt className="text-xs uppercase tracking-wide text-muted-foreground">
                    Artist
                  </dt>
                  <dd className="mt-0.5 text-sm font-medium">
                    {data.artist || (
                      <span className="text-muted-foreground">-</span>
                    )}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-muted-foreground">
                    Title
                  </dt>
                  <dd className="mt-0.5 text-sm font-medium">
                    {data.title || (
                      <span className="text-muted-foreground">-</span>
                    )}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs uppercase tracking-wide text-muted-foreground">
                    ISRC
                  </dt>
                  <dd className="mt-0.5 font-mono text-sm">
                    {data.isrc || (
                      <span className="font-sans text-muted-foreground">-</span>
                    )}
                  </dd>
                </div>
                <div className="sm:col-span-3">
                  <dt className="text-xs uppercase tracking-wide text-muted-foreground">
                    Tags
                  </dt>
                  <dd className="mt-1.5 flex flex-wrap gap-1.5">
                    {data.tags.length > 0 ? (
                      data.tags.map((tag) => (
                        <Badge key={tag} variant="secondary">
                          {tag}
                        </Badge>
                      ))
                    ) : (
                      <span className="text-sm text-muted-foreground">-</span>
                    )}
                  </dd>
                </div>
              </dl>
            </CardContent>
          </Card>

          <div className="grid gap-4 sm:grid-cols-3">
            <Card>
              <CardContent className="flex items-center gap-3">
                <Layers className="h-5 w-5 text-primary" />
                <div>
                  <div className="text-xs uppercase tracking-wide text-muted-foreground">
                    Segments
                  </div>
                  <div className="text-xl font-semibold tabular-nums">
                    {data.num_segments}
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="flex items-center gap-3">
                <Clock className="h-5 w-5 text-primary" />
                <div>
                  <div className="text-xs uppercase tracking-wide text-muted-foreground">
                    Duration
                  </div>
                  <div className="text-xl font-semibold tabular-nums">
                    {formatDuration(data.duration)}
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="flex items-center gap-3">
                <Hash className="h-5 w-5 text-primary" />
                <div>
                  <div className="text-xs uppercase tracking-wide text-muted-foreground">
                    Avg segment
                  </div>
                  <div className="text-xl font-semibold tabular-nums">
                    {data.num_segments
                      ? (data.duration / data.num_segments).toFixed(1)
                      : "0"}
                    s
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Segments</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>#</TableHead>
                    <TableHead className="text-right">Start</TableHead>
                    <TableHead className="text-right">End</TableHead>
                    <TableHead>Embedding ID</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.segments.map((s, i) => (
                    <TableRow key={s.segment_id}>
                      <TableCell className="tabular-nums">{i + 1}</TableCell>
                      <TableCell className="text-right tabular-nums">
                        {s.start_time.toFixed(2)}s
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {s.end_time.toFixed(2)}s
                      </TableCell>
                      <TableCell className="font-mono text-xs text-muted-foreground">
                        {s.embedding_id}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
