import { useMemo, useState } from "react"
import { Link } from "react-router-dom"
import { motion } from "framer-motion"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Library,
  Loader2,
  Trash2,
  ChevronRight,
  Search,
  SearchX,
  Pencil,
} from "lucide-react"
import { toast } from "sonner"

import {
  getCatalogue,
  deleteTrack,
  resetCatalogue,
  type CatalogueTrack,
} from "@/lib/api"
import { PageHeader } from "@/components/common/page-header"
import { EmptyState } from "@/components/common/empty-state"
import { ConfirmDialog } from "@/components/common/confirm-dialog"
import { BulkIngest } from "@/components/catalogue/bulk-ingest"
import { EditMetadataDialog } from "@/components/catalogue/edit-metadata-dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
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

const MotionTableBody = motion(TableBody)
const MotionRow = motion(TableRow)

const tableBodyVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.05 } },
}

const rowVariants = {
  hidden: { opacity: 0, y: 8 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.22, ease: "easeOut" } },
}
import { Skeleton } from "@/components/ui/skeleton"

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}

export function CataloguePage() {
  const queryClient = useQueryClient()
  const [filter, setFilter] = useState("")

  const { data, isLoading } = useQuery({
    queryKey: ["catalogue"],
    queryFn: getCatalogue,
  })
  const tracks: CatalogueTrack[] = data ?? []

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase()
    if (!q) return tracks
    return tracks.filter((t) =>
      [t.filename, t.title, t.artist, ...t.tags]
        .filter(Boolean)
        .some((field) => field.toLowerCase().includes(q))
    )
  }, [tracks, filter])

  const remove = useMutation({
    mutationFn: (trackId: string) => deleteTrack(trackId),
    onSuccess: (res) => {
      toast.success(res.message)
      queryClient.invalidateQueries({ queryKey: ["catalogue"] })
    },
    onError: (err: Error) => toast.error(err.message),
  })

  const reset = useMutation({
    mutationFn: resetCatalogue,
    onSuccess: () => {
      toast.success("Catalogue cleared.")
      queryClient.invalidateQueries({ queryKey: ["catalogue"] })
    },
    onError: (err: Error) => toast.error(err.message),
  })

  return (
    <div className="space-y-6">
      <PageHeader
        title="Catalogue"
        description="Reference tracks indexed for identification."
        actions={
          tracks.length > 0 ? (
            <ConfirmDialog
              title="Clear the entire catalogue?"
              description="This permanently removes all reference tracks, embeddings, and stored audio. This cannot be undone."
              confirmLabel="Clear all"
              destructive
              onConfirm={() => reset.mutate()}
              trigger={
                <Button variant="outline" disabled={reset.isPending}>
                  {reset.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                  Clear all
                </Button>
              }
            />
          ) : undefined
        }
      />

      <BulkIngest />

      <Card>
        <CardHeader className="flex-row items-center justify-between gap-3">
          <CardTitle>
            Tracks {tracks.length > 0 && `(${tracks.length})`}
          </CardTitle>
          {tracks.length > 0 && (
            <div className="relative w-full max-w-xs">
              <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="Filter by name, artist, tag…"
                className="pl-8"
              />
            </div>
          )}
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : tracks.length === 0 ? (
            <EmptyState
              icon={Library}
              title="No tracks ingested yet"
              description="Add reference tracks above to start building your catalogue."
            />
          ) : filtered.length === 0 ? (
            <EmptyState
              icon={SearchX}
              title="No matches"
              description={`No tracks match "${filter}".`}
            />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Track</TableHead>
                  <TableHead>Artist</TableHead>
                  <TableHead className="text-right">Segments</TableHead>
                  <TableHead className="text-right">Duration</TableHead>
                  <TableHead className="w-24 text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <MotionTableBody
                key={filtered.map((t) => t.track_id).join(",")}
                variants={tableBodyVariants}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, amount: 0.05 }}
              >
                {filtered.map((t) => (
                  <MotionRow key={t.track_id} variants={rowVariants}>
                    <TableCell className="max-w-[260px]">
                      <Link
                        to={`/catalogue/${t.track_id}`}
                        className="block truncate font-medium hover:text-primary"
                      >
                        {t.title || t.filename}
                      </Link>
                      {t.title && (
                        <span className="block truncate text-xs text-muted-foreground">
                          {t.filename}
                        </span>
                      )}
                    </TableCell>
                    <TableCell className="max-w-[160px] truncate text-muted-foreground">
                      {t.artist || "-"}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {t.num_segments}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {formatDuration(t.duration)}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center justify-end gap-1">
                        <EditMetadataDialog
                          track={t}
                          trigger={
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8 text-muted-foreground hover:text-foreground"
                              aria-label={`Edit ${t.filename}`}
                            >
                              <Pencil className="h-4 w-4" />
                            </Button>
                          }
                        />
                        <ConfirmDialog
                          title="Delete this track?"
                          description={`"${t.filename}" and its embeddings will be removed from the index. This cannot be undone.`}
                          confirmLabel="Delete"
                          destructive
                          onConfirm={() => remove.mutate(t.track_id)}
                          trigger={
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8 text-muted-foreground hover:text-destructive"
                              aria-label={`Delete ${t.filename}`}
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          }
                        />
                        <Link
                          to={`/catalogue/${t.track_id}`}
                          className="text-muted-foreground hover:text-foreground"
                          aria-label="View track"
                        >
                          <ChevronRight className="h-4 w-4" />
                        </Link>
                      </div>
                    </TableCell>
                  </MotionRow>
                ))}
              </MotionTableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
