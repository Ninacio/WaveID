import { useEffect, useState, type ReactNode } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Loader2 } from "lucide-react"
import { toast } from "sonner"

import { updateTrackMetadata, type CatalogueTrack } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"

export function EditMetadataDialog({
  track,
  trigger,
}: {
  track: CatalogueTrack
  trigger: ReactNode
}) {
  const queryClient = useQueryClient()
  const [open, setOpen] = useState(false)
  const [title, setTitle] = useState(track.title)
  const [artist, setArtist] = useState(track.artist)
  const [isrc, setIsrc] = useState(track.isrc)
  const [tags, setTags] = useState(track.tags.join(", "))

  // Reset fields whenever the dialog opens with the latest track values.
  useEffect(() => {
    if (open) {
      setTitle(track.title)
      setArtist(track.artist)
      setIsrc(track.isrc)
      setTags(track.tags.join(", "))
    }
  }, [open, track])

  const mutation = useMutation({
    mutationFn: () =>
      updateTrackMetadata(track.track_id, {
        title: title.trim(),
        artist: artist.trim(),
        isrc: isrc.trim(),
        tags: tags
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean),
      }),
    onSuccess: () => {
      toast.success("Metadata updated.")
      queryClient.invalidateQueries({ queryKey: ["catalogue"] })
      queryClient.invalidateQueries({ queryKey: ["track", track.track_id] })
      setOpen(false)
    },
    onError: (err: Error) => toast.error(err.message),
  })

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit metadata</DialogTitle>
          <DialogDescription className="truncate">
            {track.filename}
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          <div className="space-y-1.5">
            <Label htmlFor="meta-title">Title</Label>
            <Input
              id="meta-title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Track title"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="meta-artist">Artist</Label>
            <Input
              id="meta-artist"
              value={artist}
              onChange={(e) => setArtist(e.target.value)}
              placeholder="Artist name"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="meta-isrc">ISRC</Label>
            <Input
              id="meta-isrc"
              value={isrc}
              onChange={(e) => setIsrc(e.target.value)}
              placeholder="e.g. US-ABC-12-34567"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="meta-tags">Tags</Label>
            <Input
              id="meta-tags"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="Comma-separated, e.g. lofi, chill"
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button onClick={() => mutation.mutate()} disabled={mutation.isPending}>
            {mutation.isPending && (
              <Loader2 className="h-4 w-4 animate-spin" />
            )}
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
