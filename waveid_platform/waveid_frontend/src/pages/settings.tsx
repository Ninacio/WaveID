import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { KeyRound, Loader2, Check, Eye, EyeOff } from "lucide-react"
import { toast } from "sonner"

import { getApiKey, setApiKey, verifyApiKey } from "@/lib/api"
import { PageHeader } from "@/components/common/page-header"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

export function SettingsPage() {
  const [key, setKey] = useState(getApiKey())
  const [reveal, setReveal] = useState(false)

  const verify = useMutation({
    mutationFn: (k: string) => verifyApiKey(k),
    onSuccess: (res) => {
      if (res.authenticated) toast.success("API key is valid.")
      else toast.error("API key is invalid.")
    },
    onError: (err: Error) => toast.error(err.message),
  })

  const save = () => {
    setApiKey(key.trim())
    toast.success("API key saved to this browser.")
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Settings"
        description="Configure how this client talks to the WaveID backend."
      />

      <Card className="max-w-2xl">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <KeyRound className="h-4 w-4" />
            API key
          </CardTitle>
          <CardDescription>
            Required for ingesting tracks and clearing the catalogue when the
            backend runs with authentication enabled. Stored locally in this
            browser only.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="api-key">Key</Label>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Input
                  id="api-key"
                  type={reveal ? "text" : "password"}
                  value={key}
                  onChange={(e) => setKey(e.target.value)}
                  placeholder="Enter your WaveID API key"
                  className="pr-10"
                />
                <button
                  type="button"
                  onClick={() => setReveal((r) => !r)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  aria-label={reveal ? "Hide key" : "Show key"}
                >
                  {reveal ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button onClick={save}>
              <Check className="h-4 w-4" />
              Save
            </Button>
            <Button
              variant="outline"
              onClick={() => verify.mutate(key.trim())}
              disabled={!key.trim() || verify.isPending}
            >
              {verify.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <KeyRound className="h-4 w-4" />
              )}
              Test key
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card className="max-w-2xl">
        <CardHeader>
          <CardTitle>Upload limits</CardTitle>
          <CardDescription>
            Enforced by the backend on every upload.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <dt className="text-muted-foreground">Max file size</dt>
              <dd className="font-medium">50 MB</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Max duration</dt>
              <dd className="font-medium">10 minutes</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Accepted formats</dt>
              <dd className="font-medium">WAV, MP3, AU</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Segment length</dt>
              <dd className="font-medium">2s (1s hop)</dd>
            </div>
          </dl>
        </CardContent>
      </Card>
    </div>
  )
}
