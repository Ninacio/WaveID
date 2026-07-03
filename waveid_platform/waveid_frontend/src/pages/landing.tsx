import { Link } from "react-router-dom"

import { Logo } from "@/components/brand/logo"
import { Button } from "@/components/ui/button"
import { ThemeToggle } from "@/components/theme/theme-toggle"
import { GlowyWavesHero } from "@/components/ui/glowy-waves-hero-shadcnui"

export function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-border/40 bg-background/70 px-5 backdrop-blur lg:px-10">
        <Logo />
        <div className="flex items-center gap-2">
          <ThemeToggle />
          <Button variant="ghost" size="sm" asChild>
            <a href="/docs" target="_blank" rel="noreferrer">
              API docs
            </a>
          </Button>
          <Button size="sm" asChild>
            <Link to="/dashboard">Open app</Link>
          </Button>
        </div>
      </header>
      <div className="px-4 py-4 lg:px-10 lg:py-6">
        <GlowyWavesHero />
      </div>
    </div>
  )
}
