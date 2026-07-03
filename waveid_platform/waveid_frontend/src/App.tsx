import { Navigate, Route, Routes } from "react-router-dom"

import { AppShell } from "@/components/layout/app-shell"
import { LandingPage } from "@/pages/landing"
import { SignalLabPage } from "@/pages/signal-lab"
import { DashboardPage } from "@/pages/dashboard"
import { QueryPage } from "@/pages/query"
import { CataloguePage } from "@/pages/catalogue"
import { TrackDetailPage } from "@/pages/track-detail"
import { SettingsPage } from "@/pages/settings"

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/signal-lab" element={<SignalLabPage />} />
      <Route element={<AppShell />}>
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/query" element={<QueryPage />} />
        <Route path="/catalogue" element={<CataloguePage />} />
        <Route path="/catalogue/:trackId" element={<TrackDetailPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
