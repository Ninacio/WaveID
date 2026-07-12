# WaveID Frontend

Modern web client for the WaveID audio identification platform, built with
Vite + React + TypeScript, Tailwind CSS v4, and shadcn-style components.

## Stack

- **Vite + React 19 + TypeScript** - app foundation
- **Tailwind CSS v4** (`@tailwindcss/vite`) - styling, with WaveID purple/gold theme tokens
- **shadcn/ui conventions** - primitives in `src/components/ui`
- **React Router** - routing
- **TanStack Query** - data fetching/caching against the FastAPI backend
- **framer-motion** - hero animation
- **sonner** - toast notifications

## Getting started

```bash
cd waveid_platform/waveid_frontend
npm install
npm run dev        # http://localhost:5173
```

The dev server proxies backend routes (`/health`, `/query`, `/catalogue`,
`/ingest-track`, `/reset-catalogue`, `/auth`) to the FastAPI backend. Start the
backend separately:

```bash
cd waveid_platform
uvicorn waveid_backend.main:app --reload --port 8000
```

Override the backend target with `VITE_BACKEND_URL` if it runs elsewhere.

## Build

```bash
npm run build      # type-check + production build to dist/
npm run preview    # preview the production build
```

## Structure

```
src/
  components/
    brand/      WaveID logo + wave mark
    common/     Reusable pieces (dropzone, page header, empty state, ...)
    layout/     App shell (sidebar, topbar, health pill)
    query/      Match card
    ui/         shadcn primitives + the glowy waves hero
  lib/          API client (api.ts) + cn() util
  pages/        Landing, Dashboard, Query, Catalogue, Track detail, Settings
```

## Configuration

| Env var            | Default                 | Purpose                              |
| ------------------ | ----------------------- | ------------------------------------ |
| `VITE_BACKEND_URL` | `http://localhost:8000` | Dev proxy target for backend routes  |
| `VITE_API_BASE`    | `""` (same-origin)      | Production API base URL for fetches  |
