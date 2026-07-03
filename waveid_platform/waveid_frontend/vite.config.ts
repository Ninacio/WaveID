import path from "path"
import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"

// https://vite.dev/config/
const BACKEND_URL = process.env.VITE_BACKEND_URL ?? "http://localhost:8000"

// Backend (FastAPI) routes proxied in dev so the SPA can call them same-origin.
const apiRoutes = [
  "/health",
  "/query",
  "/catalogue",
  "/ingest-track",
  "/reset-catalogue",
  "/auth",
]

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: Object.fromEntries(
      apiRoutes.map((route) => [
        route,
        { target: BACKEND_URL, changeOrigin: true },
      ])
    ),
  },
})
