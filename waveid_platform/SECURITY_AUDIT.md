# WaveID Security Audit

Last updated: 2026-06-09

This document records security controls implemented in WaveID and remaining vulnerabilities.

## Implemented Controls

### 1. Rate limiting
- **All endpoints** are rate-limited via `slowapi` (default: `100/minute` per client IP, configurable via `WAVEID_RATE_LIMIT_DEFAULT`).
- **Authentication route** `POST /auth/verify` is limited to **5 attempts per 15 minutes** per IP (`WAVEID_RATE_LIMIT_AUTH`).

### 2. Secrets management
- Full codebase scan found **no hardcoded API keys, tokens, or passwords**.
- `WAVEID_API_KEY` and all tunable settings load from **environment variables** (`waveid_backend/config.py`).
- `.env.example` documents required variables; `.gitignore` blocks `.env`, credentials JSON, and secret files from git.
- Frontend (`static/index.html`) contains **no secrets**; API keys must never be embedded client-side.

### 3. Input validation and payload limits
- **Filename sanitisation**: basename only, no path traversal, no null bytes, allowed character whitelist, max length.
- **Extension whitelist**: `.wav`, `.mp3`, `.au` only.
- **Magic-byte validation**: file content must match declared format (not extension alone).
- **Bounded upload reads**: streaming read with hard size cap (`WAVEID_MAX_UPLOAD_MB`).
- **Empty file rejection**.
- **track_id path parameter**: strict 32-char hex UUID format.
- **Auth body**: Pydantic model with min/max length on `api_key`.
- **Audio duration cap** enforced after decode.

### 4. Authentication
- `POST /auth/verify` validates API keys (constant-time comparison via `hmac.compare_digest`).
- `POST /reset-catalogue` requires API key when `WAVEID_API_KEY` is set.
- `POST /ingest-track` requires API key when both `WAVEID_API_KEY` and `WAVEID_REQUIRE_API_KEY=true`.
- Demo/dev mode: when no API key is configured, ingest/query/reset remain open (logged warning on startup).

### 5. Security headers
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy` restricting camera/microphone/geolocation
- Content-Security-Policy for the static UI

### 6. Optional CORS lockdown
- CORS middleware only enabled when `WAVEID_CORS_ORIGINS` is set explicitly.

---

## Remaining Vulnerabilities and Risks

### High

| Issue | Risk | Mitigation recommendation |
|-------|------|---------------------------|
| **No HTTPS enforcement in app** | Credentials and uploads can be intercepted if deployed over plain HTTP | Terminate TLS at reverse proxy (nginx, Caddy, cloud load balancer) |
| **In-memory rate limiting** | Limits reset on restart; ineffective behind many IPs or a botnet | Use Redis-backed rate limiter (e.g. `slowapi` + Redis) for production |
| **No authentication on `/query` or `/catalogue`** | Anyone can query or enumerate the catalogue; query endpoint is CPU-heavy (DoS vector) | Add API key or session auth; add stricter per-route limits on `/query` |
| **Brute-force on ingest when auth disabled** | Default demo mode allows unauthenticated ingest | Set `WAVEID_API_KEY` + `WAVEID_REQUIRE_API_KEY=true` in production |

### Medium

| Issue | Risk | Mitigation recommendation |
|-------|------|---------------------------|
| **OpenAPI docs exposed (`/docs`, `/redoc`)** | Attack surface enumeration | Disable in production via env flag or protect behind auth |
| **Startup catalogue wipe default (`WAVEID_RESET_ON_STARTUP=true`)** | Data loss on deploy/restart | Set `WAVEID_RESET_ON_STARTUP=false` in production |
| **IP-based rate limiting only** | Shared NAT users blocked together; attackers rotate IPs | Combine with API key or user-based limits |
| **MP3 magic-byte check is partial** | Some malformed MP3 may pass initial bytes check | Add full decode validation (already partially done in `load_audio_from_bytes`) |
| **`.au` in allowed extensions but limited decode support** | Unexpected 400 errors or inconsistent behaviour | Add `.au` decoder in `audio_io.py` or remove from allowed list |
| **No request logging / audit trail** | Hard to detect abuse post-incident | Add structured logging for auth failures and upload events |
| **Static UI calls `/reset-catalogue` without API key** | Reset fails when auth is enabled | Proxy reset through authenticated backend or remove from public UI |

### Low

| Issue | Risk | Mitigation recommendation |
|-------|------|---------------------------|
| **No virus/malware scanning on uploads** | Malicious files stored on disk | Scan uploads or use object storage with scanning |
| **Query embeddings returned in API response** | Information leakage about internal representation | Omit `query_embedding` from public responses |
| **No Content-Length pre-check** | Client may send large body before rejection | Enforce max body size at reverse proxy |
| **Single-process in-memory search** | Memory exhaustion with large catalogues | Cap catalogue size; move to FAISS/ANN with limits |
| **Dependency vulnerabilities** | Supply-chain risk in PyTorch, librosa, etc. | Run `pip audit` / Dependabot regularly |
| **CSP allows `'unsafe-inline'` scripts** | Reduced XSS protection for inline JS in `index.html` | Move scripts to external file with nonce-based CSP |

---

## Production Checklist

Before exposing WaveID to the public internet:

1. Set `WAVEID_API_KEY` to a strong random value (32+ chars).
2. Set `WAVEID_REQUIRE_API_KEY=true`.
3. Set `WAVEID_RESET_ON_STARTUP=false`.
4. Deploy behind HTTPS.
5. Restrict or disable `/docs` and `/redoc`.
6. Set `WAVEID_CORS_ORIGINS` to your frontend origin only.
7. Use Redis-backed rate limiting.
8. Never embed API keys in `static/index.html` or any frontend bundle.

---

## Secret Scan Summary

Scanned patterns: `password=`, `api_key=`, `secret=`, `sk-`, `AKIA`, `ghp_`, Bearer tokens, hardcoded credentials.

**Result:** No matches in WaveID source code. Runtime secrets must be supplied via environment variables only.
