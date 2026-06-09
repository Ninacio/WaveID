"""
API-key authentication for protected routes.

Keys are loaded from the WAVEID_API_KEY environment variable only.
Never hardcode credentials in source code.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Request

from ..config import API_KEY, API_KEY_CONFIGURED


def _extract_api_key(
    x_api_key: str | None,
    authorization: str | None,
) -> str | None:
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        return token or None
    return None


def verify_api_key_value(provided: str | None) -> bool:
    """Constant-time comparison against configured API key."""
    if not API_KEY_CONFIGURED or not provided:
        return False
    return hmac.compare_digest(provided, API_KEY)


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    """FastAPI dependency: reject requests without a valid API key."""
    if not API_KEY_CONFIGURED:
        raise HTTPException(
            status_code=503,
            detail="Authentication is not configured on this server.",
        )

    provided = _extract_api_key(x_api_key, authorization)
    if not verify_api_key_value(provided):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
