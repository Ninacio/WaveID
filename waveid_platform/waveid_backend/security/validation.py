"""
Input validation and sanitisation for uploads and path parameters.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from fastapi import HTTPException, UploadFile

from ..config import ALLOWED_EXTENSIONS, MAX_FILENAME_LENGTH, MAX_UPLOAD_MB

# Safe filename: alphanumerics plus common punctuation in music titles.
# Path separators and null bytes are stripped before this check.
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9._\- '(),&+#!]+$")
# track_id values are uuid4().hex (32 lowercase hex chars).
_TRACK_ID_RE = re.compile(r"^[a-f0-9]{32}$")

# Magic-byte signatures for allowed audio formats.
_MAGIC_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    ".wav": (b"RIFF",),
    ".mp3": (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xfa"),
    ".au": (b".snd",),
}


def sanitize_filename(raw: str | None) -> str:
    """Return a basename-only, sanitised filename or raise HTTP 400."""
    if not raw or not raw.strip():
        raise HTTPException(status_code=400, detail="Filename is required.")

    # Strip path components and null bytes.
    name = PurePosixPath(raw.replace("\x00", "").strip()).name
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if len(name) > MAX_FILENAME_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Filename exceeds maximum length of {MAX_FILENAME_LENGTH} characters.",
        )

    if not _SAFE_FILENAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail="Filename contains disallowed characters.",
        )

    ext = PurePosixPath(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    return name


def validate_track_id(track_id: str) -> str:
    """Validate catalogue track_id path parameter."""
    if not track_id or not _TRACK_ID_RE.fullmatch(track_id.strip()):
        raise HTTPException(status_code=400, detail="Invalid track ID format.")
    return track_id.strip()


def _matches_magic(contents: bytes, ext: str) -> bool:
    if len(contents) < 4:
        return False
    signatures = _MAGIC_SIGNATURES.get(ext, ())
    for sig in signatures:
        if ext == ".wav":
            if contents[:4] == b"RIFF" and len(contents) >= 12 and contents[8:12] == b"WAVE":
                return True
        elif contents[: len(sig)] == sig:
            return True
    return False


async def read_bounded_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Read upload in chunks; reject empty or oversized payloads."""
    chunks: list[bytes] = []
    total = 0
    chunk_size = 1024 * 1024

    while True:
        piece = await file.read(chunk_size)
        if not piece:
            break
        total += len(piece)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds maximum size of {MAX_UPLOAD_MB} MB.",
            )
        chunks.append(piece)

    contents = b"".join(chunks)
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return contents


def validate_upload(filename: str, contents: bytes) -> str:
    """
    Validate filename, size, and magic bytes.
    Returns the sanitised filename.
    """
    safe_name = sanitize_filename(filename)
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds maximum size of {MAX_UPLOAD_MB} MB.",
        )

    ext = PurePosixPath(safe_name).suffix.lower()
    if not _matches_magic(contents, ext):
        raise HTTPException(
            status_code=400,
            detail="File content does not match the declared audio format.",
        )

    return safe_name
