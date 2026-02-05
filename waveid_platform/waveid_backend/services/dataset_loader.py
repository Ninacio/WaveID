"""
Dataset ingestion utilities.

Provides a helper to ingest a folder of audio files into the in-memory
catalogue and search index using the same pipeline as the API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from ..config import (
    ALLOWED_EXTENSIONS,
    HOP_SECONDS,
    MAX_DURATION_SECONDS,
    MODEL_VERSION,
    MONO,
    NORMALIZE,
    REFERENCE_DIR,
    SAMPLE_RATE,
    SEGMENT_SECONDS,
)
from .audio_io import load_audio_from_bytes
from .catalogue import add_segments, add_track
from .embedding import extract_embedding
from .search import add_reference_embeddings
from .segmentation import segment_audio


def _iter_audio_files(root: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in root.glob(pattern):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            yield path


def ingest_dataset(
    dataset_dir: str | Path,
    recursive: bool = True,
    limit: int | None = None,
    skip_errors: bool = True,
) -> Dict[str, object]:
    """Ingest a dataset directory into the catalogue.

    Args:
        dataset_dir: Root folder containing audio files.
        recursive: Whether to scan subdirectories.
        limit: Optional maximum number of files to ingest.
        skip_errors: Continue ingestion if a file fails to process.

    Returns:
        Summary dict with counts and error details.
    """
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    files = list(_iter_audio_files(root, recursive))
    if limit is not None:
        files = files[:limit]

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    errors: List[Dict[str, str]] = []
    tracks_ingested = 0
    segments_ingested = 0

    for path in files:
        try:
            contents = path.read_bytes()
            waveform, sr = load_audio_from_bytes(
                contents,
                filename=path.name,
                target_sr=SAMPLE_RATE,
                mono=MONO,
                normalize=NORMALIZE,
                max_duration_seconds=MAX_DURATION_SECONDS,
            )
            duration_seconds = float(waveform.size / sr) if sr else 0.0
            track_id = add_track(path.name, duration_seconds, sr, MODEL_VERSION)
            segments = segment_audio(waveform, sr, SEGMENT_SECONDS, HOP_SECONDS)
            embeddings = [extract_embedding(seg.samples, sr) for seg in segments]
            embedding_ids = add_reference_embeddings(embeddings)
            segment_records = [
                {
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "embedding_id": embedding_id,
                }
                for segment, embedding_id in zip(segments, embedding_ids)
            ]
            add_segments(track_id, segment_records)
            reference_path = REFERENCE_DIR / f"{track_id}{path.suffix.lower()}"
            reference_path.write_bytes(contents)

            tracks_ingested += 1
            segments_ingested += len(segments)
        except Exception as exc:  # noqa: BLE001
            if not skip_errors:
                raise
            errors.append({"file": str(path), "error": str(exc)})

    return {
        "tracks_ingested": tracks_ingested,
        "segments_ingested": segments_ingested,
        "errors": errors,
    }
