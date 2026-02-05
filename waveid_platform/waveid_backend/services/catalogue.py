"""
In-memory catalogue storage for tracks and segments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from ..config import INDEX_DIR


_tracks: Dict[str, Dict[str, object]] = {}
_segments: Dict[str, Dict[str, object]] = {}
_track_segments: Dict[str, List[str]] = {}
_loaded = False
_CATALOGUE_PATH = INDEX_DIR / "catalogue.json"


def _load_state() -> None:
    global _tracks, _segments, _track_segments, _loaded
    if _loaded:
        return
    _loaded = True
    if not _CATALOGUE_PATH.exists():
        return
    data = json.loads(_CATALOGUE_PATH.read_text(encoding="utf-8"))
    _tracks = data.get("tracks", {})
    _segments = data.get("segments", {})
    _track_segments = data.get("track_segments", {})


def _save_state() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "tracks": _tracks,
        "segments": _segments,
        "track_segments": _track_segments,
    }
    _CATALOGUE_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def add_track(filename: str, duration: float, sr: int, model_version: str) -> str:
    _load_state()
    track_id = uuid4().hex
    _tracks[track_id] = {
        "track_id": track_id,
        "filename": filename,
        "duration": float(duration),
        "sample_rate": int(sr),
        "model_version": model_version,
    }
    _track_segments[track_id] = []
    _save_state()
    return track_id


def add_segments(track_id: str, segments: List[Dict[str, object]]) -> None:
    _load_state()
    if track_id not in _tracks:
        raise ValueError("Unknown track_id.")
    for segment in segments:
        segment_id = uuid4().hex
        record = {
            "segment_id": segment_id,
            "track_id": track_id,
            "start_time": float(segment["start_time"]),
            "end_time": float(segment["end_time"]),
            "embedding_id": str(segment["embedding_id"]),
        }
        _segments[segment_id] = record
        _track_segments[track_id].append(segment_id)
    _save_state()


def list_tracks() -> List[Dict[str, object]]:
    _load_state()
    results: List[Dict[str, object]] = []
    for track_id, meta in _tracks.items():
        results.append(
            {
                "track_id": track_id,
                "filename": meta["filename"],
                "duration": meta["duration"],
                "num_segments": len(_track_segments.get(track_id, [])),
            }
        )
    return results


def get_track(track_id: str) -> Dict[str, object] | None:
    _load_state()
    if track_id not in _tracks:
        return None
    meta = _tracks[track_id]
    segments = [
        _segments[segment_id] for segment_id in _track_segments.get(track_id, [])
    ]
    return {
        "track_id": track_id,
        "filename": meta["filename"],
        "duration": meta["duration"],
        "num_segments": len(segments),
        "segments": segments,
    }
