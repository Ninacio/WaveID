"""
Analyse severity sweep CSV and produce:
  1. Per-severity mean score table (printed)
  2. Matplotlib plots saved as PDF for dissertation inclusion
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = Path("data/index/severity_sweep.csv")
OUT_DIR = Path("data/index/sweep_plots")


def load_rows() -> list[dict]:
    with CSV_PATH.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(rows: list[dict]) -> dict[str, dict[float, list[float]]]:
    """transform -> severity_value -> [scores]"""
    result: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        t = r["transform"]
        sv = float(r["severity_value"])
        sc = float(r["top_score"]) if r["top_score"] else 0.0
        result[t][sv].append(sc)
    return result


def print_table(agg: dict) -> None:
    for kind in ["pitch", "tempo", "noise", "crop", "lossy"]:
        print(f"\n=== {kind.upper()} ===")
        print(f"  {'Severity':>12}  {'Mean Score':>10}  {'Min':>8}  {'Max':>8}  {'StdDev':>8}")
        for sv in sorted(agg[kind].keys()):
            scores = agg[kind][sv]
            mn = statistics.mean(scores)
            lo = min(scores)
            hi = max(scores)
            sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
            print(f"  {sv:>12.2f}  {mn:>10.4f}  {lo:>8.4f}  {hi:>8.4f}  {sd:>8.4f}")


def plot_pitch(agg: dict, out: Path) -> None:
    data = agg["pitch"]
    xs = sorted(data.keys())
    ys = [statistics.mean(data[x]) for x in xs]
    lo = [min(data[x]) for x in xs]
    hi = [max(data[x]) for x in xs]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "o-", color="#2563eb", linewidth=2, markersize=6, label="Mean score")
    ax.fill_between(xs, lo, hi, alpha=0.15, color="#2563eb", label="Min–Max range")
    ax.set_xlabel("Pitch shift (semitones)", fontsize=11)
    ax.set_ylabel("Match score", fontsize=11)
    ax.set_title("Score vs Pitch Shift Severity", fontsize=13, fontweight="bold")
    ax.set_ylim(0.85, 1.005)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(x):+d}" for x in xs])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pitch_sweep.pdf", bbox_inches="tight")
    fig.savefig(out / "pitch_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out / 'pitch_sweep.pdf'}")


def plot_tempo(agg: dict, out: Path) -> None:
    data = agg["tempo"]
    xs = sorted(data.keys())
    ys = [statistics.mean(data[x]) for x in xs]
    lo = [min(data[x]) for x in xs]
    hi = [max(data[x]) for x in xs]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "s-", color="#059669", linewidth=2, markersize=6, label="Mean score")
    ax.fill_between(xs, lo, hi, alpha=0.15, color="#059669", label="Min–Max range")
    ax.set_xlabel("Tempo rate (×)", fontsize=11)
    ax.set_ylabel("Match score", fontsize=11)
    ax.set_title("Score vs Tempo Change Severity", fontsize=13, fontweight="bold")
    ax.set_ylim(0.95, 1.005)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:.2f}" for x in xs], rotation=45)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "tempo_sweep.pdf", bbox_inches="tight")
    fig.savefig(out / "tempo_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out / 'tempo_sweep.pdf'}")


def plot_noise(agg: dict, out: Path) -> None:
    data = agg["noise"]
    xs = sorted(data.keys())
    ys = [statistics.mean(data[x]) for x in xs]
    lo = [min(data[x]) for x in xs]
    hi = [max(data[x]) for x in xs]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "D-", color="#dc2626", linewidth=2, markersize=6, label="Mean score")
    ax.fill_between(xs, lo, hi, alpha=0.15, color="#dc2626", label="Min–Max range")
    ax.set_xlabel("Signal-to-Noise Ratio (dB)", fontsize=11)
    ax.set_ylabel("Match score", fontsize=11)
    ax.set_title("Score vs Noise Injection Severity", fontsize=13, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(x)}" for x in xs])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "noise_sweep.pdf", bbox_inches="tight")
    fig.savefig(out / "noise_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out / 'noise_sweep.pdf'}")


def plot_crop(agg: dict, out: Path) -> None:
    data = agg["crop"]
    xs = sorted(data.keys())
    ys = [statistics.mean(data[x]) for x in xs]
    lo = [min(data[x]) for x in xs]
    hi = [max(data[x]) for x in xs]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "^-", color="#7c3aed", linewidth=2, markersize=6, label="Mean score")
    ax.fill_between(xs, lo, hi, alpha=0.15, color="#7c3aed", label="Min–Max range")
    ax.set_xlabel("Seconds cropped from end", fontsize=11)
    ax.set_ylabel("Match score", fontsize=11)
    ax.set_title("Score vs Crop Severity", fontsize=13, fontweight="bold")
    ax.set_ylim(0.95, 1.005)
    ax.set_xticks(xs)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "crop_sweep.pdf", bbox_inches="tight")
    fig.savefig(out / "crop_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out / 'crop_sweep.pdf'}")


def plot_lossy(agg: dict, out: Path) -> None:
    data = agg["lossy"]
    xs = sorted(data.keys())
    ys = [statistics.mean(data[x]) for x in xs]
    lo = [min(data[x]) for x in xs]
    hi = [max(data[x]) for x in xs]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "v-", color="#ea580c", linewidth=2, markersize=6, label="Mean score")
    ax.fill_between(xs, lo, hi, alpha=0.15, color="#ea580c", label="Min–Max range")
    ax.set_xlabel("MP3 bitrate (kbps)", fontsize=11)
    ax.set_ylabel("Match score", fontsize=11)
    ax.set_title("Score vs Lossy Compression Bitrate", fontsize=13, fontweight="bold")
    ax.set_ylim(0.95, 1.005)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(x)}" for x in xs])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "lossy_sweep.pdf", bbox_inches="tight")
    fig.savefig(out / "lossy_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out / 'lossy_sweep.pdf'}")


def plot_combined_summary(agg: dict, out: Path) -> None:
    """Single figure with all transforms' mean scores overlaid."""
    fig, ax = plt.subplots(figsize=(8, 5))

    configs = [
        ("pitch", "Pitch (semitones)", "o-", "#2563eb"),
        ("tempo", "Tempo (×)", "s-", "#059669"),
        ("noise", "Noise (SNR dB)", "D-", "#dc2626"),
        ("crop", "Crop (seconds)", "^-", "#7c3aed"),
        ("lossy", "MP3 (kbps)", "v-", "#ea580c"),
    ]

    for kind, label, marker, color in configs:
        data = agg[kind]
        xs = sorted(data.keys())
        ys = [statistics.mean(data[x]) for x in xs]
        norm_xs = list(range(len(xs)))
        ax.plot(norm_xs, ys, marker, color=color, linewidth=1.8, markersize=5, label=label)

    ax.set_ylabel("Mean match score", fontsize=11)
    ax.set_xlabel("Severity index (low → high)", fontsize=11)
    ax.set_title("Score Degradation Across All Transform Types", fontsize=13, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "combined_sweep.pdf", bbox_inches="tight")
    fig.savefig(out / "combined_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out / 'combined_sweep.pdf'}")


def main() -> None:
    rows = load_rows()
    agg = aggregate(rows)
    print_table(agg)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_pitch(agg, OUT_DIR)
    plot_tempo(agg, OUT_DIR)
    plot_noise(agg, OUT_DIR)
    plot_crop(agg, OUT_DIR)
    plot_lossy(agg, OUT_DIR)
    plot_combined_summary(agg, OUT_DIR)


if __name__ == "__main__":
    main()
