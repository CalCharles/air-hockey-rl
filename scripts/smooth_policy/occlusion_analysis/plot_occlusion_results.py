#!/usr/bin/env python3
"""Generate visualization artifacts from occlusion analysis outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot occlusion analysis outputs.")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory produced by analyze_occlusion_patterns.py",
    )
    return parser.parse_args()


def load_inputs(output_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    summary_path = output_dir / "occlusion_summary.json"
    arrays_path = output_dir / "occlusion_arrays.npz"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    if not arrays_path.exists():
        raise FileNotFoundError(f"Missing arrays file: {arrays_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    arrays = dict(np.load(arrays_path))
    return summary, arrays


def _imshow_heatmap(ax: plt.Axes, hist: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, title: str) -> None:
    img = ax.imshow(
        hist.T,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)


def plot_visible_vs_occluded(output_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _imshow_heatmap(
        axes[0],
        arrays["hist_visible"],
        arrays["x_edges"],
        arrays["y_edges"],
        "Visible Puck Position Heatmap",
    )
    _imshow_heatmap(
        axes[1],
        arrays["hist_occluded"],
        arrays["x_edges"],
        arrays["y_edges"],
        "Occluded Puck Position Heatmap",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "puck_visible_vs_occluded_heatmap.png", dpi=160)
    plt.close(fig)


def plot_transition_heatmap(output_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _imshow_heatmap(
        axes[0],
        arrays["hist_occ_start"],
        arrays["x_edges"],
        arrays["y_edges"],
        "Occlusion Start Transition Heatmap",
    )
    _imshow_heatmap(
        axes[1],
        arrays["hist_occ_end"],
        arrays["x_edges"],
        arrays["y_edges"],
        "Occlusion End Transition Heatmap",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "occlusion_transition_heatmap.png", dpi=160)
    plt.close(fig)


def plot_runlength_hist(output_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    run_lengths = arrays.get("run_lengths_frames", np.zeros(0))
    fig, ax = plt.subplots(figsize=(8, 5))
    if run_lengths.size > 0:
        bins = np.arange(1, max(int(np.max(run_lengths)) + 2, 3))
        ax.hist(run_lengths, bins=bins, edgecolor="black", alpha=0.8)
        ax.set_xlim(0.5, max(2.5, bins[-1]))
    else:
        ax.text(0.5, 0.5, "No occlusion runs", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Run length (frames)")
    ax.set_ylabel("Count")
    ax.set_title("Occlusion Run Length Distribution")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "occlusion_runlength_hist.png", dpi=160)
    plt.close(fig)


def plot_window_counts(output_dir: Path, arrays: dict[str, np.ndarray], summary: dict) -> None:
    w1 = arrays.get("window_1s_counts", np.zeros(0))
    w5 = arrays.get("window_5s_counts", np.zeros(0))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if w1.size > 0:
        bins1 = np.arange(0, int(np.max(w1)) + 2) - 0.5
        axes[0].hist(w1, bins=bins1, edgecolor="black", alpha=0.8)
    else:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_title(
        f"1s Window Occlusion Counts\nFano={summary['global']['burstiness']['window_1s_fano']:.3f}"
    )
    axes[0].set_xlabel("Occluded frames in 1s window")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.25)

    if w5.size > 0:
        bins5 = np.arange(0, int(np.max(w5)) + 2) - 0.5
        axes[1].hist(w5, bins=bins5, edgecolor="black", alpha=0.8)
    else:
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_title(
        f"5s Window Occlusion Counts\nFano={summary['global']['burstiness']['window_5s_fano']:.3f}"
    )
    axes[1].set_xlabel("Occluded frames in 5s window")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "occlusion_window_counts.png", dpi=160)
    plt.close(fig)


def plot_paddle_occ_heatmap(output_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    _imshow_heatmap(
        ax,
        arrays["hist_paddle_occluded"],
        arrays["x_edges"],
        arrays["y_edges"],
        "Paddle Position During Occlusions",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "paddle_occluded_heatmap.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    summary, arrays = load_inputs(output_dir)

    plot_visible_vs_occluded(output_dir, arrays)
    plot_transition_heatmap(output_dir, arrays)
    plot_runlength_hist(output_dir, arrays)
    plot_window_counts(output_dir, arrays, summary)
    plot_paddle_occ_heatmap(output_dir, arrays)

    print("=== Occlusion Plot Generation Complete ===")
    print(f"Output directory: {output_dir}")
    print("Generated:")
    print("- puck_visible_vs_occluded_heatmap.png")
    print("- occlusion_transition_heatmap.png")
    print("- occlusion_runlength_hist.png")
    print("- occlusion_window_counts.png")
    print("- paddle_occluded_heatmap.png")


if __name__ == "__main__":
    main()

