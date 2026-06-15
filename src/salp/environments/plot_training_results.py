#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot RL Training Results from TensorBoard CSV exports.

Supports:
  - Single or multiple CSV files (one per run)
  - Exponential moving average (EMA) smoothing
  - Optional confidence band across multiple runs
  - Saving to file or interactive display

Usage:
  # Single run
  python plot_training_results.py results.csv

  # Multiple runs with legend labels
  python plot_training_results.py run1.csv run2.csv run3.csv \
      --labels "Run 1" "Run 2" "Run 3"

  # Compare specific tensorboard folders (reads all CSVs inside)
  python plot_training_results.py run1.csv run2.csv --smooth 0.97 \
      --title "SAC Training" --xlabel "Environment Steps" \
      --ylabel "Episode Reward" --save training_curve.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import uniform_filter1d

# ---------------------------------------------------------------------------
# Colour palette (colour-blind friendly)
# ---------------------------------------------------------------------------
COLORS = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#E91E63",  # pink
    "#795548",  # brown
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    """Load a TensorBoard-style CSV export (Wall time, Step, Value)."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Accept both 'Step'/'step' and 'Value'/'value'
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if lc == "step":
            rename[col] = "step"
        elif lc == "value":
            rename[col] = "value"
        elif lc in ("wall time", "wall_time", "walltime", "timestamp"):
            rename[col] = "wall_time"
    df = df.rename(columns=rename)

    required = {"step", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: could not find columns {missing}. "
            f"Found: {list(df.columns)}"
        )

    df = df.sort_values("step").reset_index(drop=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df


def ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average smoothing (same as TensorBoard's smoothing)."""
    smoothed = np.zeros_like(values, dtype=float)
    last = 0.0
    debias = 0.0
    for i, v in enumerate(values):
        last = last * alpha + (1 - alpha) * v
        debias = debias * alpha + (1 - alpha)
        smoothed[i] = last / debias
    return smoothed


def rolling_stats(
    runs: list[np.ndarray], steps: np.ndarray, window: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean ± std across runs on a common step grid.
    Runs are interpolated onto the shared grid.
    """
    # Build common step axis
    all_steps = sorted({s for r in runs for s in r[:, 0]})
    grid = np.array(all_steps)

    interp = []
    for r in runs:
        y = np.interp(grid, r[:, 0], r[:, 1])
        interp.append(y)

    arr = np.array(interp)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return grid, mean, std


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_training(
    csv_paths: list[Path],
    labels: list[str] | None = None,
    smooth: float = 0.9,
    show_raw: bool = True,
    aggregate: bool = False,
    title: str = "RL Training Curve",
    xlabel: str = "Environment Steps",
    ylabel: str = "Episode Reward",
    save_path: str | None = None,
    figsize: tuple[float, float] = (10, 5),
    step_scale: float = 1.0,
    ymin: float | None = None,
    ymax: float | None = None,
):
    """Core plotting routine."""
    if labels is None:
        labels = [p.stem for p in csv_paths]
    if len(labels) < len(csv_paths):
        labels += [p.stem for p in csv_paths[len(labels):]]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    # ---- aggregate mode: treat all CSVs as independent seeds of one run ----
    if aggregate and len(csv_paths) > 1:
        runs = []
        for path in csv_paths:
            df = load_csv(path)
            runs.append(df[["step", "value"]].values)

        grid, mean, std = rolling_stats(runs, None)
        grid = grid * step_scale

        smoothed_mean = ema_smooth(mean, smooth) if smooth > 0 else mean

        color = COLORS[0]
        ax.fill_between(
            grid,
            smoothed_mean - std,
            smoothed_mean + std,
            alpha=0.25,
            color=color,
            label="_nolegend_",
        )
        ax.plot(grid, smoothed_mean, color=color, linewidth=2.0,
                label=labels[0] if labels[0] else "Mean ± Std")

    # ---- individual runs -----------------------------------------------
    else:
        for i, (path, label) in enumerate(zip(csv_paths, labels)):
            color = COLORS[i % len(COLORS)]
            df = load_csv(path)
            steps = df["step"].values * step_scale
            values = df["value"].values

            if show_raw:
                ax.plot(steps, values, color=color, alpha=0.25,
                        linewidth=0.8, label="_nolegend_")

            if smooth > 0:
                smoothed = ema_smooth(values, smooth)
                ax.plot(steps, smoothed, color=color, linewidth=2.0,
                        label=label)
            else:
                ax.plot(steps, values, color=color, linewidth=1.5,
                        label=label)

    # ---- formatting -------------------------------------------------------
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)

    # Friendly x-axis tick labels (e.g. "1M", "500K")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, _: (
                f"{x/1e6:.1f}M" if abs(x) >= 1e6
                else f"{x/1e3:.0f}K" if abs(x) >= 1e3
                else f"{x:.0f}"
            )
        )
    )

    if len(csv_paths) > 1 or (labels and labels[0]):
        ax.legend(fontsize=10, framealpha=0.9)

    ax.tick_params(axis="both", labelsize=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to: {save_path}")
    else:
        plt.show()

    return fig, ax


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot RL training curves from TensorBoard CSV exports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "csvs",
        nargs="+",
        type=Path,
        metavar="CSV",
        help="One or more TensorBoard CSV export files.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="Legend labels for each CSV (in order).",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.9,
        metavar="ALPHA",
        help="EMA smoothing factor in [0, 1). 0 = no smoothing. Default: 0.9",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Hide the raw (un-smoothed) trace behind the smoothed line.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Treat all CSVs as seeds of one experiment and plot mean ± std.",
    )
    parser.add_argument(
        "--title",
        default="RL Training Curve",
        help="Plot title.",
    )
    parser.add_argument(
        "--xlabel",
        default="Environment Steps",
        help="X-axis label.",
    )
    parser.add_argument(
        "--ylabel",
        default="Episode Reward",
        help="Y-axis label.",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="Save figure to this path instead of displaying it.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[10, 5],
        metavar=("W", "H"),
        help="Figure width and height in inches. Default: 10 5",
    )
    parser.add_argument(
        "--step-scale",
        type=float,
        default=1.0,
        metavar="SCALE",
        help="Multiply all step values by this factor (e.g. env_steps_per_update).",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Y-axis lower limit.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Y-axis upper limit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for p in args.csvs:
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    plot_training(
        csv_paths=args.csvs,
        labels=args.labels,
        smooth=args.smooth,
        show_raw=not args.no_raw,
        aggregate=args.aggregate,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        save_path=args.save,
        figsize=tuple(args.figsize),
        step_scale=args.step_scale,
        ymin=args.ymin,
        ymax=args.ymax,
    )



if __name__ == "__main__":
    import sys as _sys

    # If called with CLI arguments, use the full argparse interface.
    if len(_sys.argv) > 1:
        main()
    else:
        # ---------------------------------------------------------------
        # Inline run: plot each CSV in its own subplot, stacked vertically.
        # ---------------------------------------------------------------
        _data_dir = Path(r"D:\presentations\ONR_2027")
        _csvs = [
            _data_dir / "salp_robot_front_pos_run_2.csv",
            _data_dir / "salp_robot_front_pos_run_2 (1).csv",
        ]
        _csvs = [p for p in _csvs if p.exists()]

        if not _csvs:
            print("No CSV files found in D:\\presentations\\ONR_2027\\")
            print("Run with a CSV path as an argument:  python plot_training_results.py <file.csv>")
        else:
            plt.style.use("seaborn-v0_8-whitegrid")
            fig, axes = plt.subplots(
                len(_csvs), 1,
                figsize=(10, 5 * len(_csvs)),
                sharex=False,
            )
            # Ensure axes is always a list even for a single file
            if len(_csvs) == 1:
                axes = [axes]

            _subtitles = [
                "salp_robot_front_pos_run_2",
                "salp_robot_front_pos_run_2 (1)",
            ]
            _ylabels = [
                "Episode Length",
                "Episode Reward",
            ]

            for ax, csv_path, subtitle, ylabel in zip(axes, _csvs, _subtitles, _ylabels):
                df = load_csv(csv_path)
                steps = df["step"].values
                values = df["value"].values

                # Raw trace
                ax.plot(steps, values, color=COLORS[0], alpha=0.25,
                        linewidth=0.8)
                # Smoothed trace
                smoothed = ema_smooth(values, alpha=0.9)
                ax.plot(steps, smoothed, color=COLORS[0], linewidth=2.0)

                ax.set_xlabel("Steps", fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11)
                ax.tick_params(axis="both", labelsize=10)
                ax.xaxis.set_major_formatter(
                    mticker.FuncFormatter(
                        lambda x, _: (
                            f"{x/1e6:.1f}M" if abs(x) >= 1e6
                            else f"{x/1e3:.0f}K" if abs(x) >= 1e3
                            else f"{x:.0f}"
                        )
                    )
                )

            fig.tight_layout()
            save_path = None   # change to a .png path to save instead of displaying
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Saved to: {save_path}")
            else:
                plt.show()
