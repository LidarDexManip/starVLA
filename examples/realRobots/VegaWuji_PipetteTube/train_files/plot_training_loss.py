#!/usr/bin/env python3
"""Plot action_dit_loss values emitted by StarVLA training."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOSS_PATTERN = re.compile(
    r"Step\s+(\d+), Loss:"
    r"(?:(?!Step\s+\d+, Loss:).)*?"
    r"['\"]action_dit_loss['\"]:\s*([0-9.eE+-]+)",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, help="StarVLA train.log")
    parser.add_argument("output", type=Path, help="Output PNG path")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=25,
        help="Moving-average window in logged points (default: 25)",
    )
    parser.add_argument(
        "--interrupted-at",
        type=int,
        default=None,
        help="Optional raw step where training stopped",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.log.read_text(errors="replace")

    # If a resumed run logs a step more than once, retain its latest value.
    samples = {
        int(step): float(loss) for step, loss in LOSS_PATTERN.findall(text)
    }
    if not samples:
        raise SystemExit(f"No action_dit_loss entries found in {args.log}")

    steps = np.asarray(sorted(samples), dtype=int)
    losses = np.asarray([samples[step] for step in steps], dtype=float)

    window = min(args.smooth_window, len(losses))
    if window < 1:
        raise SystemExit("--smooth-window must be positive")
    weights = np.full(window, 1.0 / window)
    smoothed = np.convolve(losses, weights, mode="valid")
    smooth_steps = steps[window - 1 :]

    fig, ax = plt.subplots(figsize=(10, 5.625))
    ax.plot(steps, losses, color="#5B8FF9", alpha=0.27, linewidth=0.9, label="Logged loss")
    ax.plot(
        smooth_steps,
        smoothed,
        color="#D9485F",
        linewidth=2.2,
        label=f"Moving average ({window} logged points)",
    )

    for checkpoint in range(1000, int(steps[-1]) + 1, 1000):
        ax.axvline(checkpoint, color="#777777", alpha=0.2, linewidth=0.8, linestyle="--")
    if args.interrupted_at is not None:
        ax.axvline(
            args.interrupted_at,
            color="#8C1D40",
            linewidth=1.2,
            linestyle=":",
            label=f"Interrupted at step {args.interrupted_at:,}",
        )

    ax.set_title("Vega/Wuji phase-labeled GR00T-N1.7 training")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Action DiT loss")
    ax.grid(axis="y", color="#BBBBBB", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    ax.margins(x=0.01)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "action_dit_loss"])
        writer.writerows(zip(steps, losses, strict=True))

    print(f"Parsed {len(steps)} points (steps {steps[0]}-{steps[-1]}).")
    print(f"Wrote {args.output}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
