"""Pick VideoColorTemperature.strength from what the bench actually does.

The augmentation is only worth its risk if it spans the colour range the
deployment produces. Measured on the pipette bench 2026-08-13, median hue
over every pixel brighter than the curtain ("scene hue"):

    68   the training data
    93   the same bench hours later, auto WB, room lighting changed
    13   the same bench with white balance pinned to 5800K

So training at hue 68 has to cover roughly 40..100 to absorb ordinary
lighting drift. This applies the transform to real training frames at a
ladder of strengths and reports the scene hue it produces, so the constant
is chosen against a measurement instead of by eye.
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "/home/jren313/Projects/starVLA")
from starVLA.dataloader.gr00t_lerobot.transform.video import (  # noqa: E402
    _ColourTemperatureShift)

TRAIN_HUE = 68.0          # the training data's scene hue
OBSERVED = {"bench, lights changed": 93.0, "bench, WB pinned 5800K": 13.0}


def scene_hue(bgr: np.ndarray) -> float:
    p = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    lit = p[p[:, 2] > 80]
    return float(np.median(lit[:, 0])) if len(lit) else float("nan")


ap = argparse.ArgumentParser()
ap.add_argument("--video", default="/home/jren313/Datasets/"
                "g1-pipette-tip-teleop-tipcrop/videos/chunk-000/"
                "observation.images.rgb/episode_000001.mp4")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--draws", type=int, default=60)
a = ap.parse_args()

cap = cv2.VideoCapture(a.video)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames = []
for p in np.linspace(0.1, 0.9, a.n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * p))
    ok, f = cap.read()
    if ok:
        frames.append(f)
cap.release()

base = float(np.median([scene_hue(f) for f in frames]))
print(f"source frames: {len(frames)}   scene hue as recorded: {base:.0f}")
print(f"training-data reference: {TRAIN_HUE:.0f}")
for k, v in OBSERVED.items():
    print(f"  deployment observed: {v:5.0f}   ({k})")
print(f"\n{'strength':>9}{'hue p5':>9}{'hue p50':>9}{'hue p95':>9}   covers")
print("-" * 62)
for s in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
    tf = _ColourTemperatureShift(s)
    hues = []
    for _ in range(a.draws):
        f = frames[np.random.randint(len(frames))]
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255.0
        out = tf(t) if s > 0 else t
        arr = (out[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        hues.append(scene_hue(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)))
    q = np.percentile(hues, [5, 50, 95])
    lo, hi = q[0], q[2]
    got = [k for k, v in OBSERVED.items() if lo <= v <= hi]
    print(f"{s:>9.2f}{lo:>9.0f}{q[1]:>9.0f}{hi:>9.0f}   "
          f"{', '.join(got) if got else '-'}")
