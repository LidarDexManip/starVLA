"""Split g1-pipette-tip-teleop-tipcrop into train / eval / local-holdout.

WHY SEPARATE DIRECTORIES and not one dataset with a subset manifest: the
loader computes normalisation statistics from `glob("data/*/*.parquet")`
over the dataset directory, NOT from the episodes listed in
meta/episodes.jsonl (datasets.py `_load_or_compute_statistics`). A split
that shared the parquet tree would normalise the training set using the
eval episodes' statistics — leakage that no amount of manifest filtering
would prevent. Each split therefore gets its own tree; the heavy files are
SYMLINKED, so this costs kilobytes, not gigabytes.

Two caches are deliberately NOT carried over, because both are computed
over whatever tree they were built in and would silently describe 64
episodes inside a 49-episode split:
    meta/stats_gr00t.json     normalisation statistics
    meta/steps_data_index.pkl frame index
Both regenerate on first use.

Episode NUMBERING IS PRESERVED (info.json maps episode_index straight into
the file name), so ep 37 is ep 37 in every split and in any eval report.

The split is SEEDED AND RANDOM, not contiguous. Recording order carries
real drift on this bench — the broken crops cluster in the 40s and 50s,
and bench lighting moved across the session — so a contiguous eval block
would measure a different bench, not a held-out policy.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

# Crops locked on the robot's forearm with no rack in frame, confirmed by
# eye over the whole episode (2026-08-13). 44/45 were spotted by bir; 40
# and 55 are the same failure and were found by the causal-vs-builder
# parity sweep, which flagged exactly {8,23,40,44,45,55} -- 8 and 23 keep
# the rack but drift off-centre late, and are KEPT.
DEFAULT_EXCLUDE = (40, 44, 45, 55)

#: Held out of BOTH splits, for bir's local bench testing. Episode 0 is
#: the deliberate choice: it is the only episode whose CLEAN (unboxed) rgb
#: exists locally at ~/datasets/g1-pipette-tip-teleop, so it is what the
#: console's openloop lane replays. Holding it out turns that lane from an
#: in-sample demo into a genuine generalisation check.
DEFAULT_HOLDOUT = 0

META_COPY = ("modality.json", "tasks.jsonl")
NEVER_COPY = ("stats_gr00t.json", "steps_data_index.pkl")


def load_episodes(src: Path) -> list[dict]:
    with open(src / "meta" / "episodes.jsonl") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build(src: Path, dst: Path, keep: list[int], eps: list[dict],
          video_keys: list[str], link: bool = True) -> int:
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)
    (dst / "data" / "chunk-000").mkdir(parents=True)
    for k in video_keys:
        (dst / "videos" / "chunk-000" / k).mkdir(parents=True)

    keepset = set(keep)
    rows = [e for e in eps if e["episode_index"] in keepset]
    frames = 0
    for e in rows:
        i = e["episode_index"]
        frames += int(e.get("length", 0))
        pq = f"episode_{i:06d}.parquet"
        s, d = src / "data" / "chunk-000" / pq, dst / "data" / "chunk-000" / pq
        if not s.exists():
            raise SystemExit(f"missing {s}")
        d.symlink_to(s) if link else shutil.copy2(s, d)
        for k in video_keys:
            mp = f"episode_{i:06d}.mp4"
            s2 = src / "videos" / "chunk-000" / k / mp
            d2 = dst / "videos" / "chunk-000" / k / mp
            if not s2.exists():
                raise SystemExit(f"missing {s2}")
            d2.symlink_to(s2) if link else shutil.copy2(s2, d2)

    with open(dst / "meta" / "episodes.jsonl", "w") as fh:
        for e in rows:
            fh.write(json.dumps(e) + "\n")
    for f in META_COPY:
        if (src / "meta" / f).exists():
            shutil.copy2(src / "meta" / f, dst / "meta" / f)
    info = json.load(open(src / "meta" / "info.json"))
    info["total_episodes"] = len(rows)
    info["total_frames"] = frames
    if "splits" in info:
        info["splits"] = {"train": f"0:{len(rows)}"}
    json.dump(info, open(dst / "meta" / "info.json", "w"), indent=2)
    for f in NEVER_COPY:
        assert not (dst / "meta" / f).exists(), f
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() /
                    "Datasets" / "g1-pipette-tip-teleop-tipcrop"))
    ap.add_argument("--out-prefix", default=None,
                    help="default: <src>-train / -eval")
    ap.add_argument("--exclude", default=",".join(map(str, DEFAULT_EXCLUDE)))
    ap.add_argument("--holdout", type=int, default=DEFAULT_HOLDOUT)
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--copy", action="store_true",
                    help="copy instead of symlink (for a remote sync)")
    a = ap.parse_args()

    src = Path(a.src).expanduser()
    prefix = a.out_prefix or str(src)
    eps = load_episodes(src)
    all_ids = sorted(e["episode_index"] for e in eps)
    video_keys = sorted(p.name for p in
                        (src / "videos" / "chunk-000").iterdir() if p.is_dir())

    excl = {int(x) for x in a.exclude.split(",") if x.strip()}
    missing = excl - set(all_ids)
    if missing:
        raise SystemExit(f"--exclude names episodes not in the dataset: {missing}")
    if a.holdout not in all_ids:
        raise SystemExit(f"--holdout {a.holdout} is not in the dataset")
    if a.holdout in excl:
        raise SystemExit(f"--holdout {a.holdout} is also excluded as broken")

    usable = [i for i in all_ids if i not in excl and i != a.holdout]
    if a.n_eval >= len(usable):
        raise SystemExit(f"--n-eval {a.n_eval} >= {len(usable)} usable")
    rng = random.Random(a.seed)
    shuffled = usable[:]
    rng.shuffle(shuffled)
    ev = sorted(shuffled[:a.n_eval])
    tr = sorted(shuffled[a.n_eval:])

    print(f"source          {src}")
    print(f"  {len(all_ids)} episodes, video keys {video_keys}")
    print(f"excluded (broken crops) {sorted(excl)}")
    print(f"local holdout           [{a.holdout}]  (in NEITHER split)")
    print(f"eval  ({len(ev):2d})  {ev}")
    print(f"train ({len(tr):2d})  {tr}\n")

    for name, ids in (("train", tr), ("eval", ev)):
        dst = Path(f"{prefix}-{name}")
        n = build(src, dst, ids, eps, video_keys, link=not a.copy)
        print(f"  wrote {dst}  ({len(ids)} episodes, {n} frames)")

    manifest = {
        "source": str(src), "seed": a.seed,
        "excluded_broken_crops": sorted(excl),
        "local_holdout": [a.holdout],
        "train": tr, "eval": ev,
    }
    mp = Path(f"{prefix}-split.json")
    json.dump(manifest, open(mp, "w"), indent=2)
    print(f"  wrote {mp}")


if __name__ == "__main__":
    main()
