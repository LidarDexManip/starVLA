#!/usr/bin/env python3
"""Validate RFT's LeRobot v2.1 dataset and optionally decode one GR00T sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Running this file directly puts only ``train_files`` on ``sys.path``.  Add
# the repository root so the in-tree StarVLA package is importable even when it
# has not been installed as an editable wheel in the active environment.
STARVLA_ROOT = Path(__file__).resolve().parents[4]
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))


EXPECTED_DIM = 54
EXPECTED_FPS = 30
EXPECTED_VIEWS = (
    "ego_view",
    "side_view",
    "left_wrist_view",
    "right_wrist_view",
)
EXPECTED_SLICES = {
    "left_arm": (0, 7),
    "left_hand": (7, 27),
    "right_arm": (27, 34),
    "right_hand": (34, 54),
}
LANGUAGE_KEY = "human.task_description"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_layout(dataset: Path) -> tuple[dict, list[dict]]:
    import pandas as pd

    meta = dataset / "meta"
    required = [meta / name for name in ("info.json", "modality.json", "episodes.jsonl", "tasks.jsonl")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete LeRobot dataset; missing: {missing}")

    info = json.loads((meta / "info.json").read_text())
    modality = json.loads((meta / "modality.json").read_text())
    episodes = read_jsonl(meta / "episodes.jsonl")
    tasks = read_jsonl(meta / "tasks.jsonl")

    assert info["codebase_version"] == "v2.1", info["codebase_version"]
    assert int(info["fps"]) == EXPECTED_FPS, info["fps"]
    assert info["features"]["observation.state"]["shape"] == [EXPECTED_DIM]
    assert info["features"]["action"]["shape"] == [EXPECTED_DIM]
    assert modality["annotation"][LANGUAGE_KEY]["original_key"] == "task_index"
    for group, (start, end) in EXPECTED_SLICES.items():
        for kind in ("state", "action"):
            entry = modality[kind][group]
            assert (entry["start"], entry["end"]) == (start, end), (kind, group, entry)
    for view in EXPECTED_VIEWS:
        original = modality["video"][view]["original_key"]
        assert original == f"observation.images.{view}", original

    if not episodes:
        raise ValueError("dataset has no episodes")
    assert tasks, "dataset has no language tasks"
    total_rows = 0
    for episode in episodes:
        ep = int(episode["episode_index"])
        chunk = ep // int(info["chunks_size"])
        parquet = dataset / info["data_path"].format(episode_chunk=chunk, episode_index=ep)
        if not parquet.is_file():
            raise FileNotFoundError(parquet)
        frame = pd.read_parquet(parquet)
        expected_columns = {
            "observation.state", "action", "timestamp", "frame_index",
            "episode_index", "index", "task_index", "next.done",
        }
        assert expected_columns.issubset(frame.columns), frame.columns.tolist()
        assert len(frame) == int(episode["length"]), (parquet, len(frame), episode["length"])
        assert all(len(value) == EXPECTED_DIM for value in frame["observation.state"])
        assert all(len(value) == EXPECTED_DIM for value in frame["action"])
        assert bool(frame["next.done"].iloc[-1])
        total_rows += len(frame)
        for view in EXPECTED_VIEWS:
            video_key = f"observation.images.{view}"
            video = dataset / info["video_path"].format(
                episode_chunk=chunk,
                episode_index=ep,
                video_key=video_key,
            )
            if not video.is_file():
                raise FileNotFoundError(video)

    assert total_rows == int(info["total_frames"]), (total_rows, info["total_frames"])
    print(
        f"[layout] OK version=v2.1 episodes={len(episodes)} frames={total_rows} "
        f"fps={EXPECTED_FPS} state/action={EXPECTED_DIM}/{EXPECTED_DIM} views={len(EXPECTED_VIEWS)}"
    )
    return info, episodes


def validate_starvla_sample(dataset: Path) -> None:
    from omegaconf import OmegaConf

    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    config_path = Path(__file__).with_name("starvla_gr00t_n1d7_pipette_tube_press.yaml")
    cfg = OmegaConf.load(config_path).datasets.vla_data
    cfg.data_root_dir = str(dataset.parent)
    cfg.num_workers = 0
    mixture = get_vla_dataset(data_cfg=cfg)
    sample = mixture[0]

    action = sample["action"]
    state = sample["state"]
    images = sample["image"]
    assert tuple(action.shape) == (30, EXPECTED_DIM), action.shape
    assert tuple(state.shape) == (1, EXPECTED_DIM), state.shape
    assert len(images) == len(EXPECTED_VIEWS), len(images)
    print(
        f"[starvla] OK state={tuple(state.shape)} action={tuple(action.shape)} "
        f"views={len(images)} instruction={sample['lang']!r}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/coc/flash12/jwang3617/robot/RFT/vega_wuji_isaac6/outputs/"
            "pipette_tube_press_lerobot"
        ),
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Validate files and Parquet without decoding a transformed StarVLA sample.",
    )
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    validate_layout(dataset)
    if not args.metadata_only:
        validate_starvla_sample(dataset)


if __name__ == "__main__":
    main()
