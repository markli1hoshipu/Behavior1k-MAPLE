#!/usr/bin/env python3
"""Sanity-check the fitted phase detector against HF skill annotations.

For each HF demo episode that has an annotation JSON:
  1. Walk every frame, predict the phase via PCAPhaseDetector
  2. Look up the ground-truth phase from the skill_annotation segment for that frame
  3. Map skill_id -> phase_id and print per-phase confusion stats.

Skill-id to phase mapping (turning_on_radio, from
/shared_work/behavior1k-xvla/behavior1k_training/b1k_config.py):
   0  move_to        -> phase 0 (navigate)
   2  pick_up_from   -> phase 1 (pick)         (note: skill_id naming varies; see below)
   67 press          -> phase 2 (press)
   3  place_on       -> phase 3 (place)

We use the actual SKILL_DESCRIPTION from the annotation JSON ("move to",
"pick up from", "press", "place on") to be robust.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from behavior1k_mp.core.phase_detector.detector import PCAPhaseDetector
from behavior1k_mp.core.utils.obs import extract_state_23d

DESC_TO_PHASE = {
    "move to": 0,
    "pick up from": 1,
    "press": 2,
    "place on": 3,
}

DEFAULT_DATA_DIR = Path("/shared_work/DATASETS/behavior-1k-embodiedAI-rollouts/data/task-0000")
DEFAULT_ANN_DIR = Path("/shared_work/DATASETS/behavior-1k-embodiedAI-rollouts/annotations/task-0000")
DEFAULT_CKPT_DIR = Path("/shared_work/behavior1k-mp/behavior1k_mp/tasks/turning_on_radio/checkpoints")


def build_frame_labels(ann_json: dict, n_frames: int) -> np.ndarray:
    """For each frame index in [0, n_frames), assign a phase ID (or -1)."""
    labels = -np.ones(n_frames, dtype=np.int64)
    for seg in ann_json.get("skill_annotation", []):
        desc = (seg.get("skill_description") or ["?"])[0]
        if desc not in DESC_TO_PHASE:
            continue
        phase = DESC_TO_PHASE[desc]
        fd = seg["frame_duration"]
        ranges = fd if isinstance(fd[0], list) else [fd]
        for start, end in ranges:
            start = max(0, int(start))
            end = min(n_frames, int(end))
            labels[start:end] = phase
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ann_dir", type=Path, default=DEFAULT_ANN_DIR)
    parser.add_argument("--ckpt_dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--max_episodes", type=int, default=20)
    args = parser.parse_args()

    detector = PCAPhaseDetector(args.ckpt_dir)

    counts = Counter()        # (gt_phase, pred_phase) -> n
    per_phase_tot = Counter()
    eps_evaluated = 0

    ann_files = sorted(args.ann_dir.glob("episode_*.json"))[: args.max_episodes]
    for ann_path in ann_files:
        ep_idx = int(ann_path.stem.split("_")[1])
        parquet_path = args.data_dir / f"episode_{ep_idx:08d}.parquet"
        if not parquet_path.exists():
            continue

        with ann_path.open() as f:
            ann = json.load(f)

        table = pq.read_table(parquet_path, columns=["observation.state", "action"])
        state = np.stack([np.asarray(s) for s in table["observation.state"].to_pylist()])
        action = np.stack([np.asarray(a) for a in table["action"].to_pylist()])
        T = state.shape[0]

        gt = build_frame_labels(ann, T)
        state_23 = extract_state_23d(state)
        feats = np.concatenate([state_23, action], axis=1).astype(np.float32)

        for i in range(T):
            if gt[i] < 0:
                continue
            pred = detector.predict_feature(feats[i])
            counts[(int(gt[i]), pred)] += 1
            per_phase_tot[int(gt[i])] += 1

        eps_evaluated += 1

    print(f"Evaluated {eps_evaluated} episodes")
    print("\nConfusion (rows = GT phase, cols = predicted phase):")
    print(f"     pred:    0       1       2       3")
    correct = 0
    total = 0
    for gt_p in range(4):
        row = [counts.get((gt_p, p), 0) for p in range(4)]
        total_gt = per_phase_tot[gt_p]
        if total_gt == 0:
            print(f"  GT {gt_p}: <no frames>")
            continue
        pcts = [f"{100 * c / total_gt:5.1f}%" for c in row]
        correct += counts.get((gt_p, gt_p), 0)
        total += total_gt
        print(f"  GT {gt_p}: {pcts[0]} {pcts[1]} {pcts[2]} {pcts[3]}   (n={total_gt})")
    if total > 0:
        print(f"\nOverall accuracy: {correct}/{total} = {100*correct/total:.1f}%")


if __name__ == "__main__":
    main()
