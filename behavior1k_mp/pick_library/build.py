"""Build the pick-approach trajectory library from the 200 HF demos.

For each HF demo:
  1. Find the "pick up from" segment frames from the per-episode annotation
     (skill_description == "pick up from").
  2. Detect the gripper-close frame within that segment — the first frame where
     either L-grip (action[14]) or R-grip (action[22]) goes from positive
     (open) to negative (close).
  3. Truncate the segment to the first 90% of frames between pick-start and
     gripper-close. That trajectory becomes the "approach" piece replayed at
     runtime (we deliberately stop before contact and leave the IK-based grasp
     phase to do the contact).
  4. Compute the LIBRARY KEY = radio's pose expressed in the robot-base frame
     at the pick-start frame:
       - radio pose (world): radio is static, so read from the per-instance
         scene JSON at /shared_work/BEHAVIOR-1K/.../*_template-tro_state.json
         which has exact pos + quat.
       - robot pose (world) at pick_start: position comes from
         observation.task_info[1:4]; yaw comes from atan2(task_info[9],
         task_info[6]). (Empirically verified in scripts/inspect_task_info.py
         against the scene JSON's R1Pro initial yaw.)
  5. Look up the episode's press-phase cluster label from
     `phase_detector/checkpoints/press_modes/episode_labels.json` — that's the
     `press_label` we propagate downstream.

Saved to:
  /shared_work/behavior1k-mp/behavior1k_mp/phase_detector/checkpoints/pick_library/library_pick.pkl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq

from ..utils.pose import (
    pose7_to_matrix, matrix_to_pose7, pose_in_frame, yaw_to_quat_xyzw,
)

# ─── defaults ───────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = Path("/shared_work/DATASETS/behavior-1k-embodiedAI-rollouts/data/task-0000")
DEFAULT_ANN_DIR = Path("/shared_work/DATASETS/behavior-1k-embodiedAI-rollouts/annotations/task-0000")
DEFAULT_SCENE_INSTANCE_DIR = Path(
    "/shared_work/BEHAVIOR-1K/datasets/2025-challenge-task-instances/scenes/"
    "house_double_floor_lower/json/house_double_floor_lower_task_turning_on_radio_instances"
)
DEFAULT_PRESS_LABELS = Path(
    "/shared_work/behavior1k-mp/behavior1k_mp/phase_detector/checkpoints/"
    "press_modes/episode_labels.json"
)
DEFAULT_OUT_PATH = Path(
    "/shared_work/behavior1k-mp/behavior1k_mp/phase_detector/checkpoints/"
    "pick_library/library_pick.pkl"
)

# task_info slicing (from scripts/inspect_task_info.py)
TI_ROBOT_POS_SLICE = slice(1, 4)
TI_ROBOT_COS_YAW = 6
TI_ROBOT_SIN_YAW = 9
TI_RADIO_POS_SLICE = slice(11, 14)

# action layout — gripper channels (open ≈ +1, close ≈ -1)
ACTION_L_GRIP = 14
ACTION_R_GRIP = 22

# replay truncation: keep the first f * (gripper_close - pick_start) frames
TRUNCATE_FRAC = 0.85


def _find_pick_segment(ann_json: dict) -> tuple[int, int] | None:
    for seg in ann_json.get("skill_annotation", []):
        descs = seg.get("skill_description") or []
        if descs and descs[0] == "pick up from":
            fd = seg["frame_duration"]
            return int(fd[0]), int(fd[1])
    return None


def _find_first_gripper_close(action: np.ndarray, lo: int, hi: int) -> int | None:
    """Find first frame in [lo, hi) where either gripper crosses from open to close.

    "Open" means the previous value was >= 0; "close" means current is < 0 with
    the immediate prior frame being >= 0. We accept either L or R gripper closing.
    """
    if hi <= lo + 1:
        return None
    for i in range(lo + 1, hi):
        for ch in (ACTION_L_GRIP, ACTION_R_GRIP):
            if action[i, ch] < 0.0 and action[i - 1, ch] >= 0.0:
                return i
    return None


def _radio_pose_world_from_scene(scene_path: Path) -> np.ndarray | None:
    """Return 7-D radio pose [x,y,z, qx,qy,qz,qw] in world frame."""
    if not scene_path.exists():
        return None
    with scene_path.open() as f:
        scene = json.load(f)
    rk = "radio_receiver.n.01_1"
    if rk not in scene:
        return None
    root = scene[rk].get("root_link", {})
    pos = np.asarray(root.get("pos", []), dtype=np.float64)
    quat = np.asarray(root.get("ori", []), dtype=np.float64)  # xyzw per BDDL convention
    if pos.size != 3 or quat.size != 4:
        return None
    return np.concatenate([pos, quat])


def _robot_pose_world_at_frame(task_info_row: np.ndarray) -> np.ndarray:
    """Build robot's 7-D world pose from one row of observation.task_info."""
    pos = np.asarray(task_info_row[TI_ROBOT_POS_SLICE], dtype=np.float64).reshape(-1)
    cos_y = float(task_info_row[TI_ROBOT_COS_YAW])
    sin_y = float(task_info_row[TI_ROBOT_SIN_YAW])
    yaw = float(np.arctan2(sin_y, cos_y))
    return np.concatenate([pos, yaw_to_quat_xyzw(yaw)])


def _build_one_entry(episode_index: int,
                     parquet_path: Path,
                     ann_path: Path,
                     scene_instance_dir: Path,
                     press_label: str) -> dict | None:
    if not parquet_path.exists() or not ann_path.exists():
        return None
    with ann_path.open() as f:
        ann = json.load(f)
    seg = _find_pick_segment(ann)
    if seg is None:
        return None
    pick_start, pick_end = seg

    table = pq.read_table(parquet_path,
                          columns=["observation.task_info", "action", "observation.state"])
    task_info = np.stack([np.asarray(r) for r in table["observation.task_info"].to_pylist()]).astype(np.float64)
    action = np.stack([np.asarray(r) for r in table["action"].to_pylist()]).astype(np.float32)
    state = np.stack([np.asarray(r) for r in table["observation.state"].to_pylist()]).astype(np.float32)

    pick_end = min(pick_end, action.shape[0])
    pick_start = max(0, pick_start)
    if pick_end <= pick_start:
        return None

    # Truncate: stop ~10% before whichever gripper closes first
    f_close = _find_first_gripper_close(action, pick_start, pick_end)
    if f_close is None:
        # No gripper close detected — fall back to entire pick window
        f_end_trunc = pick_end
    else:
        f_end_trunc = pick_start + int(round(TRUNCATE_FRAC * (f_close - pick_start)))
    f_end_trunc = max(pick_start + 1, min(f_end_trunc, pick_end))

    # Library key — radio pose in robot base frame at pick-start
    instance_id = episode_index // 10
    scene_path = (scene_instance_dir
                  / f"house_double_floor_lower_task_turning_on_radio_0_{instance_id}_template-tro_state.json")
    radio_world = _radio_pose_world_from_scene(scene_path)
    if radio_world is None:
        return None
    robot_world = _robot_pose_world_at_frame(task_info[pick_start])
    radio_in_robot = pose_in_frame(radio_world, robot_world)   # 7-D

    return {
        "episode_index": int(episode_index),
        "key_7d": radio_in_robot.astype(np.float64),
        "action_traj": action[pick_start:f_end_trunc].copy(),      # [T, 23]
        "state_traj":  state[pick_start:f_end_trunc].copy(),       # [T, 256] (kept for debug)
        "press_label": press_label,
        "meta": {
            "pick_start": int(pick_start),
            "pick_end": int(pick_end),
            "gripper_close_frame": int(f_close) if f_close is not None else None,
            "trunc_end": int(f_end_trunc),
            "scene_path": str(scene_path),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ann_dir", type=Path, default=DEFAULT_ANN_DIR)
    parser.add_argument("--scene_instance_dir", type=Path, default=DEFAULT_SCENE_INSTANCE_DIR)
    parser.add_argument("--press_labels", type=Path, default=DEFAULT_PRESS_LABELS)
    parser.add_argument("--out_path", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load press cluster labels {episode_index_str: 0|1}
    with args.press_labels.open() as f:
        press_blob = json.load(f)
    press_by_ep_raw = press_blob.get("by_episode", press_blob)
    cluster_to_letter = {0: "A", 1: "B"}

    ann_files = sorted(args.ann_dir.glob("episode_*.json"))
    print(f"Scanning {len(ann_files)} HF demo annotations...")

    library = []
    failed = []
    label_counts = Counter()
    for i, ann_path in enumerate(ann_files):
        ep_idx = int(ann_path.stem.split("_")[1])
        parquet_path = args.data_dir / f"episode_{ep_idx:08d}.parquet"
        cluster_id = press_by_ep_raw.get(str(ep_idx))
        if cluster_id is None:
            failed.append((ep_idx, "no press label"))
            continue
        press_label = cluster_to_letter[int(cluster_id)]
        try:
            entry = _build_one_entry(ep_idx, parquet_path, ann_path,
                                     args.scene_instance_dir, press_label)
        except Exception as e:
            entry = None
            failed.append((ep_idx, str(e)))
            continue
        if entry is None:
            failed.append((ep_idx, "build_one_entry returned None"))
            continue
        library.append(entry)
        label_counts[press_label] += 1
        if (i + 1) % 50 == 0:
            print(f"  built {i+1}/{len(ann_files)}: ep={ep_idx:5d}  "
                  f"T_traj={entry['action_traj'].shape[0]:4d}  press={press_label}")

    print(f"\n=== Build summary ===")
    print(f"  entries built: {len(library)}")
    print(f"  failed:        {len(failed)}")
    if failed[:5]:
        print(f"  first failures: {failed[:5]}")
    print(f"  press label counts: A={label_counts['A']}, B={label_counts['B']}")

    # Summary stats on traj lengths and key spread
    if library:
        lens = np.array([e["action_traj"].shape[0] for e in library])
        keys = np.stack([e["key_7d"] for e in library])
        print(f"  trajectory lengths: min={lens.min()}  median={int(np.median(lens))}  max={lens.max()}")
        print(f"  key xyz mean = {keys[:, :3].mean(axis=0).round(3).tolist()}, "
              f"std = {keys[:, :3].std(axis=0).round(3).tolist()}")

    joblib.dump(library, args.out_path, compress=3)
    size_mb = args.out_path.stat().st_size / 1024**2
    print(f"\nSaved {args.out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
