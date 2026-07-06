#!/usr/bin/env python3
"""Reverse-engineer the `observation.task_info` (46-D) schema for task 0.

We load one HF parquet + its per-instance scene JSON, find the radio's
ground-truth initial world pose in the scene file, then scan `task_info[0]`
for the contiguous 7-D block whose `(x, y, z)` matches that ground-truth
position to within 1 cm. Likewise for any other object pose blocks we want
to label.

Output is a small JSON dump under
  `behavior1k_mp/ik/r1pro_constants.json`
that downstream code reads to slice `observation.task_info` correctly.

Run-once:
    conda activate xvla
    python scripts/inspect_task_info.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DEFAULT_DATA_DIR = Path("/shared_work/DATASETS/behavior-1k-embodiedAI-rollouts/data/task-0000")
DEFAULT_ANN_DIR = Path("/shared_work/DATASETS/behavior-1k-embodiedAI-rollouts/annotations/task-0000")
DEFAULT_SCENE_INSTANCE_DIR = Path(
    "/shared_work/BEHAVIOR-1K/datasets/2025-challenge-task-instances/scenes/"
    "house_double_floor_lower/json/house_double_floor_lower_task_turning_on_radio_instances"
)
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "behavior1k_mp/ik/r1pro_constants.json"


def find_radio_world_pose_from_scene_json(scene_json_path: Path):
    """Return (radio_pos_xyz[3], radio_quat_xyzw[4], robot_pos_xyz[3]) from scene state."""
    with scene_json_path.open() as f:
        scene = json.load(f)
    rk = "radio_receiver.n.01_1"
    if rk not in scene:
        return None, None, None
    root = scene[rk].get("root_link", {})
    radio_pos = np.asarray(root.get("pos", []), dtype=np.float64)
    radio_quat = np.asarray(root.get("ori", []), dtype=np.float64)
    robot_pos = None
    if "robot_poses" in scene and "R1Pro" in scene["robot_poses"]:
        robot_pos = np.asarray(scene["robot_poses"]["R1Pro"][0]["position"], dtype=np.float64)
    print(f"  scene-json radio.pos = {radio_pos.tolist()}")
    print(f"  scene-json radio.ori = {radio_quat.tolist()}  (|q|={float(np.linalg.norm(radio_quat)):.4f})")
    print(f"  scene-json robot.pos = {robot_pos.tolist() if robot_pos is not None else None}")
    return radio_pos, radio_quat, robot_pos


def scan_task_info_for_xyz(ti: np.ndarray, target_xyz: np.ndarray, tol: float = 0.01):
    """Try every starting offset; report any window whose first 3 values match target."""
    hits = []
    for off in range(len(ti) - 2):
        d = np.abs(np.asarray(ti[off:off + 3], dtype=np.float64) - target_xyz)
        if d.max() < tol:
            hits.append(off)
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ann_dir", type=Path, default=DEFAULT_ANN_DIR)
    parser.add_argument("--scene_instance_dir", type=Path, default=DEFAULT_SCENE_INSTANCE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--episode_index", type=int, default=10,
                        help="Which HF demo to probe (default: episode 10, instance 1).")
    args = parser.parse_args()

    parquet_path = args.data_dir / f"episode_{args.episode_index:08d}.parquet"
    print(f"Loading {parquet_path}")
    table = pq.read_table(parquet_path,
                          columns=["observation.task_info", "observation.state"])
    ti_full = np.stack([np.asarray(x) for x in table["observation.task_info"].to_pylist()])
    print(f"  observation.task_info: shape={ti_full.shape}, dtype={ti_full.dtype}")

    # The episode_index → instance ID mapping for HF is episode_index // 10
    # (stride-10 numbering matches challenge instance IDs 1..300).
    instance_id = args.episode_index // 10
    print(f"  inferred instance_id = {instance_id}")

    scene_json = (
        args.scene_instance_dir
        / f"house_double_floor_lower_task_turning_on_radio_0_{instance_id}_template-tro_state.json"
    )
    if not scene_json.exists():
        # Fall back to instance_id 0 if exact match missing
        scene_json = (
            args.scene_instance_dir
            / f"house_double_floor_lower_task_turning_on_radio_0_0_template-tro_state.json"
        )
        print(f"  WARN: exact instance JSON not found, falling back to {scene_json.name}")
    print(f"Scene file: {scene_json}")

    radio_pos, radio_quat, robot_pos = find_radio_world_pose_from_scene_json(scene_json)
    if radio_pos is None or len(radio_pos) != 3:
        print("ERROR: couldn't locate the radio's world xyz in the scene JSON.")
        return

    ti_t0 = ti_full[0]
    print(f"\ntask_info[0] full (round 3): {np.asarray(ti_t0, dtype=np.float64).round(3).tolist()}")

    # Find xyz matches for radio + robot
    print(f"\nScanning for RADIO xyz {radio_pos.round(3).tolist()} (tol=1cm)...")
    radio_hits = scan_task_info_for_xyz(ti_t0, radio_pos, tol=0.01)
    print(f"  hits at offsets: {radio_hits}")

    if robot_pos is not None:
        print(f"\nScanning for ROBOT xyz {robot_pos.round(3).tolist()} (tol=1cm)...")
        robot_hits = scan_task_info_for_xyz(ti_t0, robot_pos, tol=0.01)
        print(f"  hits at offsets: {robot_hits}")
    else:
        robot_hits = []

    # For each radio xyz hit, inspect the next 4 values and check if they match
    # the scene-JSON quat (or its sign-flipped twin — quats are double-cover).
    print(f"\nFor each radio-xyz hit, check next 4 values vs scene quat "
          f"{radio_quat.round(3).tolist()}:")
    best_radio_off = None
    best_diff = float("inf")
    for off in radio_hits:
        if off + 7 > len(ti_t0):
            continue
        block = np.asarray(ti_t0[off: off + 7], dtype=np.float64)
        cand_quat = block[3:7]
        diff = min(
            float(np.linalg.norm(cand_quat - radio_quat)),
            float(np.linalg.norm(cand_quat + radio_quat)),     # double-cover sign flip
        )
        print(f"  off={off:2d}  xyz={block[:3].round(3).tolist()}  "
              f"next4={cand_quat.round(3).tolist()}  diff_to_scene_quat={diff:.3f}")
        if diff < best_diff:
            best_diff = diff
            best_radio_off = off

    print(f"\nBest radio xyz+quat offset = {best_radio_off}  (quat diff = {best_diff:.3f})")

    # Do the same for the robot
    best_robot_off = None
    if robot_hits and robot_pos is not None:
        # We don't have robot quat from scene JSON directly broken out here,
        # but report the next 4 for inspection.
        print("\nRobot xyz hits, with the next 4 values shown:")
        for off in robot_hits:
            if off + 7 > len(ti_t0):
                continue
            block = np.asarray(ti_t0[off: off + 7], dtype=np.float64)
            print(f"  off={off:2d}  xyz={block[:3].round(3).tolist()}  "
                  f"next4={block[3:7].round(3).tolist()}")
        best_robot_off = robot_hits[0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "RADIO_POSE_SLICE": [int(best_radio_off), int(best_radio_off + 7)] if best_radio_off is not None else None,
        "ROBOT_POSE_SLICE": [int(best_robot_off), int(best_robot_off + 7)] if best_robot_off is not None else None,
        "RADIO_POSE_SLICE_NOTES": (
            f"Discovered by matching task_info[0] against scene JSON {scene_json.name} for "
            f"episode {args.episode_index}. Slice format: [start, end) for a "
            f"7-D (x, y, z, qx, qy, qz, qw) block."
        ),
    }
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.out}:")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
