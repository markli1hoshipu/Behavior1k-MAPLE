#!/usr/bin/env python3
"""Build a unified press library keyed by gripper-in-radio relative pose.

For every HF demo with a "press" annotation segment:
  1. Find the press-start frame `p` from annotations.
  2. Read the radio's WORLD pose from the per-instance scene JSON (static).
  3. Read the robot's WORLD pose at frame `p` from `observation.task_info`.
  4. Read the proprioceptive joint state at frame `p`, run pytorch_kinematics
     FK on the R1Pro URDF to compute the right-gripper pose in robot base frame.
  5. Compose: `gripper_in_radio = pose_in_frame(gripper_in_base, radio_in_base)`
  6. Save (episode_index, key_7d=gripper_in_radio, action_traj, state_traj).

At runtime, `PressReplayBase.on_enter` will compute the same key from the live
env (radio + robot world poses + current joints) and find the nearest entry
by 7-D L2 distance.

Output:
  /shared_work/behavior1k-mp/behavior1k_mp/tasks/turning_on_radio/checkpoints/press_modes/library_press.pkl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import pyarrow.parquet as pq
import torch as th
import pytorch_kinematics as pk

from behavior1k_mp.core.utils.obs import extract_state_23d
from behavior1k_mp.core.utils.pose import pose7_to_matrix, matrix_to_pose7, pose_in_frame, yaw_to_quat_xyzw

DEFAULT_DATA_DIR = Path("/shared_work/DATASETS/behavior-1k-embodiedAI-rollouts/data/task-0000")
DEFAULT_ANN_DIR = Path("/shared_work/DATASETS/behavior-1k-embodiedAI-rollouts/annotations/task-0000")
DEFAULT_RAW_DIR = Path("/shared_work/DATASETS/behavior-1k-rawdata/task-0000")
DEFAULT_SCENE_INSTANCE_DIR = Path(
    "/shared_work/BEHAVIOR-1K/datasets/2025-challenge-task-instances/scenes/"
    "house_double_floor_lower/json/house_double_floor_lower_task_turning_on_radio_instances"
)
DEFAULT_OUT_PATH = Path(
    "/shared_work/behavior1k-mp/behavior1k_mp/tasks/turning_on_radio/checkpoints/"
    "press_modes/library_press.pkl"
)
URDF_PATH = "/shared_work/behavior1k-mp/third_party/r1pro.urdf"

# task_info slicing (same as pick_library/build.py)
TI_ROBOT_POS_SLICE = slice(1, 4)
TI_ROBOT_COS_YAW = 6
TI_ROBOT_SIN_YAW = 9

# Joint chain for FK
_DESIRED = [
    "torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4",
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3",
    "right_arm_joint4", "right_arm_joint5", "right_arm_joint6", "right_arm_joint7",
]


def _build_chain():
    with open(URDF_PATH, "rb") as f:
        chain = pk.build_serial_chain_from_urdf(f.read(), end_link_name="right_gripper_link")
    return chain.to(dtype=th.float64)


def _fk_gripper_in_base(chain, state256: np.ndarray) -> np.ndarray:
    """Compute right_gripper_link pose (7-D xyz + xyzw quat) in robot base
    frame from a single 256-D proprio frame."""
    s23 = extract_state_23d(state256)
    q11 = np.concatenate([s23[3:7], s23[15:22]]).astype(np.float64)
    joint_names_in_chain = list(chain.get_joint_parameter_names())
    name_to_idx = {n: i for i, n in enumerate(_DESIRED)}
    q_ord = th.tensor(
        [q11[name_to_idx[n]] for n in joint_names_in_chain],
        dtype=th.float64,
    )
    T = chain.forward_kinematics(q_ord, end_only=False)["right_gripper_link"].get_matrix().squeeze(0).numpy()
    return matrix_to_pose7(T)


def _find_press_segment(ann_json: dict) -> tuple[int, int] | None:
    for seg in ann_json.get("skill_annotation", []):
        descs = seg.get("skill_description") or []
        if descs and descs[0] == "press":
            fd = seg["frame_duration"]
            r = fd[0] if isinstance(fd[0], list) else fd
            return int(r[0]), int(r[1])
    return None


def _radio_pose_world_from_scene(scene_path: Path) -> np.ndarray | None:
    if not scene_path.exists():
        return None
    with scene_path.open() as f:
        scene = json.load(f)
    rk = "radio_receiver.n.01_1"
    if rk not in scene:
        return None
    root = scene[rk].get("root_link", {})
    pos = np.asarray(root.get("pos", []), dtype=np.float64)
    quat = np.asarray(root.get("ori", []), dtype=np.float64)
    if pos.size != 3 or quat.size != 4:
        return None
    return np.concatenate([pos, quat])


def _bddl_toggle_frame(h5_path: Path) -> int | None:
    """Return the first frame at which the BDDL goal flips from unsatisfied to
    satisfied — for `turning_on_radio`, this is when `toggled_on(radio)` goes
    False→True. Read from the raw HDF5 (not in the simplified parquet).

    The signal: in the raw rollouts, `reward[t]` is 0 except at exactly the
    success and failure transitions (where it equals +1 / -1), and
    `terminated[t]` flips True at success. We prefer reward>0 (the explicit
    goal-satisfied step) and fall back to the first `terminated==True` frame.
    """
    if not h5_path.exists():
        return None
    with h5py.File(h5_path, "r") as f:
        rew = f["data/demo_0/reward"][:]
        term = f["data/demo_0/terminated"][:]
    pos = np.where(rew > 0)[0]
    if len(pos):
        return int(pos[0])
    t = np.where(term)[0]
    return int(t[0]) if len(t) else None


def _robot_pose_world_at_frame(task_info_row: np.ndarray) -> np.ndarray:
    pos = np.asarray(task_info_row[TI_ROBOT_POS_SLICE], dtype=np.float64).reshape(-1)
    cos_y = float(task_info_row[TI_ROBOT_COS_YAW])
    sin_y = float(task_info_row[TI_ROBOT_SIN_YAW])
    yaw = float(np.arctan2(sin_y, cos_y))
    return np.concatenate([pos, yaw_to_quat_xyzw(yaw)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ann_dir", type=Path, default=DEFAULT_ANN_DIR)
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR,
                        help="Directory with raw HDF5 episodes (used to read reward/terminated for BDDL toggle).")
    parser.add_argument("--scene_instance_dir", type=Path, default=DEFAULT_SCENE_INSTANCE_DIR)
    parser.add_argument("--out_path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--no_truncate", action="store_true",
                        help="Disable BDDL-toggle truncation (keep full press skill window).")
    args = parser.parse_args()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    chain = _build_chain()
    ann_files = sorted(args.ann_dir.glob("episode_*.json"))
    print(f"Building press lookup library from {len(ann_files)} HF annotations...")

    entries: list[dict] = []
    failed = 0
    for i, ann_path in enumerate(ann_files):
        ep_idx = int(ann_path.stem.split("_")[1])
        parquet_path = args.data_dir / f"episode_{ep_idx:08d}.parquet"
        if not parquet_path.exists():
            failed += 1
            continue
        with ann_path.open() as f:
            ann = json.load(f)
        seg = _find_press_segment(ann)
        if seg is None:
            failed += 1
            continue
        press_start, press_end = seg

        table = pq.read_table(
            parquet_path,
            columns=["observation.task_info", "action", "observation.state"],
        )
        task_info = np.stack([np.asarray(r) for r in table["observation.task_info"].to_pylist()]).astype(np.float64)
        action = np.stack([np.asarray(r) for r in table["action"].to_pylist()]).astype(np.float32)
        state = np.stack([np.asarray(r) for r in table["observation.state"].to_pylist()]).astype(np.float32)
        if press_end > action.shape[0]:
            press_end = action.shape[0]
        if press_end <= press_start:
            failed += 1
            continue

        # gripper-in-base at press_start
        gripper_in_base = _fk_gripper_in_base(chain, state[press_start])
        # radio-world (static) from per-instance scene JSON
        instance_id = ep_idx // 10
        scene_path = (args.scene_instance_dir
                      / f"house_double_floor_lower_task_turning_on_radio_0_{instance_id}_template-tro_state.json")
        radio_world = _radio_pose_world_from_scene(scene_path)
        if radio_world is None:
            failed += 1
            continue
        # robot-world at press_start
        robot_world = _robot_pose_world_at_frame(task_info[press_start])
        # radio in robot base frame
        radio_in_base = pose_in_frame(radio_world, robot_world)
        # gripper in radio's local frame  ← THE LOOKUP KEY
        key_7d = pose_in_frame(gripper_in_base, radio_in_base)

        # Truncate at the BDDL toggle moment: the first frame where
        # `toggled_on(radio)` flips True (read from the raw HDF5's
        # reward/terminated). The replay ends exactly when the radio turns
        # on; everything after that is post-success hold and is left to the
        # wrap-up policy.
        if args.no_truncate:
            truncated_end = press_end
            toggle_frame = None
        else:
            h5_path = args.raw_dir / f"episode_{ep_idx:08d}.hdf5"
            toggle_frame = _bddl_toggle_frame(h5_path)
            if toggle_frame is None or not (press_start <= toggle_frame < press_end):
                # No raw HDF5, no toggle in this episode, or toggle outside
                # the press skill window → fall back to the full window.
                truncated_end = press_end
            else:
                # Inclusive of the toggle frame itself (where reward flips +1).
                truncated_end = toggle_frame + 1

        entries.append({
            "episode_index": ep_idx,
            "key_7d": key_7d.astype(np.float64),
            "action": action[press_start:truncated_end].copy(),    # [T, 23]
            "state":  state[press_start:truncated_end].copy(),     # [T, 256]
            "meta": {
                "press_start": press_start,
                "press_end": press_end,
                "truncated_end": truncated_end,
                "toggle_frame": toggle_frame,
                "scene_path": str(scene_path),
            },
        })
        if (i + 1) % 50 == 0:
            print(f"  built {i+1}/{len(ann_files)}: ep={ep_idx} "
                  f"T_full={press_end-press_start} T_trunc={truncated_end-press_start}")

    print(f"\n=== Build summary ===")
    print(f"  entries built: {len(entries)}")
    print(f"  failed:        {failed}")
    if entries:
        traj_lens = [len(e["action"]) for e in entries]
        keys = np.stack([e["key_7d"] for e in entries])
        print(f"  trajectory lengths: min={min(traj_lens)}  median={int(np.median(traj_lens))}  max={max(traj_lens)}")
        print(f"  key xyz mean = {keys[:, :3].mean(axis=0).round(3).tolist()}, std = {keys[:, :3].std(axis=0).round(3).tolist()}")
        print(f"  key quat mean = {keys[:, 3:].mean(axis=0).round(3).tolist()}, std = {keys[:, 3:].std(axis=0).round(3).tolist()}")
    joblib.dump(entries, args.out_path, compress=3)
    print(f"\nSaved {args.out_path} ({args.out_path.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
