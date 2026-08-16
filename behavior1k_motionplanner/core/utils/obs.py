"""Observation-extraction utilities.

We mirror the 23-D state slice used by X-VLA's `Behavior1KHandler`. The
canonical layout lives at:
  /shared_work/behavior1k-xvla/behavior1k_training/b1k_config.py

Order: base_qvel(3) | trunk_qpos(4) | arm_left(7) | grip_left(1) | arm_right(7) | grip_right(1)

Grippers are normalized from raw finger-joint sum [0, MAX_GRIPPER_WIDTH] to
[-1, 1] to match the action-space convention. The raw proprio carries two
finger joints per gripper (`(193, 195)` for left and `(232, 234)` for right);
we sum them and rescale.
"""
from __future__ import annotations

import numpy as np
import torch as th

# ── Proprioception indices: 256-D observation.state → 23-D state ───────────────
# Canonical values from b1k_config.py::PROPRIO_STATE_SLICES.
# Earlier versions of this file had wildly wrong indices (54-57, 43-47, etc.);
# every consumer of extract_state_23d was reading garbage.
_BASE_QVEL          = slice(253, 256)   # 3 dims
_TRUNK_QPOS         = slice(236, 240)   # 4 dims
_ARM_LEFT_QPOS      = slice(158, 165)   # 7 dims
_GRIPPER_LEFT_QPOS  = slice(193, 195)   # 2 dims (2 finger joints) → sum+normalize to 1
_ARM_RIGHT_QPOS     = slice(197, 204)   # 7 dims
_GRIPPER_RIGHT_QPOS = slice(232, 234)   # 2 dims → sum+normalize to 1

MAX_GRIPPER_WIDTH = 0.1   # raw finger-joint sum is in [0, 0.1]; normalize to [-1, +1]


def _normalize_gripper(raw_two: np.ndarray) -> np.ndarray:
    """Normalize the 2-D raw finger joint pair to a single [-1, +1] value.
    Matches X-VLA training: grip = 2 * (sum/MAX_GRIPPER_WIDTH) - 1."""
    return 2.0 * (raw_two.sum(axis=-1, keepdims=True) / MAX_GRIPPER_WIDTH) - 1.0


def extract_state_23d(state256: np.ndarray | th.Tensor) -> np.ndarray:
    """Slice the 256-D proprio into the 23-D R1Pro layout used by X-VLA training.

    Supports both 1-D (single frame) and 2-D (`[T, 256]`) inputs.
    """
    arr = state256.detach().cpu().numpy() if isinstance(state256, th.Tensor) else np.asarray(state256)
    base   = arr[..., _BASE_QVEL]                            # 3
    trunk  = arr[..., _TRUNK_QPOS]                           # 4
    l_arm  = arr[..., _ARM_LEFT_QPOS]                        # 7
    l_grip = _normalize_gripper(arr[..., _GRIPPER_LEFT_QPOS])   # 1
    r_arm  = arr[..., _ARM_RIGHT_QPOS]                       # 7
    r_grip = _normalize_gripper(arr[..., _GRIPPER_RIGHT_QPOS])  # 1
    return np.concatenate([base, trunk, l_arm, l_grip, r_arm, r_grip], axis=-1)


def obs_to_state_action_features(obs: dict, last_action) -> np.ndarray:
    """Build the 46-D (state_23 + action_23) feature vector the PCA was fit on.

    Accepts the raw obs dict the OmniGibson env passes to a policy's `forward()`.
    The 256-D proprio key is `"robot_r1::proprio"`; if not present, we look for
    `"observation.state"` (parquet-loaded frames).
    """
    if "robot_r1::proprio" in obs:
        raw = obs["robot_r1::proprio"]
    elif "observation.state" in obs:
        raw = obs["observation.state"]
    else:
        raise KeyError(
            "obs has neither 'robot_r1::proprio' nor 'observation.state'; got keys: "
            f"{list(obs.keys())[:10]}..."
        )
    state_23 = extract_state_23d(raw)

    if last_action is None:
        action_23 = np.zeros(23, dtype=np.float32)
    elif isinstance(last_action, th.Tensor):
        action_23 = last_action.detach().cpu().numpy().astype(np.float32).reshape(-1)
    else:
        action_23 = np.asarray(last_action, dtype=np.float32).reshape(-1)

    return np.concatenate([state_23.reshape(-1), action_23], axis=-1).astype(np.float32)
