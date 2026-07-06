"""Unified press-replay action.

At `on_enter` we do a **closed-form nearest-neighbor lookup** over the
200-demo press library, keyed by the gripper-in-radio relative pose at the
moment press starts:

  1. Compute `gripper_in_base` via pytorch_kinematics FK on the current
     `state23` (trunk + R_arm joints).
  2. Compute `radio_in_base = pose_in_frame(radio_world, robot_world)` from
     the live env.
  3. `key = pose_in_frame(gripper_in_base, radio_in_base)` — gripper expressed
     in the radio's local frame (7-D `[x, y, z, qx, qy, qz, qw]`).
  4. Look up the nearest of 200 entries in `library_press.pkl` under plain
     L2 distance over the 7-D key (no separate position/quat weighting; the
     180° flip between A and B grasps naturally shows up as opposite
     quaternions in the key, so A-grasps match A-source demos and B-grasps
     match B-source demos without any explicit label routing).
  5. Prepend a smoothing bridge from current state to the demo's frame 0,
     then replay the demo's recorded action trajectory (truncated at the
     BDDL `toggled_on` flip frame from the raw HDF5 reward signal).
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pytorch_kinematics as pk
import torch as th

from .base import Action, Executor
from ..ik import DEFAULT_URDF_PATH
from ..utils.action_bridge import linear_bridge_to
from ..utils.obs import extract_state_23d
from ..utils.pose import matrix_to_pose7, pose_in_frame

logger = logging.getLogger(__name__)

_DEFAULT_LIBRARY_PATH = (
    Path(__file__).resolve().parents[1]
    / "phase_detector"
    / "checkpoints"
    / "press_modes"
    / "library_press.pkl"
)

# Joint chain for FK (lazy-built, shared across calls)
_DESIRED = [
    "torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4",
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3",
    "right_arm_joint4", "right_arm_joint5", "right_arm_joint6", "right_arm_joint7",
]
_CHAIN: pk.SerialChain | None = None
_CHAIN_NAME_TO_IDX: dict[str, int] | None = None


def _get_chain():
    global _CHAIN, _CHAIN_NAME_TO_IDX
    if _CHAIN is None:
        with open(DEFAULT_URDF_PATH, "rb") as f:
            chain = pk.build_serial_chain_from_urdf(f.read(), end_link_name="right_gripper_link")
        _CHAIN = chain.to(dtype=th.float64)
        _CHAIN_NAME_TO_IDX = {n: i for i, n in enumerate(_DESIRED)}
    return _CHAIN


def _fk_gripper_in_base(state256: np.ndarray) -> np.ndarray:
    """7-D pose of right_gripper_link in robot base frame from one proprio frame."""
    chain = _get_chain()
    s23 = extract_state_23d(state256)
    q11 = np.concatenate([s23[3:7], s23[15:22]]).astype(np.float64)
    joint_names_in_chain = list(chain.get_joint_parameter_names())
    q_ord = th.tensor(
        [q11[_CHAIN_NAME_TO_IDX[n]] for n in joint_names_in_chain],
        dtype=th.float64,
    )
    T = chain.forward_kinematics(q_ord, end_only=False)["right_gripper_link"].get_matrix().squeeze(0).numpy()
    return matrix_to_pose7(T)


def _nearest_press_entry(library: list[dict], query_key_7d: np.ndarray) -> tuple[dict, float]:
    """Pure L2 over the 7-D key (no separate position/quat weighting)."""
    qk = np.asarray(query_key_7d, dtype=np.float64).reshape(-1)
    dists = [float(np.linalg.norm(qk - e["key_7d"])) for e in library]
    idx = int(np.argmin(dists))
    return library[idx], float(dists[idx])


class PressReplayBase(Action):
    """Plays back a press trajectory selected by nearest-neighbor lookup on
    the gripper-in-radio relative pose. A/B variants are merged into one
    unified `library_press.pkl`; routing is implicit via the 7-D key."""

    name = "press_radio"
    expected_phase_id = 2
    executor = Executor.MOTION_PLANNER

    def __init__(
        self,
        env,
        robot,
        target_obj,
        *,
        library_path: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(env, robot, target_obj, **kwargs)
        self.library_path = Path(library_path) if library_path else _DEFAULT_LIBRARY_PATH
        self._library: list[dict] | None = None
        self._traj_actions: np.ndarray | None = None
        self._chosen_ep: int | None = None
        self._t = 0

    def _load_library(self) -> None:
        if self._library is None:
            if not self.library_path.exists():
                raise FileNotFoundError(
                    f"Unified press library not found at {self.library_path}. "
                    "Run `scripts/build_press_lookup_library.py` first."
                )
            self._library = joblib.load(self.library_path)
            if not self._library:
                raise RuntimeError(f"Press library {self.library_path} is empty.")

    def _compute_query_key(self, obs) -> np.ndarray:
        """gripper-in-radio relative pose at the start of press, in 7-D."""
        raw_proprio = obs.get("robot_r1::proprio")
        if raw_proprio is None:
            raise KeyError("obs missing 'robot_r1::proprio'")
        gripper_in_base = _fk_gripper_in_base(raw_proprio)
        # radio in robot base
        radio_pos, radio_quat = self.target_obj.get_position_orientation()
        robot_pos, robot_quat = self.robot.get_position_orientation()

        def _np1d(x):
            return (x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)).reshape(-1).astype(np.float64)
        radio_world = np.concatenate([_np1d(radio_pos)[:3], _np1d(radio_quat)[:4]])
        robot_world = np.concatenate([_np1d(robot_pos)[:3], _np1d(robot_quat)[:4]])
        radio_in_base = pose_in_frame(radio_world, robot_world)
        # gripper in radio's local frame
        return pose_in_frame(gripper_in_base, radio_in_base)

    def on_enter(self, obs) -> None:
        self._load_library()

        query_key = self._compute_query_key(obs)
        entry, dist = _nearest_press_entry(self._library, query_key)
        raw = entry["action"].astype(np.float32)  # [T, 23]
        self._chosen_ep = int(entry["episode_index"])

        # Bridge from current state to demo frame 0
        raw_proprio = obs.get("robot_r1::proprio")
        state23 = extract_state_23d(raw_proprio).astype(np.float32)
        bridge = linear_bridge_to(state23, raw[0])
        self._traj_actions = np.concatenate([bridge, raw], axis=0).astype(np.float32)
        self._t = 0
        logger.info(
            "%s.on_enter: matched episode %d by gripper-in-radio L2 (dist=%.4f, key=(%.3f,%.3f,%.3f, %.3f,%.3f,%.3f,%.3f)) "
            "[bridge_len=%d, traj_len=%d, total_len=%d]",
            self.name, self._chosen_ep, dist,
            query_key[0], query_key[1], query_key[2],
            query_key[3], query_key[4], query_key[5], query_key[6],
            len(bridge), len(raw), len(self._traj_actions),
        )

    def forward(self, obs) -> th.Tensor:
        if self._traj_actions is None or self._t >= len(self._traj_actions):
            return th.zeros(23, dtype=th.float32)
        action = th.from_numpy(self._traj_actions[self._t]).to(th.float32)
        self._t += 1
        return action

    def is_done(self, obs, phase_detector) -> bool:
        if self._traj_actions is None:
            return False
        return self._t >= len(self._traj_actions)

    def reset(self) -> None:
        super().reset()
        self._traj_actions = None
        self._chosen_ep = None
        self._t = 0
