"""Phase 1a — approach the radio by replaying the closest HF demo trajectory.

At `on_enter`:
  1. Compute the radio's pose in the robot-base frame from the live env
     (`env.task.object_scope[radio].get_position_orientation()` +
      `env.robot.get_position_orientation()`).
  2. Find the library entry whose key (radio-in-robot at the recorded
     pick-start frame) is closest under a quat-weighted L2.
  3. Cache the selected entry's truncated action trajectory and remember the
     entry's grasp-variant label (A or B).

`forward()` replays the recorded `action[t]` open-loop until the trajectory
is exhausted, then `is_done()` returns True so the orchestrator advances to
`PickUpGraspAction`.

`on_exit` writes the matched grasp variant into `shared_state['grasp_variant']`
so `PickUpGraspAction` flips the gripper 180° for B-grasps. The downstream
press phase doesn't use this hint — it routes A/B implicitly via gripper-in-
radio nearest-neighbor lookup over the unified press library.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch as th

from behavior1k_mp.core.action import Action, Executor
from behavior1k_mp.core.pick_library.lookup import load_library, nearest_entry
from behavior1k_mp.core.utils.action_bridge import linear_bridge_to, required_bridge_frames
from behavior1k_mp.core.utils.obs import extract_state_23d
from behavior1k_mp.core.utils.pose import pose_in_frame

logger = logging.getLogger(__name__)

_DEFAULT_LIBRARY_PATH = (
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "pick_library"
    / "library_pick.pkl"
)


class PickUpApproachAction(Action):
    name = "pick_up_radio_approach"
    expected_phase_id = 1
    executor = Executor.MOTION_PLANNER

    def __init__(
        self,
        env,
        robot,
        target_obj,
        *,
        library_path: str | Path | None = None,
        w_pos: float = 1.0,
        w_quat: float = 1.0,   # balanced (was 4.0): position and orientation weighted equally
        **kwargs,
    ):
        super().__init__(env, robot, target_obj, **kwargs)
        self.library_path = Path(library_path) if library_path else _DEFAULT_LIBRARY_PATH
        self.w_pos = w_pos
        self.w_quat = w_quat
        self._library: list[dict] | None = None
        self._traj: np.ndarray | None = None        # [T, 23]
        self._chosen_ep: int | None = None
        self._grasp_label: str | None = None
        self._t = 0

    # ───────────────────────── helpers ─────────────────────────
    def _load_library(self) -> None:
        if self._library is None:
            self._library = load_library(self.library_path)

    def _compute_query_key(self) -> np.ndarray:
        """Radio pose expressed in robot base frame (7-D)."""
        # Each accessor returns (pos[3], quat[4]) in scipy xyzw convention.
        radio_pos, radio_quat = self.target_obj.get_position_orientation()
        robot_pos, robot_quat = self.robot.get_position_orientation()
        radio_pos = np.asarray(radio_pos).reshape(-1)[:3]
        radio_quat = np.asarray(radio_quat).reshape(-1)[:4]
        robot_pos = np.asarray(robot_pos).reshape(-1)[:3]
        robot_quat = np.asarray(robot_quat).reshape(-1)[:4]
        radio_world = np.concatenate([radio_pos, radio_quat]).astype(np.float64)
        robot_world = np.concatenate([robot_pos, robot_quat]).astype(np.float64)
        return pose_in_frame(radio_world, robot_world)

    # ───────────────────────── lifecycle ─────────────────────────
    def on_enter(self, obs) -> None:
        self._load_library()
        query_key = self._compute_query_key()
        entry, dist, _ = nearest_entry(
            self._library, query_key, w_pos=self.w_pos, w_quat=self.w_quat
        )
        raw_traj = np.asarray(entry["action_traj"], dtype=np.float32)
        self._chosen_ep = int(entry["episode_index"])
        # Field is still named `press_label` in the pick library for back-compat;
        # it labels the grasp variant (A or B) inferred from the demo's press
        # cluster at build time. We surface it as `grasp_label` internally.
        self._grasp_label = str(entry["press_label"])

        # Bridge from current state to recorded frame 0 — eliminates the
        # nav-output → demo-frame-0 step jump. Length auto-scales with the
        # size of the jump (see utils/action_bridge.py).
        raw_proprio = obs.get("robot_r1::proprio")
        if raw_proprio is None:
            raise KeyError("obs missing 'robot_r1::proprio'")
        state23 = extract_state_23d(raw_proprio).astype(np.float32)
        bridge = linear_bridge_to(state23, raw_traj[0])
        self._traj = np.concatenate([bridge, raw_traj], axis=0).astype(np.float32)
        self._t = 0
        logger.info(
            "PickUpApproach.on_enter: matched episode %d (grasp=%s, dist=%.4f, "
            "bridge_len=%d, traj_len=%d, total_len=%d)",
            self._chosen_ep, self._grasp_label, dist,
            len(bridge), len(raw_traj), len(self._traj),
        )

    def forward(self, obs) -> th.Tensor:
        if self._traj is None or self._t >= len(self._traj):
            return th.zeros(23, dtype=th.float32)
        a = th.from_numpy(self._traj[self._t]).to(th.float32)
        self._t += 1
        return a

    def is_done(self, obs, phase_detector) -> bool:
        if self._traj is None:
            return False
        return self._t >= len(self._traj)

    def on_exit(self, obs) -> None:
        # Publish only the grasp variant (A/B). `PickUpGraspAction` reads it
        # to decide whether to flip the gripper 180° around the approach axis.
        # The press phase doesn't read this — it routes via key-based lookup.
        if self.shared_state is not None and self._grasp_label:
            self.shared_state["grasp_variant"] = self._grasp_label
            logger.info(
                "PickUpApproach.on_exit: set shared_state[grasp_variant=%s]",
                self._grasp_label,
            )

    def reset(self) -> None:
        super().reset()
        self._traj = None
        self._chosen_ep = None
        self._grasp_label = None
        self._t = 0
