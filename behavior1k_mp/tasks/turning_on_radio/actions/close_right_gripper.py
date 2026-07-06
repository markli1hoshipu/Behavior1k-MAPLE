"""Phase 1c — close the right gripper after the IK has positioned the wrist.

`PickUpGraspAction` (phase 1b) leaves the wrist at the grasp pose with a
short tail of close-gripper frames. This action makes the closure an
explicit, semantically-named phase: at `on_enter` we read the current
state, build a held-pose action where every dim equals the current state
EXCEPT `action[22]` (right gripper) which is forced to −1 (closed), and
replay that for `HOLD_FRAMES` frames.

Net effect:
  - Base velocity zeroed → robot stays put.
  - Trunk + L_arm + R_arm held at their current joint positions.
  - L_grip held at its current command.
  - R_grip commanded to −1 (closed) until the physics finishes squeezing.

Placement in the orchestration chain:
  navigate → approach → grasp → **close_right_gripper** → press → put_down
"""
from __future__ import annotations

import logging

import numpy as np
import torch as th

from behavior1k_mp.core.action import Action, Executor
from behavior1k_mp.core.utils.obs import extract_state_23d

logger = logging.getLogger(__name__)


class CloseRightGripperAction(Action):
    name = "close_right_gripper"
    executor = Executor.MOTION_PLANNER
    expected_phase_id = 1   # still part of "pick up" semantically

    HOLD_FRAMES: int = 30           # ~1 s at 30 Hz
    GRIP_CLOSED_VALUE: float = -1.0  # MultiFingerGripperController smooth-mode close

    def __init__(self, env, robot, target_obj, **kwargs):
        super().__init__(env, robot, target_obj, **kwargs)
        self._traj: np.ndarray | None = None
        self._t = 0

    def on_enter(self, obs) -> None:
        # Hold the current proprioceptive state (state23) for HOLD_FRAMES
        # frames, with base velocity zeroed and R_grip forced to −1 (close).
        # Earlier this used `shared_state['_last_emitted_action']` to hold the
        # IK target rather than the actually-reached joint position — but the
        # user is investigating whether the controller itself behaves
        # differently when we command the current state vs the IK target, so
        # we're back to state23 here for that experiment.
        raw_proprio = obs.get("robot_r1::proprio")
        if raw_proprio is None:
            raise KeyError("obs missing 'robot_r1::proprio'")
        state23 = extract_state_23d(raw_proprio).astype(np.float32)
        hold = np.tile(state23, (self.HOLD_FRAMES, 1))
        hold[:, 0:3] = 0.0                          # zero base velocity
        hold[:, 22] = self.GRIP_CLOSED_VALUE        # close right gripper
        self._traj = hold.astype(np.float32)
        self._t = 0
        logger.info(
            "CloseRightGripper.on_enter: holding pose [source=state23, R_grip=%+.1f] for %d frames "
            "(prev R_grip state=%+.3f)",
            self.GRIP_CLOSED_VALUE, self.HOLD_FRAMES, float(state23[22]),
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

    def reset(self) -> None:
        super().reset()
        self._traj = None
        self._t = 0
