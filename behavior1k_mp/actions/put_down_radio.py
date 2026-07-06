"""Phase 3 — put down the radio.

`turning_on_radio`'s BDDL goal is `(toggled_on radio_receiver.n.01_1)` — no
placement required to satisfy success. But emitting `th.zeros(23)` right
after the press phase causes a violent step on every commanded dim
(velocities + joint positions all snap to 0), so instead we **hold the
current pose** for `HOLD_FRAMES` to let physics settle, then exit.

Implementation: at `on_enter`, sample the current proprio, replicate it as
absolute commands for the trunk + both arms + grippers, force base velocity
to 0 (don't drive away), and replay that constant 23-D action for
`HOLD_FRAMES` frames.
"""
from __future__ import annotations

import logging

import numpy as np
import torch as th

from .base import Action, Executor
from ..utils.obs import extract_state_23d

logger = logging.getLogger(__name__)


class PutDownRadioAction(Action):
    name = "put_down_radio"
    executor = Executor.NOOP
    expected_phase_id = 3

    HOLD_FRAMES: int = 30   # 1 s at 30 Hz

    def __init__(self, env, robot, target_obj, **kwargs):
        super().__init__(env, robot, target_obj, **kwargs)
        self._traj: np.ndarray | None = None
        self._t = 0

    def on_enter(self, obs) -> None:
        # Prefer the orchestrator's last commanded action over state23 — the
        # state lags the command due to PD controller dynamics, so holding the
        # last command lets the controller finish converging.
        last_cmd = None
        if self.shared_state is not None:
            last_cmd = self.shared_state.get("_last_emitted_action")
        if last_cmd is not None:
            base = last_cmd.detach().cpu().numpy().astype(np.float32).reshape(-1) \
                if hasattr(last_cmd, "detach") else np.asarray(last_cmd, dtype=np.float32).reshape(-1)
            source = "last commanded action"
        else:
            raw_proprio = obs.get("robot_r1::proprio")
            if raw_proprio is None:
                raise KeyError("obs missing 'robot_r1::proprio'")
            base = extract_state_23d(raw_proprio).astype(np.float32)
            source = "state23 (fallback)"
        hold = np.tile(base, (self.HOLD_FRAMES, 1))
        hold[:, 0:3] = 0.0   # zero base velocity command — don't drive
        self._traj = hold.astype(np.float32)
        self._t = 0
        logger.info("PutDownRadio.on_enter: holding %s for %d frames",
                    source, self.HOLD_FRAMES)

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
