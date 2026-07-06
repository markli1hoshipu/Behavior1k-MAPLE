"""Phase 6 — wrap-up via X-VLA policy.

After the orchestrated press phase finishes, hand control back to the X-VLA
`WebsocketPolicy` for any remaining task-completion steps:

  * If the press wasn't quite hard enough, X-VLA might tap the button again.
  * If the radio needs to be released or placed elsewhere, X-VLA continues.
  * If env already terminated with success (radio toggled_on), this phase
    never runs because the evaluator exits the step loop.

Runs for up to `MAX_FRAMES` (default 300 = 10 s @ 30 Hz). If the BDDL goal
fires during wrap-up, `env.step` returns terminated=True and the evaluator
loop exits before this action finishes.
"""
from __future__ import annotations

import logging

import torch as th

from .base import Action, Executor

logger = logging.getLogger(__name__)


class WrapUpPolicyAction(Action):
    name = "wrap_up_policy"
    executor = Executor.POLICY
    expected_phase_id = 3   # post-press / "place" semantically

    MAX_FRAMES: int = 300   # 10 s @ 30 Hz

    def __init__(self, env, robot, target_obj, *, policy, **kwargs):
        """
        Args:
            policy: an X-VLA policy (`WebsocketPolicy`) exposing
                    `forward(obs) -> th.Tensor[23]` and `reset()`.
        """
        super().__init__(env, robot, target_obj, **kwargs)
        self.policy = policy
        self._frames = 0

    def on_enter(self, obs) -> None:
        # Reset the policy's internal action-chunk buffer so it starts
        # fresh under the new (post-press) observation distribution.
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self._frames = 0
        logger.info("WrapUpPolicy.on_enter: handing control to X-VLA for up to %d frames",
                    self.MAX_FRAMES)

    def forward(self, obs) -> th.Tensor:
        action = self.policy.forward(obs)
        self._frames += 1
        if isinstance(action, th.Tensor):
            return action.to(th.float32)
        import numpy as np
        return th.tensor(np.asarray(action, dtype=np.float32))

    def is_done(self, obs, phase_detector) -> bool:
        if self._frames >= self.MAX_FRAMES:
            logger.info("WrapUpPolicy: emitted %d frames; phase done.", self._frames)
            return True
        return False

    def reset(self) -> None:
        super().reset()
        self._frames = 0
