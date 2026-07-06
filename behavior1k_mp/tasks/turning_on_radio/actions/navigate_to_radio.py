"""Phase 0 — navigate to the radio.

Driven by the X-VLA policy (`WebsocketPolicy`). The exit condition is the
default base-class behavior: PCA phase detector reports phase != 0 for K
consecutive frames AND `_geometric_done(obs)` returns True (currently stubbed).
"""
from __future__ import annotations

import torch as th

from behavior1k_mp.core.action import Action, Executor


class NavigateToRadioAction(Action):
    name = "navigate_to_radio"
    executor = Executor.POLICY
    expected_phase_id = 0

    def __init__(self, env, robot, target_obj, *, policy, **kwargs):
        """
        Args:
            policy: an X-VLA policy instance (e.g. `WebsocketPolicy`) exposing
                    `forward(obs) -> th.Tensor[23]` and `reset()`.
        """
        super().__init__(env, robot, target_obj, **kwargs)
        self.policy = policy

    def on_enter(self, obs) -> None:
        # Make sure the upstream policy has a clean state for this episode.
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def forward(self, obs) -> th.Tensor:
        return self.policy.forward(obs)

    def _geometric_done(self, obs) -> bool:
        # TODO(user): when navigation completes geometrically, exit early.
        #   e.g.  base_xy = obs["robot_r1::proprio"][:2]
        #         target_xy = self.target_obj.get_position_orientation()[0][:2]
        #         return float((base_xy - target_xy).norm()) < 1.0
        return True
