"""Abstract base class for a single task phase ("action").

Each `Action` is a self-contained controller that:
  * drives the robot through one phase via `forward(obs) -> 23-D tensor`
  * reports when its exit condition is met via `is_done(obs, phase_detector) -> bool`
  * declares which executor it uses (X-VLA policy, motion planner, or no-op)

The exit condition defaults to **`pca_phase_shift AND _geometric_done(obs)`** —
phases stay active until the PCA phase detector has reported a *later* phase for
`K` consecutive frames AND a geometric/state check passes. The geometric check is
a stub (`return True`) on the base class; subclasses or the user fill it in later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import torch as th


class Executor(str, Enum):
    POLICY = "policy"           # X-VLA via WebsocketPolicy
    MOTION_PLANNER = "mp"       # StarterSemanticActionPrimitives
    NOOP = "noop"


class Action(ABC):
    # Subclasses MUST set these
    name: str = "unnamed"
    executor: Executor = Executor.NOOP
    expected_phase_id: int = -1

    # Debounce — how many consecutive "phase advanced" frames before we accept the shift.
    pca_debounce_steps: int = 10

    def __init__(self, env, robot, target_obj, **kwargs):
        self.env = env
        self.robot = robot
        self.target_obj = target_obj
        self._consecutive_in_next_phase = 0
        self._last_action: th.Tensor | None = None
        # Most-recent PCA-predicted phase (set by `is_done`). Exposed so the
        # orchestrator's heartbeat can log it without re-running the detector.
        self._last_phase_pred: int = -1
        # Orchestrator-level mutable dict. Set by `Orchestrator.__init__` after
        # all actions are constructed. Actions may read/write to coordinate
        # cross-phase hints (e.g. PickUpApproachAction writes 'grasp_variant').
        self.shared_state: dict | None = None

    # ───────────────────────── lifecycle hooks ─────────────────────────
    def on_enter(self, obs) -> None:
        """Called once when this action becomes active."""

    def on_exit(self, obs) -> None:
        """Called once when this action becomes inactive."""

    def reset(self) -> None:
        self._consecutive_in_next_phase = 0
        self._last_action = None

    # ───────────────────────── per-step API ─────────────────────────
    @abstractmethod
    def forward(self, obs) -> th.Tensor:
        """Return the next 23-D action."""

    def is_done(self, obs, phase_detector) -> bool:
        """Default exit: PCA phase shifted past `expected_phase_id` (debounced)
        AND `_geometric_done(obs)`."""
        z_phase = phase_detector.predict_from_obs(obs, self._last_action)
        self._last_phase_pred = int(z_phase)
        if z_phase > self.expected_phase_id:
            self._consecutive_in_next_phase += 1
        else:
            self._consecutive_in_next_phase = 0
        pca_done = self._consecutive_in_next_phase >= self.pca_debounce_steps
        return pca_done and self._geometric_done(obs)

    # ───────────────────────── per-phase geometric check (stub) ─────────────────────────
    def _geometric_done(self, obs) -> bool:
        """Per-phase geometric / task-state check.

        Currently returns True for every phase — the user will fill this in later
        with phase-specific checks (e.g. distance-to-target for navigate,
        `is_grasping(radio)` for pick, `radio.toggled_on` for press).
        """
        # TODO: per-phase geometric checks go here in subclasses.
        return True
