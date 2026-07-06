"""State machine that walks through an ordered list of `Action`s.

Two extensions over the original orchestrator:

  1. ``shared_state``: a mutable dict accessible to every Action via
     ``self.shared_state``. Used to pass hints between phases (e.g.
     ``PickUpApproachAction`` writes ``shared_state['grasp_variant']``
     which ``PickUpGraspAction`` reads to flip the gripper 180° for B-grasps).

  2. **Lazy action entries**: any list element that's callable (but not an
     `Action` instance) is treated as a factory `f(shared_state) -> Action`.
     The factory is invoked the first time the orchestrator steps into that
     slot, after upstream actions have had a chance to populate
     ``shared_state``. (Currently no actions are registered as lazy — the
     unified press class supersedes the old A/B press factory.)

Per step, the orchestrator:
  1. Resolves the current entry (lazy → instance) if needed.
  2. Lazily fires `on_enter()` of the active action.
  3. Calls `forward(obs)` to get the 23-D action.
  4. Caches that action on the Action instance for `is_done()`'s PCA call.
  5. Asks `is_done(obs, phase_detector)`. If True: `on_exit()`, advance index,
     drop the resolved instance, clear `_entered`.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Union

import torch as th

from .action import Action
from .phase_detector.detector import PCAPhaseDetector

logger = logging.getLogger(__name__)

ActionFactory = Callable[[dict], Action]
ActionEntry = Union[Action, ActionFactory]


class Orchestrator:
    # Heartbeat: log once every N steps so we have liveness in long episodes
    # without flooding stderr (max-step caps run into the thousands).
    HEARTBEAT_EVERY: int = 50

    def __init__(
        self,
        actions: List[ActionEntry],
        phase_detector: PCAPhaseDetector,
        shared_state: dict | None = None,
    ):
        self.actions = list(actions)
        self.phase_detector = phase_detector
        self.shared_state = shared_state if shared_state is not None else {}
        # Propagate the shared_state ref to every already-built Action so they
        # can read/write it without explicit plumbing.
        for a in self.actions:
            if isinstance(a, Action):
                a.shared_state = self.shared_state
        self.idx = 0
        self._entered = False
        self._resolved: Action | None = None
        self._step = 0
        self._steps_in_action = 0
        # Most-recent action emitted by `forward`. Used to "hold pose" when the
        # orchestrator runs out of slots — emitting `zeros(23)` instead would
        # be interpreted as "command all joints to zero", which slews the arms
        # to the home pose and opens the grippers (releasing whatever was held).
        self._last_emitted_action: th.Tensor | None = None
        self._post_orchestration_log_done: bool = False
        # Records (action_name, enter_step, exit_step) for each slot as it
        # runs. Consumed by `HybridPolicy.build_skill_annotation` to write
        # HF-format `skill_annotation` JSON next to the saved episode.
        self._action_spans: list[tuple[str, int, int]] = []
        self._current_enter_step: int | None = None
        logger.info(
            "Orchestrator built: %d slots: %s",
            len(self.actions),
            ", ".join(
                a.name if isinstance(a, Action) else f"<lazy:{i}>"
                for i, a in enumerate(self.actions)
            ),
        )

    # ───────────────────────── helpers ─────────────────────────
    def _current(self) -> Action:
        """Return the currently active Action, resolving the factory if needed."""
        entry = self.actions[self.idx]
        if isinstance(entry, Action):
            return entry
        # It's a factory. Resolve once.
        if self._resolved is None:
            logger.info("[orch] resolving lazy action at slot %d", self.idx)
            self._resolved = entry(self.shared_state)
            if not isinstance(self._resolved, Action):
                raise TypeError(
                    f"Lazy entry at slot {self.idx} returned {type(self._resolved).__name__}; "
                    "must return an `Action`."
                )
            self._resolved.shared_state = self.shared_state
        return self._resolved

    # ───────────────────────── lifecycle ─────────────────────────
    def reset(self) -> None:
        for entry in self.actions:
            if isinstance(entry, Action):
                entry.reset()
        if self._resolved is not None:
            self._resolved.reset()
        self.shared_state.clear()
        self.idx = 0
        self._entered = False
        self._resolved = None
        self._step = 0
        self._steps_in_action = 0
        self._last_emitted_action = None
        self._post_orchestration_log_done = False
        self._action_spans = []
        self._current_enter_step = None
        logger.info("Orchestrator.reset (shared_state cleared, idx=0)")

    @property
    def current_action(self) -> Action | None:
        if 0 <= self.idx < len(self.actions):
            return self._current()
        return None

    @property
    def is_done(self) -> bool:
        """True once the orchestrator has stepped past the last action slot."""
        return self.idx >= len(self.actions)

    def action_spans(self, total_frames: int | None = None) -> list[tuple[str, int, int]]:
        """Return [(action_name, enter_step, exit_step), ...] for every slot
        that has run. If a slot was mid-execution when the episode ended
        (env terminated before `is_done`), its exit_step is clamped to
        `total_frames` (or the current step if not provided)."""
        spans = list(self._action_spans)
        if self._entered and self._current_enter_step is not None:
            a = self._current_action_name()
            end = total_frames if total_frames is not None else self._step + 1
            spans.append((a, self._current_enter_step, end))
        return spans

    def _current_action_name(self) -> str:
        if 0 <= self.idx < len(self.actions):
            entry = self.actions[self.idx]
            if isinstance(entry, Action):
                return entry.name
            if self._resolved is not None:
                return self._resolved.name
        return "unknown"

    def forward(self, obs) -> th.Tensor:
        if self.idx >= len(self.actions):
            # Past the last action — hold the last commanded pose with base
            # velocity zeroed. Emitting all-zeros here would slew the arms to
            # the home pose and half-open the grippers (releasing the radio).
            if self._last_emitted_action is None:
                held = th.zeros(23, dtype=th.float32)
            else:
                held = self._last_emitted_action.clone()
                held[0:3] = 0.0   # zero base velocity (safety)
            if not self._post_orchestration_log_done:
                logger.info(
                    "[orch step %d] all actions done; holding last pose "
                    "(base_vel=0) until episode termination.",
                    self._step,
                )
                self._post_orchestration_log_done = True
            self._step += 1
            return held

        a = self._current()
        if not self._entered:
            logger.info(
                "[orch step %d] ENTER action %d/%d: %s (executor=%s, expected_phase=%d)",
                self._step, self.idx + 1, len(self.actions),
                a.name, a.executor.value, a.expected_phase_id,
            )
            a.on_enter(obs)
            self._entered = True
            self._steps_in_action = 0
            self._current_enter_step = self._step

        out = a.forward(obs)
        a._last_action = out  # cached for is_done's PCA call
        self._last_emitted_action = out  # cached for post-orchestration hold
        # Also publish into shared_state so downstream phases (e.g.
        # CloseRightGripperAction) can read the LAST COMMANDED action — the
        # IK target — rather than reading state23 (which lags the command
        # because of PD controller dynamics).
        self.shared_state["_last_emitted_action"] = out
        self._steps_in_action += 1

        done = a.is_done(obs, self.phase_detector)

        if self._step % self.HEARTBEAT_EVERY == 0:
            logger.info(
                "[orch step %d] action=%s in_action_step=%d "
                "pca_phase=%d debounce=%d/%d shared_state=%s",
                self._step, a.name, self._steps_in_action,
                a._last_phase_pred, a._consecutive_in_next_phase,
                a.pca_debounce_steps, dict(self.shared_state),
            )

        if done:
            logger.info(
                "[orch step %d] EXIT action %s (in_action_step=%d, "
                "final_pca_phase=%d, shared_state=%s)",
                self._step, a.name, self._steps_in_action,
                a._last_phase_pred, dict(self.shared_state),
            )
            a.on_exit(obs)
            if self._current_enter_step is not None:
                self._action_spans.append((a.name, self._current_enter_step, self._step + 1))
                self._current_enter_step = None
            self.idx += 1
            self._entered = False
            self._resolved = None  # drop the lazy instance if any

        self._step += 1
        return out
