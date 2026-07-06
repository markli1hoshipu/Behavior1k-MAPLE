"""HybridPolicy — pluggable into OmniGibson's `eval_data_gen.py`.

Subclasses `omnigibson.learning.policies.LocalPolicy` so the existing Hydra
instantiation `instantiate(self.cfg.model)` works without changes.

Construction is in two phases:
  1. `__init__(...)` — pure config; the env doesn't exist yet.
  2. `attach_env(env)` — called by the patched `eval_data_gen.py` after the env
     has been built; resolves the radio object, instantiates the phase detector,
     builds the Action list (including a lazy press slot dispatching on the
     A/B hint published by PickUpApproachAction), and wires up the Orchestrator.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import torch as th

logger = logging.getLogger(__name__)


def _ensure_basic_logging() -> None:
    """Install a StreamHandler on the `behavior1k_mp` root logger so our
    INFO-level lifecycle logs surface in stderr regardless of how the host
    process (OmniGibson evaluator, pytest, ad-hoc script) has configured
    logging. Idempotent: safe to call from multiple entry points.
    """
    root = logging.getLogger("behavior1k_mp")
    root.setLevel(logging.INFO)
    if not any(getattr(h, "_b1k_mp_marker", False) for h in root.handlers):
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter(
            "[b1k_mp %(asctime)s %(name)s %(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        ))
        h._b1k_mp_marker = True  # type: ignore[attr-defined]
        root.addHandler(h)
        # Don't propagate to the root logger (the evaluator's setup might
        # double-format / filter our lines).
        root.propagate = False

# Import LocalPolicy lazily so this module can be imported in a non-OmniGibson env
# (e.g. for unit tests of the phase detector).
try:
    from omnigibson.learning.policies import LocalPolicy, WebsocketPolicy
    _HAS_OG = True
except Exception:  # pragma: no cover
    _HAS_OG = False

    class LocalPolicy:
        """Minimal stub so the class definition still imports outside OmniGibson."""

        def __init__(self, *args, action_dim=23, **kwargs):
            self.action_dim = action_dim

    WebsocketPolicy = None  # type: ignore


class HybridPolicy(LocalPolicy):
    def __init__(
        self,
        action_dim: int = 23,
        vla_host: str = "127.0.0.1",
        vla_port: int = 8765,
        phase_ckpt_dir: Optional[str] = None,
        target_obj_scope_name: str = "radio_receiver.n.01_1",
        **kwargs,
    ):
        super().__init__(action_dim=action_dim)
        _ensure_basic_logging()
        logger.info("HybridPolicy.__init__: action_dim=%d, vla=%s:%d, "
                    "phase_ckpt=%s, target=%s",
                    action_dim, vla_host, vla_port, phase_ckpt_dir,
                    target_obj_scope_name)
        self.vla_host = vla_host
        self.vla_port = vla_port
        self.phase_ckpt_dir = (
            Path(phase_ckpt_dir)
            if phase_ckpt_dir is not None
            else Path(__file__).resolve().parent / "phase_detector" / "checkpoints"
        )
        self.target_obj_scope_name = target_obj_scope_name

        self.env = None
        self.orch = None

    # ───────────────────────── attach + reset ─────────────────────────
    def attach_env(self, env) -> None:
        """Called once by the evaluator after the env is built."""
        from .actions import (
            CloseRightGripperAction,
            NavigateToRadioAction,
            PickUpApproachAction,
            PickUpGraspAction,
            PressReplayBase,
            PutDownRadioAction,
            WrapUpPolicyAction,
        )
        from .orchestrator import Orchestrator
        from .phase_detector.detector import PCAPhaseDetector

        self.env = env

        # Resolve the radio target by its BDDL scope name (e.g. "radio_receiver.n.01_1").
        target_obj = env.task.object_scope[self.target_obj_scope_name]
        robot = env.robot if hasattr(env, "robot") else env.robots[0]

        if WebsocketPolicy is None:
            raise RuntimeError("omnigibson.learning.policies.WebsocketPolicy not importable")
        vla = WebsocketPolicy(host=self.vla_host, port=self.vla_port)

        phase_det = PCAPhaseDetector(self.phase_ckpt_dir)

        actions = [
            NavigateToRadioAction(env, robot, target_obj, policy=vla),
            PickUpApproachAction(env, robot, target_obj),
            PickUpGraspAction(env, robot, target_obj, active_arm="right"),
            CloseRightGripperAction(env, robot, target_obj),  # 1c: explicit close
            PressReplayBase(env, robot, target_obj),          # unified press: A/B routed via key lookup
            WrapUpPolicyAction(env, robot, target_obj, policy=vla),  # 5: X-VLA wrap-up
            PutDownRadioAction(env, robot, target_obj),
        ]
        self.orch = Orchestrator(actions, phase_det, shared_state={})
        logger.info("HybridPolicy attached: target=%s, vla=%s:%d, ckpt=%s",
                    self.target_obj_scope_name, self.vla_host, self.vla_port,
                    self.phase_ckpt_dir)

    def reset(self) -> None:
        if self.orch is not None:
            self.orch.reset()

    def is_done(self) -> bool:
        """True when the orchestrator has run through all 5 action slots and
        the post-orchestration hold has begun. Consumed by the patched
        evaluator loop to short-circuit the episode instead of running to
        the 4300-step truncation.
        """
        return self.orch is not None and self.orch.is_done

    def forward(self, obs, *args, **kwargs) -> th.Tensor:
        if self.orch is None:
            raise RuntimeError(
                "HybridPolicy.forward called before attach_env(env). "
                "Make sure eval_data_gen.py was patched to call "
                "`self.policy.attach_env(self.env)`."
            )
        return self.orch.forward(obs)

    # ───────────────── HF-format skill annotation export ─────────────────
    # Mapping from our 7-slot orchestrator chain to the HF dataset's 4-skill
    # / 3-primitive segmentation for `turning_on_radio`. The orchestrator
    # already records (action_name, enter_step, exit_step) for every slot;
    # we just bucket those slots into the HF skill labels.

    _SLOT_TO_SKILL_IDX = {
        # idx 0 = "move to" / navigation
        "navigate_to_radio": 0,
        # idx 1 = "pick up from" / uncoordinated
        "pick_up_radio_approach": 1,
        "pick_up_radio_grasp": 1,
        "close_right_gripper": 1,
        # idx 2 = "press" / coordinated
        "press_radio": 2,
        "press_radio_a": 2,   # legacy slot names, harmless to keep
        "press_radio_b": 2,
        # idx 3 = "place on" / uncoordinated
        "wrap_up_policy": 3,
        "put_down_radio": 3,
    }

    # Per-skill template fields — exact match to HF format.
    _SKILL_TEMPLATES = [
        {  # 0
            "skill_description": ["move to"],
            "skill_id": [1],
            "skill_type": ["navigation"],
            "object_ids_outer": [["radio_89"]],
            "manipulating_object_id": [],
            "memory_prefix": [],
            "spatial_prefix": [],
        },
        {  # 1
            "skill_description": ["pick up from"],
            "skill_id": [2],
            "skill_type": ["uncoordinated"],
            "object_ids_outer": [["radio_89", "coffee_table_koagbh_0"]],
            "manipulating_object_id": ["radio_89"],
            "memory_prefix": [],
            "spatial_prefix": [],
        },
        {  # 2
            "skill_description": ["press"],
            "skill_id": [67],
            "skill_type": ["coordinated"],
            "object_ids_outer": [["radio_89"]],
            "manipulating_object_id": ["radio_89"],
            "memory_prefix": [],
            "spatial_prefix": [],
        },
        {  # 3
            "skill_description": ["place on"],
            "skill_id": [3],
            "skill_type": ["uncoordinated"],
            "object_ids_outer": [["radio_89", "coffee_table_koagbh_0"]],
            "manipulating_object_id": ["radio_89"],
            "memory_prefix": ["back"],
            "spatial_prefix": [],
        },
    ]

    def build_skill_annotation(self, total_frames: int) -> dict:
        """Build an HF-format annotations JSON (skill_annotation +
        primitive_annotation) from the orchestrator's recorded action spans.

        Skill boundaries are derived from the *exit* step of the last slot
        in each bucket. If a bucket has no executed slot (e.g. nav stuck →
        pick/press/place never ran), the matching skill gets a zero-width
        span at the episode end.
        """
        if self.orch is None:
            return {}
        spans = self.orch.action_spans(total_frames=total_frames)

        # Per HF skill, find the latest exit_step among slots in that bucket,
        # and the earliest enter_step.
        per_skill: list[dict[str, int]] = [{} for _ in range(4)]
        for name, enter, exit_ in spans:
            si = self._SLOT_TO_SKILL_IDX.get(name)
            if si is None:
                continue
            d = per_skill[si]
            d["enter"] = min(enter, d["enter"]) if "enter" in d else enter
            d["exit"] = max(exit_, d["exit"]) if "exit" in d else exit_

        # Skill 0 always starts at frame 0. Subsequent skills start where the
        # previous skill ended (this makes the 4 segments contiguous like HF).
        # Skill 3 ends at total_frames.
        skill_annotation = []
        prev_end = 0
        for si in range(4):
            d = per_skill[si]
            if d:
                start = max(prev_end, d.get("enter", prev_end))
                end = max(start, d.get("exit", start))
            else:
                start = prev_end
                end = prev_end  # zero-width
            if si == 3:
                end = max(end, total_frames)  # tail belongs to "place on"
            if si == 0:
                start = 0  # nav always anchors at the episode start
            tpl = self._SKILL_TEMPLATES[si]
            skill_annotation.append({
                "skill_idx": si,
                "skill_id": tpl["skill_id"],
                "skill_description": tpl["skill_description"],
                "object_id": tpl["object_ids_outer"],
                "manipulating_object_id": tpl["manipulating_object_id"],
                "memory_prefix": tpl["memory_prefix"],
                "spatial_prefix": tpl["spatial_prefix"],
                "frame_duration": [start, end],
                "mp_ef": [],
                "skill_type": tpl["skill_type"],
            })
            prev_end = end

        # Primitive annotation: skills 0+1 merge into primitive 0,
        # skill 2 → primitive 1, skill 3 → primitive 2.
        s = skill_annotation
        primitive_annotation = [
            {
                "primitive_idx": 0,
                "primitive_id": s[1]["skill_id"],
                "primitive_description": s[1]["skill_description"],
                "object_id": s[1]["object_id"],
                "manipulating_object_id": s[1]["manipulating_object_id"],
                "memory_prefix": s[1]["memory_prefix"],
                "spatial_prefix": s[1]["spatial_prefix"],
                "frame_duration": [0, s[1]["frame_duration"][1]],
                "skill_idxes": [0, 1],
            },
            {
                "primitive_idx": 1,
                "primitive_id": s[2]["skill_id"],
                "primitive_description": s[2]["skill_description"],
                "object_id": s[2]["object_id"],
                "manipulating_object_id": s[2]["manipulating_object_id"],
                "memory_prefix": s[2]["memory_prefix"],
                "spatial_prefix": s[2]["spatial_prefix"],
                "frame_duration": s[2]["frame_duration"],
                "skill_idxes": [2],
            },
            {
                "primitive_idx": 2,
                "primitive_id": s[3]["skill_id"],
                "primitive_description": s[3]["skill_description"],
                "object_id": s[3]["object_id"],
                "manipulating_object_id": s[3]["manipulating_object_id"],
                "memory_prefix": s[3]["memory_prefix"],
                "spatial_prefix": s[3]["spatial_prefix"],
                "frame_duration": s[3]["frame_duration"],
                "skill_idxes": [3],
            },
        ]

        return {
            "task_name": "turning on radio",
            "data_folder": "",
            "meta_data": {
                "task_duration": int(total_frames),
                "valid_duration": [0, int(total_frames)],
            },
            "skill_annotation": skill_annotation,
            "primitive_annotation": primitive_annotation,
        }
