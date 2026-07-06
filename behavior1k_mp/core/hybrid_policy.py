"""HybridPolicy — pluggable into OmniGibson's `eval_data_gen.py`.

Subclasses `omnigibson.learning.policies.LocalPolicy` so the existing Hydra
instantiation `instantiate(self.cfg.model)` works without changes.

Task-agnostic: the concrete Action list, HF skill-mapping, and default
target-object name all come from the requested task module under
`behavior1k_mp.tasks.<task_name>`. See `docs/task_authoring.md`.

Construction is in two phases:
  1. `__init__(...)` — pure config; the env doesn't exist yet.
  2. `attach_env(env)` — called by the patched `eval_data_gen.py` after the env
     has been built; resolves the target object, instantiates the phase detector,
     builds the Action list from the task module, and wires up the Orchestrator.
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
        task_name: str = "turning_on_radio",
        action_dim: int = 23,
        vla_host: str = "127.0.0.1",
        vla_port: int = 8765,
        phase_ckpt_dir: Optional[str] = None,
        target_obj_scope_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(action_dim=action_dim)
        _ensure_basic_logging()

        from behavior1k_mp.tasks import load_task
        self.task_name = task_name
        self.task = load_task(task_name)

        self.vla_host = vla_host
        self.vla_port = vla_port
        self.phase_ckpt_dir = (
            Path(phase_ckpt_dir)
            if phase_ckpt_dir is not None
            else self.task.CHECKPOINT_DIR
        )
        self.target_obj_scope_name = (
            target_obj_scope_name
            if target_obj_scope_name is not None
            else self.task.TARGET_OBJECT_SCOPE_NAME
        )
        logger.info("HybridPolicy.__init__: task=%s, action_dim=%d, vla=%s:%d, "
                    "phase_ckpt=%s, target=%s",
                    task_name, action_dim, vla_host, vla_port,
                    self.phase_ckpt_dir, self.target_obj_scope_name)

        self.env = None
        self.orch = None

    # ───────────────────────── attach + reset ─────────────────────────
    def attach_env(self, env) -> None:
        """Called once by the evaluator after the env is built."""
        from .orchestrator import Orchestrator
        from .phase_detector.detector import PCAPhaseDetector

        self.env = env
        target_obj = env.task.object_scope[self.target_obj_scope_name]
        robot = env.robot if hasattr(env, "robot") else env.robots[0]

        if WebsocketPolicy is None:
            raise RuntimeError(
                "omnigibson.learning.policies.WebsocketPolicy not importable"
            )
        vla = WebsocketPolicy(host=self.vla_host, port=self.vla_port)

        phase_det = PCAPhaseDetector(self.phase_ckpt_dir)
        actions = self.task.build_actions(env, robot, target_obj, vla)
        self.orch = Orchestrator(actions, phase_det, shared_state={})
        logger.info("HybridPolicy attached: task=%s, target=%s, vla=%s:%d, ckpt=%s",
                    self.task_name, self.target_obj_scope_name,
                    self.vla_host, self.vla_port, self.phase_ckpt_dir)

    def reset(self) -> None:
        if self.orch is not None:
            self.orch.reset()

    def is_done(self) -> bool:
        """True when the orchestrator has run through all action slots and
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
    def build_skill_annotation(self, total_frames: int) -> dict:
        """Build an HF-format annotations JSON (skill_annotation +
        primitive_annotation) from the orchestrator's recorded action spans.

        The task module provides the sub-phase → skill bucketing and the
        per-skill template fields; the generic contiguous-span construction
        below is task-agnostic.
        """
        if self.orch is None:
            return {}
        return build_hf_skill_annotation(
            self.orch.action_spans(total_frames=total_frames),
            total_frames=total_frames,
            slot_to_skill_idx=self.task.SLOT_TO_SKILL_IDX,
            skill_templates=self.task.SKILL_TEMPLATES,
            task_display_name=self.task.TASK_DISPLAY_NAME,
        )


def build_hf_skill_annotation(
    spans: list[tuple[str, int, int]],
    total_frames: int,
    slot_to_skill_idx: dict[str, int],
    skill_templates: list[dict],
    task_display_name: str,
) -> dict:
    """Generic HF-format annotations builder used by HybridPolicy.

    Buckets orchestrator sub-phase spans into the task's 4-skill HF schema.
    Contiguous, gap-free segmentation: each skill picks up where the last
    ended; skill 0 anchors at frame 0; the tail belongs to the last skill.
    """
    n_skills = len(skill_templates)
    per_skill: list[dict[str, int]] = [{} for _ in range(n_skills)]
    for name, enter, exit_ in spans:
        si = slot_to_skill_idx.get(name)
        if si is None:
            continue
        d = per_skill[si]
        d["enter"] = min(enter, d["enter"]) if "enter" in d else enter
        d["exit"] = max(exit_, d["exit"]) if "exit" in d else exit_

    skill_annotation = []
    prev_end = 0
    for si in range(n_skills):
        d = per_skill[si]
        if d:
            start = max(prev_end, d.get("enter", prev_end))
            end = max(start, d.get("exit", start))
        else:
            start = prev_end
            end = prev_end  # zero-width when this skill never executed
        if si == n_skills - 1:
            end = max(end, total_frames)  # tail belongs to last skill
        if si == 0:
            start = 0  # first skill anchors at episode start
        tpl = skill_templates[si]
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

    # Primitives: task-specific fold (default = 0+1 → primitive 0, 2 → 1, 3 → 2)
    # Keep the historical shape for turning_on_radio; other tasks can override
    # by exposing PRIMITIVE_FOLD on the task module.
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
        "task_name": task_display_name,
        "data_folder": "",
        "meta_data": {
            "task_duration": int(total_frames),
            "valid_duration": [0, int(total_frames)],
        },
        "skill_annotation": skill_annotation,
        "primitive_annotation": primitive_annotation,
    }
