"""Task module for BEHAVIOR-1K task 0: turning on the radio.

Provides the concrete Action list executed by HybridPolicy and the HF
skill-annotation config used by the annotation exporter.

The 7-slot pipeline: navigate → pick_approach → pick_grasp (MP-IK) →
close_right_gripper → press → wrap_up (X-VLA) → put_down.
"""
from __future__ import annotations

from pathlib import Path

TASK_NAME = "turning_on_radio"
TASK_DISPLAY_NAME = "turning on radio"
TARGET_OBJECT_SCOPE_NAME = "radio_receiver.n.01_1"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def build_actions(env, robot, target_obj, vla):
    """Return the ordered list of Actions this task runs.

    Called by `HybridPolicy.attach_env` once the env exists.
    """
    from .actions import (
        CloseRightGripperAction,
        NavigateToRadioAction,
        PickUpApproachAction,
        PickUpGraspAction,
        PressReplayBase,
        PutDownRadioAction,
        WrapUpPolicyAction,
    )
    return [
        NavigateToRadioAction(env, robot, target_obj, policy=vla),
        PickUpApproachAction(env, robot, target_obj),
        PickUpGraspAction(env, robot, target_obj, active_arm="right"),
        CloseRightGripperAction(env, robot, target_obj),
        PressReplayBase(env, robot, target_obj),
        WrapUpPolicyAction(env, robot, target_obj, policy=vla),
        PutDownRadioAction(env, robot, target_obj),
    ]


# ─── HF-format annotation config ──────────────────────────────────────────
# Bucket the 7 orchestrator slots into the HF challenge's 4-skill schema.
SLOT_TO_SKILL_IDX = {
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

# Exact-match HF challenge template per skill index.
SKILL_TEMPLATES = [
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
