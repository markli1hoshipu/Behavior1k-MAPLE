"""Task registry.

Each task lives in its own submodule under `behavior1k_mp.tasks.<name>` and
must expose these module-level attributes:

  TASK_NAME               : str, short slug matching the folder name
  TASK_DISPLAY_NAME       : str, human-readable ("turning on radio")
  TARGET_OBJECT_SCOPE_NAME: str, BDDL object-scope key that Actions target
  CHECKPOINT_DIR          : pathlib.Path to per-task checkpoints
  SLOT_TO_SKILL_IDX       : dict[str, int], orchestrator slot name → HF skill idx
  SKILL_TEMPLATES         : list[dict], one HF skill-annotation template per skill
  build_actions(env, robot, target_obj, vla) -> list[Action]
      Return the ordered Action list this task executes.

See `behavior1k_mp/tasks/turning_on_radio/__init__.py` for a reference impl.
"""
from __future__ import annotations

import importlib
from types import ModuleType


def load_task(task_name: str) -> ModuleType:
    """Import and return the task module by short name.

    Raises ImportError if the task module isn't installed under
    `behavior1k_mp.tasks.<task_name>`.
    """
    return importlib.import_module(f"behavior1k_mp.tasks.{task_name}")
