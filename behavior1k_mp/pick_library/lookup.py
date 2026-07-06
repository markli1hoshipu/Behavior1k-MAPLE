"""Runtime nearest-neighbor lookup over the pick-approach trajectory library.

Each library entry is a dict:
    {
        "episode_index": int,
        "key_7d":        np.ndarray shape (7,)  — radio pose in robot base frame
                                                  at pick-start frame
        "action_traj":   np.ndarray shape (T, 23) — truncated to ~90% pre-gripper-close
        "state_traj":    np.ndarray shape (T, 23) — same frames' state (for replay
                                                    drift-correction if we want it later)
        "press_label":   str ∈ {"A", "B"}        — grasp-variant label, inherited
                                                    from the demo's press cluster
                                                    (legacy field name; semantically
                                                    this is the grasp variant used
                                                    by PickUpGraspAction)
    }

The library file lives at `phase_detector/checkpoints/pick_library/library_pick.pkl`.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from ..utils.pose import weighted_l2_pose


def load_library(path: str | Path) -> list[dict]:
    """Load the pick library; raises FileNotFoundError if absent."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Pick library missing at {path}. Run `scripts/build_pick_library.py` first."
        )
    return joblib.load(path)


def nearest_entry(library: list[dict], query_key_7d: np.ndarray,
                  w_pos: float = 1.0, w_quat: float = 4.0) -> tuple[dict, float, list[float]]:
    """Return the library entry closest to `query_key_7d` under weighted L2.

    Returns (best_entry, best_distance, all_distances).
    """
    if not library:
        raise ValueError("empty library")
    dists = [weighted_l2_pose(query_key_7d, e["key_7d"], w_pos=w_pos, w_quat=w_quat)
             for e in library]
    idx = int(np.argmin(dists))
    return library[idx], float(dists[idx]), dists
