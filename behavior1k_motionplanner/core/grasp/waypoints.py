"""Vertical descent waypoint generator for top-down grasping.

Mirrors the world-frame z-offsets used in `zexternal/run_server_final.py:825-840`:
[ +0.4 m, +0.3 m, +0.2 m, +0.1 m ] above the final grasp pose, then the grasp
pose itself. The first 4 are pre-grasp waypoints; the last is the contact pose.

Each waypoint is a 7-D `[x, y, z, qx, qy, qz, qw]` array in the same frame as
the input grasp pose (usually robot base).
"""
from __future__ import annotations

from typing import List

import numpy as np

# 5-waypoint descent at +0.20, +0.15, +0.10, +0.05, +0.00 m above grasp pose.
# Note: the offsets here are relative to the GRASP POSE, not the radio. The
# grasp pose itself carries an absolute offset above the radio via
# `PickUpGraspAction.GRASP_HEIGHT_OFFSET_M`, so the final wrist z is
# (radio.z + GRASP_HEIGHT_OFFSET_M + wp_offset).
_DEFAULT_Z_OFFSETS_M = [0.07, 0.00]


def vertical_descent_waypoints(
    grasp_pose_7d: np.ndarray,
    z_offsets_m: List[float] | None = None,
) -> List[np.ndarray]:
    """Build a list of 7-D pre-grasp + grasp waypoints.

    Args:
        grasp_pose_7d: target contact pose `[x, y, z, qx, qy, qz, qw]`.
        z_offsets_m: vertical offsets above the grasp pose, top first. Default
                     `[0.2, 0.0]` — a single pre-grasp 20 cm above + the grasp.

    Returns:
        A list of 7-D arrays sharing the same orientation as `grasp_pose_7d`,
        with z-positions equal to `grasp_pose_7d.z + offset`. The final waypoint
        (offset=0) is the grasp pose itself.
    """
    grasp = np.asarray(grasp_pose_7d, dtype=np.float64).reshape(-1)
    if grasp.size != 7:
        raise ValueError(f"grasp_pose_7d must be 7-D; got shape {grasp.shape}")
    offsets = list(z_offsets_m) if z_offsets_m is not None else _DEFAULT_Z_OFFSETS_M
    out = []
    for dz in offsets:
        wp = grasp.copy()
        wp[2] = grasp[2] + float(dz)
        out.append(wp)
    return out
