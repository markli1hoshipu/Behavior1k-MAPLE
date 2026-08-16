"""Grasp utilities — vertical waypoint generator only.

`compute_top_down_grasp_pose` came from `zexternal/pose6d/6d2grasp.py` which
depended on `omnigibson` + the SAM-6D camera frame pipeline. That was overkill
for our needs (we have ground-truth radio pose from the env directly), so it
was dropped along with the rest of the now-deleted `third_party/pose6d/` tree.

If you later need a real grasp-pose computer, write it locally against
`env.task.object_scope[radio].get_position_orientation()` plus the radio's
bounding-box axes — far simpler than the 6d2grasp.py pipeline.
"""
from __future__ import annotations

from .waypoints import vertical_descent_waypoints

__all__ = ["vertical_descent_waypoints"]
