"""Pose math helpers — 7-D `[x, y, z, qx, qy, qz, qw]` representation throughout.

We deliberately use numpy + scipy.spatial.transform.Rotation rather than torch
for these helpers, so they're usable both offline (library build, scripts) and
at runtime (action `on_enter`).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


# ───────────────────────── core composition / inverse ─────────────────────────

def quat_xyzw_to_R(q):
    return R.from_quat(np.asarray(q, dtype=np.float64))


def R_to_quat_xyzw(rot):
    return rot.as_quat()  # scipy returns xyzw


def pose7_to_matrix(pose7):
    """[x,y,z, qx,qy,qz,qw] -> 4x4 homogeneous transform."""
    pose7 = np.asarray(pose7, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_R(pose7[3:7]).as_matrix()
    T[:3, 3] = pose7[:3]
    return T


def matrix_to_pose7(T):
    """4x4 -> [x,y,z, qx,qy,qz,qw]."""
    q = R.from_matrix(T[:3, :3]).as_quat()
    return np.concatenate([T[:3, 3], q]).astype(np.float64)


def pose_in_frame(pose_world: np.ndarray, frame_world: np.ndarray) -> np.ndarray:
    """Express `pose_world` in the local frame `frame_world` (both 7-D)."""
    T_w = pose7_to_matrix(pose_world)
    T_f = pose7_to_matrix(frame_world)
    return matrix_to_pose7(np.linalg.inv(T_f) @ T_w)


def compose_pose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`a` ∘ `b` — i.e. apply b in a's frame."""
    return matrix_to_pose7(pose7_to_matrix(a) @ pose7_to_matrix(b))


def invert_pose(pose: np.ndarray) -> np.ndarray:
    return matrix_to_pose7(np.linalg.inv(pose7_to_matrix(pose)))


# ───────────────────────── yaw helpers (planar) ─────────────────────────

def yaw_to_quat_xyzw(yaw: float) -> np.ndarray:
    """Planar rotation around +Z as a quaternion."""
    half = 0.5 * float(yaw)
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float64)


def quat_xyzw_to_yaw(q) -> float:
    """Extract yaw (rotation around +Z) from a quaternion. Ignores roll/pitch."""
    q = np.asarray(q, dtype=np.float64)
    # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy² + qz²))
    qx, qy, qz, qw = q
    return float(np.arctan2(2.0 * (qw * qz + qx * qy),
                             1.0 - 2.0 * (qy * qy + qz * qz)))


# ───────────────────────── special rotations ─────────────────────────

def apply_roll_180_deg(pose7: np.ndarray) -> np.ndarray:
    """Rotate a 7-D pose by 180° around its **local Z-axis** (the gripper's
    approach axis after the base top-down rotation).

    Earlier this rotated around local X, which *cancelled* the base 180° X-flip
    in `PickUpGraspAction._top_down_grasp_pose` — leaving the gripper pointing
    upward (away from the radio). The intended use is to produce a second
    valid top-down grasp by flipping which finger goes on which side of the
    radio (rotating the gripper around the direction it's reaching toward the
    object). That's a yaw around the gripper's local Z, not a roll.

    Function name kept for back-compat with call sites that read it as
    "the A→B flip"; semantics now match what those call sites actually need.
    """
    pose7 = np.asarray(pose7, dtype=np.float64)
    pos = pose7[:3]
    q = quat_xyzw_to_R(pose7[3:7])
    flip_180 = R.from_euler("z", 180.0, degrees=True)
    return np.concatenate([pos, (q * flip_180).as_quat()]).astype(np.float64)


# ───────────────────────── distance for nearest-neighbor lookup ─────────────────────────

def weighted_l2_pose(a: np.ndarray, b: np.ndarray,
                     w_pos: float = 1.0, w_quat: float = 4.0) -> float:
    """Distance between two 7-D poses.

    Squared Euclidean on xyz (weight `w_pos`) + rotation-invariant quaternion
    distance `1 - |a·b|²` (weight `w_quat`, in [0, 1] regardless of sign-cover).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pos_d2 = float(((a[:3] - b[:3]) ** 2).sum())
    qd = float(np.dot(a[3:7], b[3:7]))
    quat_d2 = 1.0 - qd * qd
    return w_pos * pos_d2 + w_quat * quat_d2
