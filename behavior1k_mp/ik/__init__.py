"""IK facade — exposes Mark's gradient-descent IK from
`third_party/zexternal_utils.py::solve_ik_r1pro`.

Mark's IK is intentionally simple: pytorch_kinematics builds an FK chain from
the R1Pro URDF, then `torch.optim.Adam` minimises a weighted (position + 6-D
rotation) loss against the target pose. No CasADi, no Pinocchio, no NLP — just
gradient descent on `q`. Runtime dependency: `torch + pytorch_kinematics`.

The URDF lives at `third_party/r1pro.urdf` and is shipped with the package.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Optional

import numpy as np

_THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"
DEFAULT_URDF_PATH = _THIRD_PARTY / "r1pro.urdf"
_zexternal_utils_path = _THIRD_PARTY / "zexternal_utils.py"
_cached_module = None


def _load_zexternal_utils():
    """Load `third_party/zexternal_utils.py` once, by file path (not by
    `import zexternal_utils`). Keeps it isolated from any other module names."""
    global _cached_module
    if _cached_module is None:
        spec = importlib.util.spec_from_file_location(
            "behavior1k_mp_third_party.zexternal_utils", _zexternal_utils_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load {_zexternal_utils_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _cached_module = mod
    return _cached_module


def solve_ik_r1pro(
    mode: str,
    target_pos_m: np.ndarray,
    target_quat_xyzw: np.ndarray,
    urdf_path: Optional[str] = None,
    **kwargs: Any,
):
    """Solve IK for the R1Pro using Mark's simple gradient-descent solver.

    Args:
        mode: "left_torso" or "right_torso" — selects which arm chain to drive.
        target_pos_m:  3-vector, target end-effector position (m, robot base frame).
        target_quat_xyzw: 4-vector, target end-effector orientation (scipy xyzw).
        urdf_path: override the URDF path; defaults to the in-tree
                   `third_party/r1pro.urdf`.
        **kwargs: forwarded to the underlying `solve_ik_r1pro` (max_iters, lr,
                  pos_weight, ori_weight, pos_tol, ori_tol, initial_guess,
                  extra_link_xyzquat, verbose).

    Returns:
        Dict with keys like `q` (joint solution), `pos_err_m`, `ori_err_rad`,
        `converged`, etc. — see the upstream function for the full schema.
    """
    fn = _load_zexternal_utils().solve_ik_r1pro
    return fn(
        mode=mode,
        target_pos_m=np.asarray(target_pos_m, dtype=np.float64),
        target_quat_xyzw=np.asarray(target_quat_xyzw, dtype=np.float64),
        urdf_path=str(urdf_path) if urdf_path is not None else str(DEFAULT_URDF_PATH),
        **kwargs,
    )


__all__ = ["solve_ik_r1pro", "DEFAULT_URDF_PATH"]
