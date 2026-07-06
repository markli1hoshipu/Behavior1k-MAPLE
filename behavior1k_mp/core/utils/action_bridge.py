"""Dynamic action-bridge utilities.

When an action class switches to playing back an open-loop recorded trajectory
(approach/press replays), its first commanded action is the recorded frame 0,
which has no relation to the robot's current joint state. The discontinuity
shows up as a visible jerk.

This helper builds a short interpolation from `state23` (treated as a virtual
"current action" — works because state and action layouts are identical for
R1Pro, see `utils/obs.py`) into the recorded frame 0. The bridge length is
**computed per call** from the size of the jump: each interpolating dim is
allowed to change by at most `MAX_STEP_PER_DIM[d]` per 30 Hz frame, so small
jumps get short bridges (no wasted budget) and large jumps get long ones.

What interpolates and what is HELD CONSTANT during the bridge:
  - `action[0:3]`  base velocity → HELD AT 0. We never want the base to drive
                                   during a phase transition.
  - `action[3:7]`  trunk         → INTERPOLATED state23 → target
  - `action[7:14]` L_arm         → INTERPOLATED
  - `action[14]`   L_grip        → HELD AT state23[14]. The gripper should not
                                    open/close mid-bridge; whatever it was
                                    doing at phase entry is what it keeps doing.
  - `action[15:22]` R_arm        → INTERPOLATED
  - `action[22]`   R_grip        → HELD AT state23[22] (same reason).

Only the interpolating dims drive `n_frames`. Base + grippers are constant
within the bridge, so they don't need "time to ramp" — including them in the
length calculation would just produce wasted frames.
"""
from __future__ import annotations

import numpy as np

# Per-dim maximum allowed change per 30 Hz frame.
# Derived from p99 of per-frame |Δaction| observed across 171,195 transitions
# in 200 pick + 79 + 121 press demos (see scripts/analyze_action_deltas.py).
# A floor of 0.001 is applied to any dim with p99 ≈ 0 (so we never divide by
# zero in bridge math). Base + grippers are held in the bridge — their entries
# here are unused.
MAX_STEP_PER_DIM = np.array(
    [0.0286, 0.0242, 0.0156,   # base_vel              (held at 0; unused)
     0.0038, 0.0070, 0.0031, 0.0010,   # trunk q1..q4  (q4 floored — never moves)
     0.0149, 0.0141, 0.0216, 0.0441, 0.0187, 0.0319, 0.0244,   # L_arm q1..q7
     0.0010,   # L_grip                                (held; unused)
     0.0245, 0.0147, 0.0309, 0.0478, 0.0249, 0.0384, 0.0534,   # R_arm q1..q7
     0.0010],  # R_grip                                (held; unused)
    dtype=np.float32,
)
assert MAX_STEP_PER_DIM.shape == (23,)

# Indices that LINEARLY INTERPOLATE during a bridge.
_INTERP_DIMS = np.array(
    list(range(3, 7))      # trunk
    + list(range(7, 14))   # L_arm
    + list(range(15, 22)), # R_arm
    dtype=np.int64,
)
# Indices that are HELD AT ZERO during a bridge (base velocity).
_HOLD_ZERO_DIMS = np.array([0, 1, 2], dtype=np.int64)
# Indices that are HELD AT state23 during a bridge (grippers).
_HOLD_STATE_DIMS = np.array([14, 22], dtype=np.int64)


def required_bridge_frames(
    state23: np.ndarray,
    target_action_23: np.ndarray,
    max_step_per_dim: np.ndarray = MAX_STEP_PER_DIM,
    min_frames: int = 1,
    max_frames: int = 3000,
) -> int:
    """Compute the bridge length needed so no INTERPOLATING dim changes by
    more than `max_step_per_dim[d]` per frame.

    Only `_INTERP_DIMS` (trunk + both arms) contribute — base and grippers
    are held constant in the bridge and don't need ramp time.

    Clamped to `[min_frames, max_frames]`. Default cap = 3000 frames (100 s
    at 30 Hz) — effectively unbounded; a huge required bridge would only
    happen if some joint started ~60 rad off, which is unphysical.
    """
    state23 = np.asarray(state23, dtype=np.float32).reshape(-1)
    target = np.asarray(target_action_23, dtype=np.float32).reshape(-1)
    delta = np.abs(target - state23)
    delta_motion = delta[_INTERP_DIMS]
    step_motion = max_step_per_dim[_INTERP_DIMS]
    per_dim_frames = np.ceil(delta_motion / step_motion)
    n = int(per_dim_frames.max())
    return int(np.clip(n, min_frames, max_frames))


def linear_bridge_to(
    state23: np.ndarray,
    target_action_23: np.ndarray,
    n_frames: int | None = None,
    **kwargs,
) -> np.ndarray:
    """Return an [n_frames, 23] interpolation bridge from `state23` → target.

    - trunk + both arms: linearly interpolated state23 → target_action_23
    - base velocity dims: forced to 0
    - gripper dims: held at state23[14] and state23[22]

    If `n_frames is None`, computed via `required_bridge_frames(**kwargs)`.
    Endpoints (on interpolating dims): row 0 is one step away from state23
    (`alpha = 1/n_frames`); row n_frames-1 is `target_action_23` on those
    dims (`alpha = 1`). The caller's `forward()` returns row 0 first.
    """
    state23 = np.asarray(state23, dtype=np.float32).reshape(-1)
    target = np.asarray(target_action_23, dtype=np.float32).reshape(-1)
    if n_frames is None:
        n_frames = required_bridge_frames(state23, target, **kwargs)
    alphas = (np.arange(1, n_frames + 1, dtype=np.float32) / float(n_frames))[:, None]
    # Default: linear interpolation on all dims …
    bridge = (1.0 - alphas) * state23[None, :] + alphas * target[None, :]
    # … then override the held dims.
    bridge[:, _HOLD_ZERO_DIMS] = 0.0
    bridge[:, _HOLD_STATE_DIMS] = state23[_HOLD_STATE_DIMS][None, :]
    return bridge.astype(np.float32)
