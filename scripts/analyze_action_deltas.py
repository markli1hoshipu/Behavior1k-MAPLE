"""Analyze per-frame action deltas in the human-demo libraries.

We use the resulting distribution to set MAX_STEP_PER_DIM in
`behavior1k_mp/utils/action_bridge.py`. Specifically: the bridge threshold
for each dim should be roughly the 99th-percentile (or the max) of
|action[t+1] - action[t]| observed in real demos. If humans never moved a
joint by more than 0.012 rad/frame, our bridge's 0.02 rad/frame cap is
*too coarse* — when the bridge ramps at the max rate, it moves faster than
any real demo ever does, which produces visible jerks.

Outputs a table:
    dim    name        max     p99     p95     p90     mean    +-----+
                                                                       \\
                                                              recommended MAX_STEP
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

CKPT = Path("/shared_work/behavior1k-mp/behavior1k_mp/tasks/turning_on_radio/checkpoints")

# 23-D layout (must match utils/obs.py)
DIM_NAMES = (
    ["base_vx", "base_vy", "base_wz"]
    + [f"trunk_q{i+1}"  for i in range(4)]
    + [f"l_arm_q{i+1}"  for i in range(7)]
    + ["l_grip"]
    + [f"r_arm_q{i+1}"  for i in range(7)]
    + ["r_grip"]
)
assert len(DIM_NAMES) == 23


def collect_deltas() -> np.ndarray:
    """Return a (T_total, 23) array of |a[t+1] - a[t]| across all demos."""
    pick = joblib.load(CKPT / "pick_library" / "library_pick.pkl")
    prA  = joblib.load(CKPT / "press_modes" / "library_A.pkl")
    prB  = joblib.load(CKPT / "press_modes" / "library_B.pkl")

    chunks = []
    for e in pick:
        a = np.asarray(e["action_traj"], dtype=np.float64)
        if len(a) > 1:
            chunks.append(np.abs(np.diff(a, axis=0)))
    for e in prA:
        a = np.asarray(e["action"], dtype=np.float64)
        if len(a) > 1:
            chunks.append(np.abs(np.diff(a, axis=0)))
    for e in prB:
        a = np.asarray(e["action"], dtype=np.float64)
        if len(a) > 1:
            chunks.append(np.abs(np.diff(a, axis=0)))
    return np.concatenate(chunks, axis=0)


def main():
    d = collect_deltas()
    print(f"Total transitions analyzed: {d.shape[0]:,} (across 200 pick + 79 + 121 press demos)\n")

    p99  = np.percentile(d, 99, axis=0)
    p95  = np.percentile(d, 95, axis=0)
    p90  = np.percentile(d, 90, axis=0)
    p50  = np.percentile(d, 50, axis=0)
    mx   = d.max(axis=0)
    mn_nz = np.array([d[d[:, i] > 0, i].min() if (d[:, i] > 0).any() else 0.0
                       for i in range(23)])

    # Current bridge thresholds (from utils/action_bridge.py)
    CURRENT = np.array(
        [0.10, 0.10, 0.10,
         0.02, 0.02, 0.02, 0.02,
         0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
         0.10,
         0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
         0.10],
        dtype=np.float64,
    )

    print(f"{'dim':>3}  {'name':<10}  {'median':>10}  {'p90':>10}  {'p95':>10}  "
          f"{'p99':>10}  {'max':>10}     {'current':>10}  {'p99/curr':>9}")
    print("-" * 110)
    for i in range(23):
        ratio = p99[i] / CURRENT[i] if CURRENT[i] > 0 else float("nan")
        flag = ""
        if ratio < 0.5:    flag = "  ← bridge too LOOSE"
        elif ratio > 1.0:  flag = "  ← bridge tighter than humans (good)"
        print(f"{i:>3}  {DIM_NAMES[i]:<10}  "
              f"{p50[i]:>10.5f}  {p90[i]:>10.5f}  {p95[i]:>10.5f}  "
              f"{p99[i]:>10.5f}  {mx[i]:>10.5f}     {CURRENT[i]:>10.5f}  "
              f"{ratio:>9.3f}{flag}")
    print()

    # Suggest a new bridge threshold: p99 of human deltas, with a small safety
    # multiplier (1.0×) and per-group floor so we don't go ridiculously small.
    print("Suggested MAX_STEP_PER_DIM (p99 of human deltas):")
    print(f"  base [0:3]:    {p99[0:3].round(4).tolist()}    (using max of base group: {p99[0:3].max():.4f})")
    print(f"  trunk [3:7]:   {p99[3:7].round(4).tolist()}    (group max: {p99[3:7].max():.4f})")
    print(f"  L_arm [7:14]:  {p99[7:14].round(4).tolist()}   (group max: {p99[7:14].max():.4f})")
    print(f"  L_grip [14]:   {p99[14]:.4f}")
    print(f"  R_arm [15:22]: {p99[15:22].round(4).tolist()}  (group max: {p99[15:22].max():.4f})")
    print(f"  R_grip [22]:   {p99[22]:.4f}")

    # Also: per-action TOTAL range (max - min across full trajectory) — the
    # biggest single jump a single-step bridge could need to absorb.
    print("\nFor reference — typical full-traj range per dim (p95 across all demos):")
    pick = joblib.load(CKPT / "pick_library" / "library_pick.pkl")
    ranges = []
    for e in pick:
        a = np.asarray(e["action_traj"], dtype=np.float64)
        ranges.append(a.max(axis=0) - a.min(axis=0))
    ranges = np.stack(ranges)
    rng_p95 = np.percentile(ranges, 95, axis=0)
    for grp_name, idx in [("trunk", slice(3, 7)),
                          ("L_arm", slice(7, 14)),
                          ("R_arm", slice(15, 22))]:
        print(f"  {grp_name}: max={rng_p95[idx].max():.3f} rad over a full pick demo")


if __name__ == "__main__":
    main()
