#!/usr/bin/env python3
"""Build the two press-action libraries from the 200 HF demos.

Workflow:
  1. For every HF demo, slice out the frames whose `skill_annotation` says "press".
  2. Compute the 36-D (state_18 + action_18) feature per frame.
  3. Fit PCA(36 -> 2) + KMeans(k=2) on the pooled press frames.
  4. Per episode, take the **mode** of its frames' cluster IDs => episode label ∈ {0, 1}.
  5. For each episode, save the full press-phase (state_23, action_23) trajectory into
     either library_A.pkl (cluster 0) or library_B.pkl (cluster 1).
  6. Save press_pca.pkl + press_kmeans.pkl + episode_labels.json for runtime use.

All artifacts land in
  /shared_work/behavior1k-mp/behavior1k_mp/phase_detector/checkpoints/press_modes/
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from behavior1k_mp.utils.obs import extract_state_23d

DEFAULT_DATA_DIR = Path("/shared_work/DATASETS/behavior-1k-embodiedAI-rollouts/data/task-0000")
DEFAULT_ANN_DIR = Path("/shared_work/DATASETS/behavior-1k-embodiedAI-rollouts/annotations/task-0000")
DEFAULT_OUT_DIR = Path("/shared_work/behavior1k-mp/behavior1k_mp/phase_detector/checkpoints/press_modes")

# 18-DOF "active" indices on a 23-D layout: trunk(4) + L-arm(7) + R-arm(7).
ACT_18DOF_INDEX = list(range(3, 7)) + list(range(7, 14)) + list(range(15, 22))


def extract_press_segments(parquet_path: Path, ann_path: Path):
    """Return (state23_full, action23_full, press_frame_mask) for one episode.

    Returns (None, None, None) if no press segment exists in the annotation."""
    if not parquet_path.exists() or not ann_path.exists():
        return None, None, None
    with ann_path.open() as f:
        ann = json.load(f)

    ranges: list[tuple[int, int]] = []
    for seg in ann.get("skill_annotation", []):
        descs = seg.get("skill_description") or []
        if not descs or descs[0] != "press":
            continue
        fd = seg["frame_duration"]
        for r in (fd if isinstance(fd[0], list) else [fd]):
            ranges.append((int(r[0]), int(r[1])))
    if not ranges:
        return None, None, None

    table = pq.read_table(parquet_path, columns=["observation.state", "action"])
    state256 = np.stack([np.asarray(s) for s in table["observation.state"].to_pylist()])
    action23 = np.stack([np.asarray(a) for a in table["action"].to_pylist()]).astype(np.float32)
    state23 = extract_state_23d(state256).astype(np.float32)

    T = state23.shape[0]
    mask = np.zeros(T, dtype=bool)
    for lo, hi in ranges:
        mask[max(0, lo): min(T, hi)] = True
    return state23, action23, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ann_dir", type=Path, default=DEFAULT_ANN_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(args.ann_dir.glob("episode_*.json"))
    print(f"Scanning {len(ann_files)} HF annotation files...")

    # ─── pass 1: load all press frames + remember which episode each frame came from
    per_ep_frames: dict[int, np.ndarray] = {}   # ep_idx -> [Ti, 36] feature
    per_ep_traj: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # ep_idx -> (state23 of press, action23 of press)

    for ann_path in ann_files:
        ep_idx = int(ann_path.stem.split("_")[1])
        parquet_path = args.data_dir / f"episode_{ep_idx:08d}.parquet"
        state23, action23, mask = extract_press_segments(parquet_path, ann_path)
        if state23 is None or not mask.any():
            continue
        state_press = state23[mask]
        action_press = action23[mask]
        feat = np.concatenate(
            [state_press[:, ACT_18DOF_INDEX], action_press[:, ACT_18DOF_INDEX]],
            axis=1,
        )
        per_ep_frames[ep_idx] = feat
        per_ep_traj[ep_idx] = (state_press, action_press)
    print(f"  episodes with a press segment: {len(per_ep_frames)}")

    # ─── pass 2: fit PCA + KMeans on pooled press frames.
    # We use STATE-ONLY features (not state+action). Reasons:
    #   1. The standalone PCA showed state-only had the highest explained variance
    #      (66.3%) for press frames, since press is mostly a static pose hold.
    #   2. At runtime, the auto-dispatcher classifies at phase-2 *entry*, before
    #      any action has been emitted — only state is available there.
    # We assemble state-18 from the cached 36-D (first 18 columns are state_18).
    X = np.vstack([f[:, :18] for f in per_ep_frames.values()])
    print(f"  pooled press frames (state-only): {X.shape}")
    pca = PCA(n_components=2, random_state=args.seed).fit(X)
    Z = pca.transform(X)
    km = KMeans(n_clusters=2, n_init=10, random_state=args.seed).fit(Z)
    print(f"  PCA evr={pca.explained_variance_ratio_.round(3).tolist()}  "
          f"KMeans inertia={km.inertia_:.1f}")

    # ─── pass 3: per-episode cluster = mode of its frames' cluster IDs
    # (feat[:, :18] is the state-18 portion the classifier was fit on)
    episode_labels: dict[int, int] = {}
    label_counter: Counter = Counter()
    for ep_idx, feat in per_ep_frames.items():
        z = pca.transform(feat[:, :18])
        per_frame = km.predict(z)
        label = int(Counter(per_frame).most_common(1)[0][0])
        episode_labels[ep_idx] = label
        label_counter[label] += 1
    print(f"  episode-level labels: A(cluster 0) = {label_counter[0]}, "
          f"B(cluster 1) = {label_counter[1]}")

    # ─── pass 4: dump per-cluster libraries
    # Each library: list[ dict(episode_index=int, state=[Ti,23], action=[Ti,23]) ]
    library_a, library_b = [], []
    for ep_idx, label in episode_labels.items():
        state_press, action_press = per_ep_traj[ep_idx]
        entry = {
            "episode_index": ep_idx,
            "state":  state_press,    # [Ti, 23]   absolute proprio for press window
            "action": action_press,   # [Ti, 23]   recorded teleop action for press window
        }
        (library_a if label == 0 else library_b).append(entry)

    joblib.dump(library_a, args.out_dir / "library_A.pkl", compress=3)
    joblib.dump(library_b, args.out_dir / "library_B.pkl", compress=3)
    joblib.dump(pca,       args.out_dir / "press_pca.pkl")
    joblib.dump(km,        args.out_dir / "press_kmeans.pkl")
    with (args.out_dir / "episode_labels.json").open("w") as f:
        json.dump(
            {"by_episode": {str(k): int(v) for k, v in episode_labels.items()},
             "library_A_size": label_counter[0],
             "library_B_size": label_counter[1],
             "feature_indices_18dof": ACT_18DOF_INDEX},
            f, indent=2,
        )
    print(f"\nSaved to {args.out_dir}:")
    for p in sorted(args.out_dir.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:25s}  {size_kb:8.1f} KB")


if __name__ == "__main__":
    main()
