#!/usr/bin/env python3
"""PCA analysis of phase-2 ("press") frames only.

Filter every HF demo to just the frames whose `skill_annotation` entry has
skill_description == "press", then run PCA on those frames' 18-DOF state and
18-DOF action vectors. The hypothesis is that the press phase contains
two sub-modes — "forward grasp" vs "backward grasp" — which should show up
as two clusters in PCA space.

Outputs:
  /shared_work/markhsp/runnings/viz_data/phase2_press_pca.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

DEFAULT_DATA_DIR = Path("/shared_work/DATASETS/behavior-1k-embodiedAI-rollouts/data/task-0000")
DEFAULT_ANN_DIR = Path("/shared_work/DATASETS/behavior-1k-embodiedAI-rollouts/annotations/task-0000")
DEFAULT_OUT = Path("/shared_work/markhsp/runnings/viz_data/phase2_press_pca.png")

# 18-DOF "active" indices — same slicing as your earlier delta-PCA script:
# trunk(4) + L-arm(7) + R-arm(7). Drops base velocity (0:3), L-grip (14), R-grip (22).
ACT_18DOF_INDEX = list(range(3, 7)) + list(range(7, 14)) + list(range(15, 22))


def extract_press_frames(parquet_path: Path, ann_path: Path):
    """Return (state_18, action_18) only for frames where the skill is 'press'.
    Returns empty arrays if the episode has no press segment.
    """
    if not ann_path.exists() or not parquet_path.exists():
        return None, None
    with ann_path.open() as f:
        ann = json.load(f)

    # Build per-frame label array
    press_ranges: list[tuple[int, int]] = []
    for seg in ann.get("skill_annotation", []):
        descs = seg.get("skill_description") or []
        if not descs or descs[0] != "press":
            continue
        fd = seg["frame_duration"]
        for r in (fd if isinstance(fd[0], list) else [fd]):
            press_ranges.append((int(r[0]), int(r[1])))
    if not press_ranges:
        return None, None

    table = pq.read_table(parquet_path, columns=["observation.state", "action"])
    state = np.stack([np.asarray(s) for s in table["observation.state"].to_pylist()])
    action = np.stack([np.asarray(a) for a in table["action"].to_pylist()])

    # Slice to 18-DOF active dims on action; for state we use the same indices
    # off the extracted-23D vector. Reuse the existing extractor for state.
    from behavior1k_mp.utils.obs import extract_state_23d
    state_23 = extract_state_23d(state)
    state_18 = state_23[:, [i - 0 if i < 23 else i for i in ACT_18DOF_INDEX]]  # 23-D layout matches action layout
    action_18 = action[:, ACT_18DOF_INDEX]

    # Concatenate frame indices in all press ranges
    keep = np.zeros(state.shape[0], dtype=bool)
    for lo, hi in press_ranges:
        keep[max(0, lo): min(state.shape[0], hi)] = True
    return state_18[keep], action_18[keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ann_dir", type=Path, default=DEFAULT_ANN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--k_sweep", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(args.ann_dir.glob("episode_*.json"))
    print(f"Found {len(ann_files)} HF annotation files")

    per_ep_state, per_ep_action, per_ep_id = [], [], []
    for i, ann_path in enumerate(ann_files):
        ep_idx = int(ann_path.stem.split("_")[1])
        parquet_path = args.data_dir / f"episode_{ep_idx:08d}.parquet"
        s, a = extract_press_frames(parquet_path, ann_path)
        if s is None or len(s) == 0:
            continue
        per_ep_state.append(s)
        per_ep_action.append(a)
        per_ep_id.append(np.full(len(s), ep_idx, dtype=np.int64))
        if (i + 1) % 50 == 0:
            print(f"  scanned {i+1}/{len(ann_files)}")

    state_18 = np.vstack(per_ep_state)
    action_18 = np.vstack(per_ep_action)
    ep_ids = np.concatenate(per_ep_id)
    print(f"\nPress frames pooled: state={state_18.shape}, action={action_18.shape}, "
          f"episodes={len(np.unique(ep_ids))}")

    feat_combined = np.concatenate([state_18, action_18], axis=1)  # 36-D

    # ───────────────────────── PCA on three feature sets ─────────────────────────
    def fit_pca(feat, label):
        pca = PCA(n_components=2, random_state=42).fit(feat)
        Z = pca.transform(feat)
        evr = pca.explained_variance_ratio_
        print(f"  {label:>20s}  PC1={evr[0]*100:5.1f}%  PC2={evr[1]*100:5.1f}%  "
              f"sum={evr.sum()*100:5.1f}%")
        return Z, pca

    print("\nPCA(2) explained variance:")
    Z_s, pca_s = fit_pca(state_18,     "state-only (18D)")
    Z_a, pca_a = fit_pca(action_18,    "action-only (18D)")
    Z_c, pca_c = fit_pca(feat_combined, "state+action (36D)")

    # ───────────────────────── KMeans sweep on state+action PCA ──────────────────
    print("\nKMeans on state+action PCA (silhouette score, higher is better):")
    sweep = {}
    for k in args.k_sweep:
        if k < 2 or len(Z_c) < k:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z_c)
        sil = silhouette_score(Z_c, km.labels_, sample_size=min(10_000, len(Z_c)),
                               random_state=42)
        sweep[k] = (km, sil)
        print(f"  k={k}: silhouette={sil:.3f}  cluster sizes={np.bincount(km.labels_).tolist()}")

    # ───────────────────────── Plotting ──────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    def scatter(ax, Z, c, title, cmap="tab20", s=4, alpha=0.5, label_cbar=None):
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=c, cmap=cmap, s=s, alpha=alpha)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if label_cbar:
            cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label(label_cbar, fontsize=8)
        return sc

    # Row 1: color by episode index (per-episode coherence)
    scatter(axes[0, 0], Z_s, ep_ids, f"state-only (PCA, {Z_s.shape[0]} press frames)\n"
            f"color = episode_index", label_cbar="episode")
    scatter(axes[0, 1], Z_a, ep_ids, f"action-only (PCA)\ncolor = episode_index",
            label_cbar="episode")
    scatter(axes[0, 2], Z_c, ep_ids, f"state+action (PCA)\ncolor = episode_index",
            label_cbar="episode")

    # Row 2: KMeans clustering on combined PCA, k=2 (the question) plus k=3 and 4
    for j, k in enumerate([2, 3, 4]):
        ax = axes[1, j]
        if k not in sweep:
            ax.axis("off"); continue
        km, sil = sweep[k]
        scatter(ax, Z_c, km.labels_,
                f"state+action PCA   KMeans(k={k}, silhouette={sil:.3f})",
                cmap="tab10")
        # plot centroids
        cx, cy = km.cluster_centers_[:, 0], km.cluster_centers_[:, 1]
        ax.scatter(cx, cy, s=120, marker="x", c="black", linewidths=2)

    fig.suptitle("Phase 2 (press) — PCA of HF demo frames", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved figure: {args.out}")
    plt.close(fig)

    # ───────────────────────── Quick interpretation of k=2 clusters ───────────
    if 2 in sweep:
        km, _ = sweep[2]
        labels = km.predict(Z_c)
        print("\nIf k=2 separates 'forward grasp' vs 'backward grasp', the two "
              "clusters should have systematically different arm orientations. "
              "Per-cluster mean state[7:14] (L-arm joints, first 3 dims):")
        for c in (0, 1):
            mu = state_18[labels == c].mean(axis=0)
            print(f"  cluster {c}  (n={np.sum(labels==c):5d}):  "
                  f"trunk_dy={mu[1]:+.3f}  L_arm_d0..d2={mu[4:7].round(3).tolist()}  "
                  f"R_arm_d0..d2={mu[11:14].round(3).tolist()}")


if __name__ == "__main__":
    main()
