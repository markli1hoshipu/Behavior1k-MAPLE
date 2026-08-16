"""Fit the PCA + KMeans phase detector on the 200 HF demo episodes.

For each demo, every frame contributes a 46-D (state_23 ⊕ action_23) vector plus
its normalized time-in-episode in [0, 1]. After fitting:

  1. PCA reduces 46-D → 2-D (or N-D, see --n_components)
  2. KMeans clusters the PCA-projected frames into K=4 groups
  3. Each cluster gets a *phase ID* assigned by ranking the clusters' mean
     normalized timestamp: smallest mean t → phase 0, largest → phase 3.

Artifacts saved to `--ckpt_dir`:
  pca.pkl, kmeans.pkl, phase_map.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ..utils.obs import extract_state_23d

DEFAULT_DATA_DIR = Path("/shared_work/DATASETS/behavior-1k-embodiedAI-rollouts/data/task-0000")
# Fit script defaults to the turning_on_radio task's checkpoint dir. Override
# with --ckpt-dir when training a detector for another task.
DEFAULT_CKPT_DIR = (
    Path(__file__).resolve().parents[3]
    / "tasks" / "turning_on_radio" / "checkpoints"
)
DEFAULT_HF_INDEX_MAX = 5000  # HF episodes have idx < 5000; collected start at 5000


def load_hf_episode_features(parquet_path: Path):
    """Return (features_46, normalized_t) for one HF demo parquet."""
    table = pq.read_table(parquet_path, columns=["observation.state", "action"])
    state = np.stack([np.asarray(s) for s in table["observation.state"].to_pylist()])
    action = np.stack([np.asarray(a) for a in table["action"].to_pylist()])
    state_23 = extract_state_23d(state)  # [T, 23]
    feat = np.concatenate([state_23, action], axis=1).astype(np.float32)  # [T, 46]
    T = feat.shape[0]
    t = np.arange(T, dtype=np.float32) / max(T - 1, 1)
    return feat, t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ckpt_dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--n_components", type=int, default=2, help="PCA target dim")
    parser.add_argument("--k", type=int, default=4, help="KMeans cluster count")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1) Gather HF-only episodes (idx < DEFAULT_HF_INDEX_MAX).
    parquets = sorted(args.data_dir.glob("episode_*.parquet"))
    hf_parquets = [p for p in parquets if int(p.stem.split("_")[1]) < DEFAULT_HF_INDEX_MAX]
    print(f"Found {len(hf_parquets)} HF demo parquets (filter idx < {DEFAULT_HF_INDEX_MAX})")

    feats, ts = [], []
    for i, p in enumerate(hf_parquets):
        feat, t = load_hf_episode_features(p)
        feats.append(feat); ts.append(t)
        if (i + 1) % 50 == 0:
            print(f"  loaded {i+1}/{len(hf_parquets)}: {p.name}  T={feat.shape[0]}")
    X = np.vstack(feats)
    t = np.concatenate(ts)
    print(f"Pooled features: X.shape={X.shape}, t.shape={t.shape}")

    # 2) Fit PCA.
    pca = PCA(n_components=args.n_components, random_state=args.seed).fit(X)
    Z = pca.transform(X)
    print(f"PCA: explained_variance_ratio={pca.explained_variance_ratio_.tolist()}")

    # 3) Fit KMeans on PCA features.
    km = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed).fit(Z)
    labels = km.labels_

    # 4) Rank clusters by mean normalized timestamp → assign phase IDs.
    cluster_mean_t = np.array([t[labels == k].mean() for k in range(args.k)])
    cluster_count = np.array([(labels == k).sum() for k in range(args.k)])
    order = np.argsort(cluster_mean_t)
    phase_map = {int(c): int(p) for p, c in enumerate(order)}

    print("\nCluster → phase assignment (by mean normalized t):")
    for cid in range(args.k):
        print(f"  cluster {cid}: mean_t={cluster_mean_t[cid]:.3f}  "
              f"n={cluster_count[cid]:6d}  → phase {phase_map[cid]}")

    # 5) Save artifacts.
    joblib.dump(pca, args.ckpt_dir / "pca.pkl")
    joblib.dump(km, args.ckpt_dir / "kmeans.pkl")
    with (args.ckpt_dir / "phase_map.json").open("w") as f:
        json.dump(phase_map, f, indent=2)
    print(f"\nSaved pca.pkl, kmeans.pkl, phase_map.json to {args.ckpt_dir}")


if __name__ == "__main__":
    main()
