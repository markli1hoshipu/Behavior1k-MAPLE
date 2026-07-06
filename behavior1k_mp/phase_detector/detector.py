"""Runtime PCA-+-KMeans phase classifier.

Inputs:  46-D feature = `extract_state_23d(state_256) ⊕ action_23`
Output:  phase ID ∈ {0, 1, 2, 3}

The pipeline is fit once, offline, by `phase_detector/fit.py` (or
`scripts/fit_phase_detector.py`). At runtime we just project + nearest-cluster.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from ..utils.obs import obs_to_state_action_features


class PCAPhaseDetector:
    def __init__(self, ckpt_dir: str | Path):
        ckpt_dir = Path(ckpt_dir)
        self.pca = joblib.load(ckpt_dir / "pca.pkl")
        self.km = joblib.load(ckpt_dir / "kmeans.pkl")
        with (ckpt_dir / "phase_map.json").open() as f:
            self.phase_map = {int(k): int(v) for k, v in json.load(f).items()}

    # ───────────────────────── raw-vector API ─────────────────────────
    def predict_feature(self, feat_46: np.ndarray) -> int:
        """Predict a phase ID from a precomputed 46-D feature vector."""
        feat_46 = np.asarray(feat_46, dtype=np.float32).reshape(1, -1)
        cluster_id = int(self.km.predict(self.pca.transform(feat_46))[0])
        return self.phase_map[cluster_id]

    # ───────────────────────── obs-dict API ─────────────────────────
    def predict_from_obs(self, obs: dict, last_action) -> int:
        """Predict a phase ID from the raw env-obs dict + the last 23-D action."""
        feat = obs_to_state_action_features(obs, last_action)
        return self.predict_feature(feat)
