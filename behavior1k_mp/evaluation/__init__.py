"""Leaderboard evaluation driver.

Runs metric-only rollouts (no full trajectory recording) and packages the
resulting run into a challenge-submission archive.

Entry point: `maple eval --task <name> --ckpt <path> [...]`.

Per-run summary logs (one JSON per rollout) land under
`behavior1k_mp/evaluation/logs/`.
"""
