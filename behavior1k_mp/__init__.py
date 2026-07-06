"""behavior1k_mp (MAPLE) — hybrid motion-planner + X-VLA policy pipeline for BEHAVIOR-1K.

Layout:
  core/           task-agnostic infrastructure (orchestrator, IK, phase detector, ...)
  tasks/<name>/   per-task Action list, checkpoints, and HF annotation config
  collect/        data-collection driver + CLI
  evaluation/     leaderboard rollout driver + CLI
"""
__version__ = "0.1.0"
