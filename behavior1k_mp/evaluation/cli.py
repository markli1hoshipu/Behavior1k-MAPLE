"""maple eval — leaderboard rollout CLI.

Scaffold. Intended flow:
  1. Launch the X-VLA server (or use an existing one).
  2. Run BEHAVIOR-1K challenge rollouts via OmniGibson's eval_data_gen.py
     (the metric-only variant, not eval_data_gen_par_save_all.py).
  3. Aggregate q_scores and per-instance outcomes.
  4. Write a summary JSON to `evaluation/logs/`.
  5. Optionally package the challenge submission archive.

Usage sketch:
    maple eval \\
        --task turning_on_radio \\
        --ckpt /path/to/xvla/step_14000 \\
        --n-trials 100 \\
        --instances 1-1000 \\
        --package
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="maple eval", description=__doc__)
    p.add_argument("--task", required=True,
                   help="Task slug matching a behavior1k_mp.tasks.<name> module.")
    p.add_argument("--ckpt", required=True,
                   help="Path to the checkpoint being evaluated.")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--instances", default="1-1000",
                   help="Instance id spec: comma list or 'lo-hi' range.")
    p.add_argument("--package", action="store_true",
                   help="After the rollout, zip into a leaderboard submission archive.")
    p.add_argument("--log-dir", type=Path, default=LOG_DIR,
                   help="Where to write the per-run summary JSON.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the OmniGibson invocation without launching it.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(
        f"[maple eval] task={args.task}  ckpt={args.ckpt}  n_trials={args.n_trials}",
        file=sys.stderr,
    )
    print(
        "[maple eval] NOTE: scaffold only. Wire this to the OmniGibson "
        "eval_data_gen.py invocation and the challenge archive packager.",
        file=sys.stderr,
    )
    if args.dry_run:
        return 0
    return 1  # not implemented yet


if __name__ == "__main__":
    sys.exit(main())
