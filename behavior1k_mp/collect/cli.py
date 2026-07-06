"""maple collect — data-collection CLI.

Thin wrapper that shells out to OmniGibson's `eval_data_gen_par_save_all.py`
with the right Hydra overrides. Intended to be invoked either directly on a
worker node or via `slurm/collect.sbatch`.

This is a scaffold: current data collection is still driven end-to-end by
`slurm/collect.sbatch`. Migration path: move the SLURM script's OmniGibson
invocation into `run()` here, then let the sbatch script call
`maple collect ...` instead of hardcoding paths.

Usage sketch:
    maple collect \\
        --task turning_on_radio \\
        --instances 301-700 \\
        --n-trials-per-instance 5 \\
        --vla-host 127.0.0.1 --vla-port 8765 \\
        --output-root /shared_work/DATASETS/behavior-1k-mp-collected
"""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="maple collect", description=__doc__)
    p.add_argument("--task", required=True,
                   help="Task slug matching a behavior1k_mp.tasks.<name> module.")
    p.add_argument("--instances", required=True,
                   help="Instance id spec: comma list or 'lo-hi' range.")
    p.add_argument("--n-trials-per-instance", type=int, default=5)
    p.add_argument("--vla-host", default="127.0.0.1")
    p.add_argument("--vla-port", type=int, default=8765)
    p.add_argument("--output-root",
                   default="/shared_work/DATASETS/behavior-1k-mp-collected")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the OmniGibson invocation without launching it.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(
        f"[maple collect] task={args.task}  instances={args.instances}  "
        f"n_trials={args.n_trials_per_instance}",
        file=sys.stderr,
    )
    print(
        "[maple collect] NOTE: scaffold only. Wire this to the OmniGibson "
        "eval_data_gen_par_save_all.py invocation (see slurm/collect.sbatch).",
        file=sys.stderr,
    )
    if args.dry_run:
        return 0
    return 1  # not implemented yet


if __name__ == "__main__":
    sys.exit(main())
