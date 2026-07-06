#!/usr/bin/env python3
"""Run the hybrid pipeline on a single task-0 instance, end-to-end.

This is a thin wrapper that defers to BEHAVIOR-1K's `omnigibson.learning.eval_data_gen`.
We just override the policy config and pass through the task / instance args.

Prerequisites (in two separate shells / sbatch jobs):
  1. X-VLA inference server up on the specified port:
        python /shared_work/behavior1k-xvla/behavior1k_training/deploy_b1k.py \
            --model_path /shared_work/behavior1k-xvla/checkpoints/v20/task0-60k \
            --port 8765
  2. The `hybrid_mp.yaml` config file present at
        /shared_work/BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs/policy/hybrid_mp.yaml
     (symlink or copy from /shared_work/behavior1k-mp/configs/policy/hybrid_mp.yaml)

Usage:
    conda activate behavior
    cd /shared_work/BEHAVIOR-1K
    python /shared_work/behavior1k-mp/scripts/smoke_test_one_instance.py \
        task.name=turning_on_radio instance_id=301 \
        policy=hybrid_mp \
        model.vla_port=8765 \
        log_path=/shared_work/logs/behavior_datacollect/test_mp_$(date +%s)
"""
import os
import sys

# Ensure behavior1k_mp is importable (it's pip-installable, but in case not).
sys.path.insert(0, "/shared_work/behavior1k-mp")

# Delegate to the upstream entrypoint — it already handles Hydra config + env setup
# + recording. We just need the patched `eval_data_gen.py` (which calls
# `policy.attach_env(env)` if available) on the import path.
from omnigibson.learning import eval_data_gen as _entry  # noqa: F401

if __name__ == "__main__":
    # eval_data_gen.py is decorated with @hydra.main; calling its `main` invokes hydra.
    _entry.main()
