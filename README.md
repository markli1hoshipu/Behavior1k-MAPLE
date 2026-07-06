# behavior1k-mp

Hybrid motion-planner + X-VLA-policy data-collection pipeline for
BEHAVIOR-1K task 0 (`turning_on_radio`).

Task 0 is decomposed into 4 phases — **navigate → pick up → press → put down** —
and each phase chooses its executor (X-VLA policy via `WebsocketPolicy`, or
OmniGibson `StarterSemanticActionPrimitives`) independently. Phase boundaries
are detected by a PCA + KMeans classifier trained offline on the 200 HF demos.

## Layout

```
behavior1k_mp/
├── actions/          # one Action class per task phase
├── phase_detector/   # PCA + KMeans + timestamp-ranked phase mapping
├── orchestrator.py   # state machine over Actions
├── hybrid_policy.py  # LocalPolicy subclass — plugs into eval_data_gen.py
└── utils/            # obs helpers, state extraction, debug logger
configs/policy/       # Hydra config for the hybrid policy
scripts/              # fit / smoke-test / collection entrypoints
slurm/                # SLURM templates
```

## Quick start

```bash
# 1) Install (editable) so other code can `import behavior1k_mp`
pip install -e /shared_work/behavior1k-mp

# 2) Fit the phase detector on the 200 HF demos (one-time, ~1 min)
python /shared_work/behavior1k-mp/scripts/fit_phase_detector.py

# 3) Start X-VLA inference server (in another shell or sbatch)
python /shared_work/behavior1k-xvla/behavior1k_training/deploy_b1k.py \
    --model_path /shared_work/behavior1k-xvla/checkpoints/v20/task0-60k \
    --port 8765

# 4) Smoke-test one instance
python /shared_work/behavior1k-mp/scripts/smoke_test_one_instance.py \
    task=turning_on_radio instance_id=301 policy=hybrid_mp model.vla_port=8765
```

## Status

- **Phase 0 (navigate)**: X-VLA policy, exit via PCA + (stubbed) geometric check
- **Phase 1 (pick)**: `StarterSemanticActionPrimitives.GRASP(radio)`
- **Phase 2 (press)**: `StarterSemanticActionPrimitives.TOGGLE_ON(radio)`
- **Phase 3 (put-down)**: stub (immediate done — `turning_on_radio` goal is `(toggled_on radio)`)

The per-phase `_geometric_done()` is currently `return True`; user fills in
distance / grasp / toggle checks as we observe real rollouts.
