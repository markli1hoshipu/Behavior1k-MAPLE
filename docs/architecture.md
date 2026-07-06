# Architecture

## Two use-modes over one core

The repo supports two runtime modes that share the entire policy stack:

| Mode | Entry point | Writes | Purpose |
|---|---|---|---|
| **Data collection** | `maple collect ...` (or `slurm/collect.sbatch`) | parquet + hdf5 + annotations + phase_segments + videos | Produce training data |
| **Evaluation** | `maple eval ...` (or `slurm/eval_*.sbatch`) | metrics JSON + submission archive | Leaderboard rollout |

Both modes instantiate the same `HybridPolicy` with the same task module, orchestrator, phase detector, and Action library. Only the outer driver differs.

## Runtime pipeline (single episode)

```
┌───────────────────────────────────────────────────────────────────┐
│  SLURM job launcher                                               │
│    1. Start VLA server (X-VLA or openpi) in background            │
│    2. wait_for_port <VLA_PORT>                                    │
│    3. Launch OmniGibson eval_data_gen*.py with policy=hybrid_mp   │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  OmniGibson evaluator                                             │
│    • builds env, calls policy.attach_env(env)                     │
│    • per step: obs = env.step(a);  a = policy.forward(obs)        │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  HybridPolicy  (behavior1k_mp/core/hybrid_policy.py)              │
│    • loads task module (behavior1k_mp/tasks/<name>/)              │
│    • task.build_actions(env, robot, target_obj, vla) → [Action]   │
│    • wires up Orchestrator + PCAPhaseDetector                     │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  Orchestrator  (behavior1k_mp/core/orchestrator.py)               │
│    • advances the current Action.forward(obs)                     │
│    • bridges phase-to-phase transitions (utils/action_bridge.py)  │
│    • exits phase when Action.is_done(obs, phase_det) returns True │
│    • logs ENTER / EXIT lines with orch step numbers               │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  Action.forward(obs)                                              │
│    Executor = POLICY  → returns VLA(obs)                          │
│    Executor = MP      → returns hand-crafted / IK / retrieval     │
│    Executor = NOOP    → returns held pose                         │
└───────────────────────────────────────────────────────────────────┘
```

## Where things live

- **Core policy stack** — `behavior1k_mp/core/`
    - `orchestrator.py` : phase state machine, per-step ENTER/EXIT logging
    - `hybrid_policy.py` : OmniGibson `LocalPolicy` wrapper, task-agnostic
    - `phase_detector/` : PCA + KMeans + timestamp ranking (task-agnostic)
    - `ik/`, `grasp/` : R1Pro IK + vertical-descent waypoints
    - `pick_library/` : nearest-neighbor retrieval infrastructure
    - `utils/` : obs slicer, action-bridge interpolation, pose helpers

- **Per-task modules** — `behavior1k_mp/tasks/<task_name>/`
    - `__init__.py` : registers task attributes and `build_actions(...)`
    - `actions/` : concrete `Action` subclasses (one per phase)
    - `checkpoints/` : trained PCA + KMeans + retrieval libraries for this task
    - `config.yaml` : Hydra config override (optional)

- **Drivers** — `behavior1k_mp/{collect,evaluation}/`
    - Argparse CLIs called via the top-level `maple` entrypoint
    - `evaluation/logs/` : per-rollout summary JSONs

## HF skill annotation flow

For every collected episode, `HybridPolicy.build_skill_annotation()` bucket-folds the orchestrator's sub-phase spans into the HF challenge's 4-skill schema using `tasks/<name>/SLOT_TO_SKILL_IDX` and `SKILL_TEMPLATES`. The output file mirrors the official annotations format used by the BEHAVIOR-1K leaderboard.

Complementary richer view: `phase_segments/task-XXXX/episode_NNNNNNNN_orchestrator_phases.json` (see the [MP-collected dataset README](https://huggingface.co/datasets/Hoshipu/behavior-1k-mp-collected-turning-on-radio)) preserves the full 7-slot trace + IK metadata.
