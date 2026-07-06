# MAPLE docs

Lightweight design + results documentation for the BEHAVIOR-1K MAPLE repo.

## Contents

- **[architecture.md](./architecture.md)** — end-to-end dataflow: SLURM → X-VLA/openpi server → OmniGibson → HybridPolicy → writer.
- **[task_authoring.md](./task_authoring.md)** — how to add a new task under `behavior1k_mp/tasks/<name>/`.
- **[fsm/](./fsm/)** — orchestrator FSM diagrams (Mermaid). GitHub renders these inline.
    - [turning_on_radio.mmd](./fsm/turning_on_radio.mmd)
- **[results/](./results/)** — per-milestone result reports (success rates, PCA plots, video links).

## Repo layout at a glance

```
behavior1k_mp/
├── core/                  task-agnostic infrastructure
├── tasks/<name>/          per-task actions + checkpoints + config
├── collect/               data-collection CLI driver
└── evaluation/            leaderboard rollout CLI driver
configs/                   Hydra defaults
slurm/                     sbatch scripts + shared_env.sh
docs/                      you are here
scripts/                   dev-time tools (fit_phase_detector, build_pick_library, ...)
third_party/               r1pro.urdf, zexternal_utils.py (Mark's IK)
openpi -> /shared_work/openpi   (symlink, gitignored)
```

## Adding a new task

See [task_authoring.md](./task_authoring.md). Short version:

1. `mkdir behavior1k_mp/tasks/<new_task>`
2. Write `__init__.py` exporting: `TASK_NAME`, `TASK_DISPLAY_NAME`, `TARGET_OBJECT_SCOPE_NAME`, `CHECKPOINT_DIR`, `SLOT_TO_SKILL_IDX`, `SKILL_TEMPLATES`, `build_actions(env, robot, target_obj, vla)`.
3. Drop your `Action` subclasses into `actions/` and your PCA/retrieval libraries into `checkpoints/`.
4. `maple collect --task <new_task> ...` and `maple eval --task <new_task> ...` should now work.
