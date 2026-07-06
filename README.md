# MAPLE — BEHAVIOR-1K hybrid MP + VLA pipeline

Dual-use repo for BEHAVIOR-1K:

- **Data collection** — hybrid motion-planner + X-VLA policy, writes parquet + hdf5 + annotations + phase_segments per episode.
- **Leaderboard evaluation** — metric-only rollouts and challenge-archive packaging.

Both modes share the same policy stack (orchestrator, task-specific Action library, PCA phase detector, retrieval libraries, IK).

## Layout

```
behavior1k_mp/
├── core/                      # task-agnostic infrastructure
│   ├── action.py              #   Action, Executor
│   ├── orchestrator.py        #   phase state machine
│   ├── hybrid_policy.py       #   OmniGibson LocalPolicy wrapper
│   ├── phase_detector/        #   PCA + KMeans + timestamp ranking
│   ├── ik/  grasp/            #   Mark's IK, vertical-descent waypoints
│   ├── pick_library/          #   retrieval library builder + lookup
│   └── utils/                 #   obs slicer, action bridge, pose helpers
├── tasks/
│   ├── __init__.py            # registry: load_task(name)
│   └── turning_on_radio/
│       ├── __init__.py        #   build_actions, SLOT_TO_SKILL_IDX, CHECKPOINT_DIR
│       ├── actions/           #   7 Action subclasses
│       └── checkpoints/       #   kmeans.pkl, pca.pkl, pick_library/, press_modes/
├── collect/                   # data-collection CLI driver
├── evaluation/                # leaderboard rollout CLI driver
│   └── logs/                  # per-run summary JSONs
└── __main__.py                # maple CLI dispatcher
configs/                       # Hydra defaults (policy=hybrid_mp)
docs/                          # architecture, task_authoring, FSM, results
scripts/                       # dev-time tools (fit_phase_detector, build_pick_library, ...)
slurm/                         # sbatch + shared_env.sh
third_party/                   # r1pro.urdf, zexternal_utils.py (Mark's IK)
openpi -> /shared_work/openpi  # symlink, gitignored
```

## Quick start

```bash
# 1) Install (editable) so other code can `import behavior1k_mp`
pip install -e /shared_work/behavior1k-mp

# 2) (Optional) refit the phase detector for turning_on_radio
python scripts/fit_phase_detector.py

# 3) Data collection — SLURM launcher
sbatch slurm/collect.sbatch turning_on_radio 301

# 4) Evaluation
sbatch slurm/eval_xvla.sbatch turning_on_radio 242

# 5) The unified CLI (scaffold; wire-up in progress):
maple collect --task turning_on_radio --instances 301-700
maple eval    --task turning_on_radio --ckpt /path/to/ckpt
```

## Adding a new task

See [docs/task_authoring.md](docs/task_authoring.md). Short version: drop a new folder into `behavior1k_mp/tasks/<name>/` that exports `TASK_NAME`, `TARGET_OBJECT_SCOPE_NAME`, `CHECKPOINT_DIR`, `SLOT_TO_SKILL_IDX`, `SKILL_TEMPLATES`, and `build_actions(env, robot, target_obj, vla)`.

## Docs

- [docs/architecture.md](docs/architecture.md) — end-to-end dataflow
- [docs/task_authoring.md](docs/task_authoring.md) — how to add a task
- [docs/fsm/turning_on_radio.mmd](docs/fsm/turning_on_radio.mmd) — orchestrator FSM (Mermaid, renders inline on GitHub)
- [docs/results/](docs/results/) — milestone result reports
