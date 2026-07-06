# Adding a new task

Every task lives in its own submodule under `behavior1k_mp/tasks/<task_name>/`. The task module is the only thing the orchestrator + `HybridPolicy` need to run your task — no changes to `core/` should be necessary.

## Minimum shape

```
behavior1k_mp/tasks/<task_name>/
├── __init__.py            required — exports task attributes
├── actions/
│   ├── __init__.py        re-exports the Action classes
│   └── <phase>.py         one file per phase
└── checkpoints/
    ├── kmeans.pkl         PCA phase-detector artifacts
    ├── pca.pkl
    ├── phase_map.json
    └── <library.pkl>      any retrieval libraries the actions load
```

## Required module-level attributes

The task's `__init__.py` MUST export:

| Name | Type | Purpose |
|---|---|---|
| `TASK_NAME` | `str` | Short slug matching the folder name |
| `TASK_DISPLAY_NAME` | `str` | Human-readable ("turning on radio") |
| `TARGET_OBJECT_SCOPE_NAME` | `str` | BDDL object-scope key targeted by Actions |
| `CHECKPOINT_DIR` | `pathlib.Path` | Root of this task's checkpoints |
| `SLOT_TO_SKILL_IDX` | `dict[str, int]` | Orchestrator slot name → HF skill idx (0..3) |
| `SKILL_TEMPLATES` | `list[dict]` | One HF skill-annotation template per skill (4 entries) |
| `build_actions(env, robot, target_obj, vla) -> list[Action]` | callable | Return the ordered Action list |

See `behavior1k_mp/tasks/turning_on_radio/__init__.py` as a reference.

## Action subclass contract

Every `Action` subclass must set:

```python
class YourAction(Action):
    name: str              = "your_action_name"       # matches SLOT_TO_SKILL_IDX key
    executor: Executor     = Executor.POLICY | MP | NOOP
    expected_phase_id: int = 0..3   # PCA-detected phase this action lives in

    def forward(self, obs) -> torch.Tensor: ...   # returns 23-D action
    def _geometric_done(self, obs) -> bool: ...   # optional exit check
```

`is_done(obs, phase_det)` is inherited from `Action` and returns True once the PCA detector has reported a *later* phase for `pca_debounce_steps` consecutive frames AND `_geometric_done(obs)` is True.

## Trained artifacts

Every task needs its own **PCA + KMeans phase detector** trained on that task's demos. Use `scripts/fit_phase_detector.py --task <name>` (or set `--ckpt-dir` explicitly) — it writes `kmeans.pkl`, `pca.pkl`, and `phase_map.json` into your task's `checkpoints/`.

Retrieval libraries (used by the pick-approach and press-replay actions) are also per-task. Use `scripts/build_pick_library.py` and `scripts/build_press_libraries.py`.

## Register your task

There's no explicit registration step — the task registry (`behavior1k_mp/tasks/__init__.py`) uses `importlib` to load `behavior1k_mp.tasks.<name>` on demand. Just make sure your folder is inside `behavior1k_mp/tasks/` and installable via `pip install -e .`.

## Verify

```bash
# Instantiation smoke test
python -c "from behavior1k_mp.tasks import load_task; t = load_task('<name>'); print(t.TASK_NAME, t.CHECKPOINT_DIR)"

# End-to-end (dry run)
maple collect --task <name> --instances 0 --dry-run
```
