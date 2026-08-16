# CLAUDE.md — behavior1k_statuschecker

Context for whoever (human or Claude) picks this project back up.

## What this is

A LoRA-fine-tuned Qwen3.5-0.8B that reads one frame from a fixed head-mounted
camera + a task instruction, and predicts the robot's current subskill as
**"`<verb>` `<main object>`"** (e.g. "pick up from radio"), trained on
BEHAVIOR-1K's own `skill_annotation` ground truth - no manual prompt
engineering or LLM-written CoT needed once you actually fine-tune (prompt-only
approaches, see `legacy/phase1_prompt_engineering/`, plateaued well below what
LoRA achieves and are not worth repeating).

`pipeline/` is the current working pipeline. `legacy/` is prior iterations,
kept for reference - see its `README.md` note before assuming any `legacy/`
script still matches the current data format or model.

## Where the actual data lives

This repo ships code + `results/` only. The 70GB+ frame datasets, the 145GB HF
cache, and the LoRA checkpoints are NOT in git - they're on shared cluster
storage at `/shared_work/markhsp/behavior1k-statuschecker/` (same machine this
was developed on: `TRT-EAI-OLDLAB-1`, `192.168.50.103`). Every script in
`pipeline/` hardcodes `ROOT = "/shared_work/markhsp/behavior1k-statuschecker"`
at the top - if you're running from a fresh checkout elsewhere, that constant
is the first thing to change (or symlink your data dir to that path).

## How to prepare data and launch training

Data prep (from `pipeline/`, run on the cluster - needs `/shared_work` access):

```bash
# training set: episodes 1-20/task, all 100 tasks -> data_2026/
python3 download_extract.py 0 99

# eval set: episodes 21-40/task (fully disjoint from training) -> data_2026_eval/
EPISODE_START=20 N_EPISODES_PER_TASK=20 \
  OUT_ROOT=/shared_work/markhsp/behavior1k-statuschecker/data_2026_eval \
  python3 download_extract.py 0 99
```

Idempotent - safe to re-run, skips already-downloaded episodes. Pass a task
index range (`0 99` above) to shard the work; `EPISODE_START`/
`N_EPISODES_PER_TASK`/`OUT_ROOT` are the knobs for building a different
episode slice (e.g. a second held-out set, or more episodes/task).

Launch training via the sbatch launchers in `pipeline/sbatch/`:

```bash
# full 100 tasks, 2 epochs, single GPU (safest, slowest)
sbatch pipeline/sbatch/lora_train_v3.sbatch

# full 100 tasks, 2 epochs, 2-GPU single-node DDP (faster; see CLAUDE.md gotchas
# before trying multi-node)
sbatch pipeline/sbatch/lora_train_v3_multigpu.sbatch

# a task subset, N epochs - copy lora_train_v3_10task*.sbatch as a template and
# edit its TASK_SUBSET / EPOCHS_TO_RUN / ADAPTER_OUT env vars
```

`pipeline/lora_train_v3.py`'s relevant env-var overrides (all optional, sane
defaults if unset):

| Env var | Purpose |
|---|---|
| `TASK_SUBSET` | comma-separated `task-XXXX` list to train on fewer than all 100 tasks |
| `ADAPTER_OUT` | output path for the LoRA adapter (default `lora_adapter_v3`) |
| `HELDOUT_OUT` | where to write the held-out-episode manifest JSON |
| `EPOCHS_TO_RUN` / `EPOCH_OFFSET` | how many epochs to run, and what epoch number to start counting from |
| `RESUME_ADAPTER` | load this checkpoint instead of a fresh LoRA init - combine with `EPOCH_OFFSET`/`EPOCHS_TO_RUN` to continue training instead of restarting from scratch |

A checkpoint is saved after every epoch (`<ADAPTER_OUT>_epoch<N>`) plus a final
save at `<ADAPTER_OUT>` when all requested epochs finish.

## Key conventions

- **Target format**: `"<verb> <object>"`, built in `pipeline/download_extract.py`.
  Object ids in the raw dataset look like `coffee_table_koagbh_0` or `radio_89`
  - normalized to `coffee table` / `radio` by matching the *longest* prefix
    against the real BEHAVIOR category list
    (`/shared_work/BEHAVIOR-2026/datasets/behavior-1k-assets/metadata/categories.txt`),
    not a regex guess. A regex-only approach misfires on categories that happen
    to be 6 letters (e.g. it'll treat "candle" in "pillar_candle" as if it were
    a random model-id suffix).
- **Per-class frame cap** (60/class) is applied AFTER aggregating across all
  episodes for a task, not per-episode - otherwise long-duration classes like
  "move to" dominate the dataset.
- **Held-out eval**: `data_2026/` = episodes 1-20/task (training).
  `data_2026_eval/` = episodes 21-40/task (eval, fully disjoint - not just one
  held-out episode). Always eval against `data_2026_eval/`, and always build
  the model's prompt from the *training* set's `distinct_labels` (not the eval
  set's) - the model was only ever shown the training vocabulary, and an
  eval-only label the model never saw during training should count as a miss.
- **Model loading class**: `AutoModelForImageTextToText` /
  `Qwen3_5ForConditionalGeneration` for Qwen3.5 - not a guessed "AutoModelForMultimodalLM".

## Known gotchas (hit these once already, don't rediscover them)

- **SLURM venv portability**: a venv's python symlink pointing into
  `/home/<user>/.local/share/uv/python/...` is node-local and breaks silently
  on other compute nodes (`/home` isn't cluster-shared). Venvs must point at
  `/shared_work/.../uv_python/...` instead.
- **Cross-node NCCL hang between `trt-eai-oldlab-3` and `trt-eai-oldlab-4`
  specifically**: reproducible, deterministic collective timeout (identical
  NCCL seq number across 3 separate attempts, including on a fully idle node -
  ruled out resource contention). Root cause not fully diagnosed (py-spy
  needed sudo, wasn't available) but the pattern (one rank fully stalls while
  the other keeps issuing collectives) points to a specific training example
  stalling one rank's data loading, not a code bug. **Workaround**: pin
  multi-node jobs to `trt-eai-oldlab-3`+`5` (verified working, ran a full 8h
  training job successfully), or just use single-GPU for anything small enough
  to tolerate it (no NCCL at all, zero risk of this class of failure). See
  `legacy/multinode_debugging/`.
- **Checkpoint-save barrier bug** (fixed in `pipeline/lora_train_v3.py`): after
  each epoch, only rank 0 does `save_pretrained()` to disk; there must be an
  `accelerator.wait_for_everyone()` call *after* that save (not just before
  it), or non-main ranks can race into the next epoch's first gradient-sync
  collective before rank 0 is done writing, causing a watchdog timeout crash.
- **`accelerate launch` subprocess stdout can look stale** even though it's
  actually unbuffered (`-u` is applied automatically by torchrun) - if a job
  looks stuck with no new log lines, check actual GPU utilization
  (`srun --jobid=<id> --overlap nvidia-smi`) before assuming a hang.

## Results so far (see `results/*_summary.json` for full breakdowns)

| Model | Verb+object macro recall | Verb-only macro recall |
|---|---|---|
| Base Qwen3.5-0.8B (no LoRA) | 9.7% | 16.2% |
| LoRA v3, 100 tasks, 2 epochs | 62.2% | 71.5% |
| LoRA v3, 10-task subset, 4 epochs | 66.5% | 73.8% |

The 10-task/4-epoch ablation shows real headroom left in the 100-task/2-epoch
model - see TODO.md.

## How to report results

Every time results are reported (a new model, a new ablation, a checkpoint
comparison), produce two artifacts, not just a chat summary:

**1. A comparison CSV.** One row per (model, task) pair, columns:

```
model, task_chunk, task_name, n_classes, overall_acc, overall_macro,
skill_acc, skill_macro, obj_acc, obj_macro
```

- `overall_*` = verb+object combined (the primary metric) - from
  `results/lora_v3_<tag>_task-*.json` / `*_summary.json`.
- `skill_*` = verb-only, ignoring object correctness - from
  `compute_verb_only_metrics.py <tag>` /
  `results/lora_v3_<tag>_verbonly_summary.json`.
- `obj_*` = object-only, ignoring verb correctness (**not implemented yet** -
  needs a new `compute_object_only_metrics.py`, mirroring
  `compute_verb_only_metrics.py`'s approach: rescore the same saved
  predictions using an object-only match instead of a verb-only match, keyed
  off `target_to_object` built from each task's ground-truth `object` field
  the same way `target_to_verb` is built today).

Build the CSV by joining across every `*_summary.json` you want in the
comparison (e.g. `baseeval`, `epoch0eval`, `fulleval` for the 100-task run;
`10task_base`/`10task_2ep`/`10task_4ep` for the ablation) - one script that
takes a list of result tags and models, writes one CSV row per (tag, task).

**2. A PPT.** Should cover:

- Results tables (the CSV above, rendered as slides/charts) - overall trend
  across models/checkpoints, plus a per-task breakdown highlighting the best
  and worst tasks and why (class count, unseen-label frame count, etc.)
- Training configs used to produce each model in the comparison: base model,
  LoRA hyperparams (r, alpha, dropout, target modules), epochs, dataset size
  (n_tasks x n_episodes x per-class cap), GPU setup, wall-clock time
- Example input/output pairs - a frame + the exact prompt sent to the model +
  its prediction vs. ground truth, for at least one correct and one wrong
  example (the `demo/*.mp4` videos and the still-frame renders in
  `pipeline/build_demo_video_v3.py` are the source material for these - either
  embed a still frame per example or link/embed the actual video)
- Embed or link the demo videos themselves, not just a screenshot - they're
  the clearest way to show a non-technical reviewer what the model is actually
  doing frame-by-frame
