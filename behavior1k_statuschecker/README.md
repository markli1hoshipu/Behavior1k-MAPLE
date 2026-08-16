# behavior1k_statuschecker

VLM-based per-frame subskill annotator for BEHAVIOR-2026: fine-tunes Qwen3.5-0.8B
(LoRA) to read a single frame from a fixed head-mounted camera and predict the
robot's current subskill as **verb + main object** (e.g. "pick up from radio"),
using BEHAVIOR-1K's own `skill_annotation` ground truth as training labels - no
manual prompt engineering or LLM-written CoT required at this scale.

Conceptually this is an alternative to `behavior1k_motionplanner/core/phase_detector/`
(which segments an episode into phases via PCA + KMeans on proprioception) - same
goal of "what phase/skill is happening right now", different input modality
(a single RGB frame + task instruction, via a VLM) and different backend (a
fine-tuned language model instead of an unsupervised clustering pipeline).

## Layout

```
pipeline_2026/download_extract.py   # pulls BEHAVIOR-2026 (behavior-1k/2026-challenge-demos,
                                     #   LeRobot v3 packed-video format) + extracts 1fps frames,
                                     #   normalizes object ids against the real BEHAVIOR category
                                     #   list, builds "<verb> <object>" targets
lora_train_v3.py                    # LoRA fine-tune (accelerate, single- or multi-GPU DDP),
                                     #   supports RESUME_ADAPTER/EPOCH_OFFSET/EPOCHS_TO_RUN/
                                     #   TASK_SUBSET/ADAPTER_OUT env overrides
eval_lora_v3_fulleval.py            # full-eval-set scoring (macro recall), supports sharding
                                     #   for a SLURM job array, NO_ADAPTER=1 for base-model eval
aggregate_fulleval_summary.py       # combine sharded eval results into one summary
compute_verb_only_metrics.py        # rescore existing predictions on verb alone (ignore object)
build_demo_video_v3.py              # green-text-on-black demo videos (frame + prompt + prediction)
*.sbatch                            # SLURM launchers for the above
results/                            # per-task JSON results + macro-recall summaries (this is the
                                     #   actual output of the project - safe/small enough for git)
```

## Data & checkpoints (NOT in this repo - too large for git)

The raw downloaded frames (`data_2026/`, `data_2026_eval/`, ~70GB each), the HF
cache (`hf_cache/`, ~145GB), and the LoRA checkpoints (`lora_adapter_v3*/`, ~25MB
each) live on shared storage, not in git:

```
/shared_work/markhsp/behavior1k-statuschecker/
```

Re-running `pipeline_2026/download_extract.py` regenerates `data_2026*/` from
scratch (idempotent, skips already-downloaded episodes) if you're working from a
fresh checkout without shared-storage access.

## Headline results (verb+object macro recall, full 100-task eval set)

| Model | Avg macro recall | Tasks >=70% |
|---|---|---|
| Base Qwen3.5-0.8B (no LoRA) | 9.7% | 0/100 |
| LoRA v3, 2 epochs, 100 tasks | 62.2% | 30/100 |

Verb-only (ignoring object correctness): base 16.2%, LoRA v3 71.5%.

See `results/lora_v3_fulleval_summary.json` (and sibling `*_summary.json` /
`*_verbonly_summary.json` files) for the full per-task breakdown, including the
10-task/4-epoch ablation (`10task_base` / `10task_2ep` / `10task_4ep` tags).
