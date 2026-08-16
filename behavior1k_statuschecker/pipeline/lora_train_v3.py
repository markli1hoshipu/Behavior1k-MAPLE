"""LoRA fine-tune Qwen3.5-0.8B on the full BEHAVIOR-2026 challenge dataset: all 100
tasks, 20 episodes/task (data_2026/task-XXXX/ground_truth.json, extracted by
pipeline_2026/download_extract.py from the packed LeRobot-v3 videos).

Same classification-only objective as v2 (lora_train_v2.py): no Opus CoT, no
per-task engineered rules either this time (that doesn't scale to 100 tasks) - the
prompt just lists each task's own raw distinct labels plainly. Target is "verb-norm":
"<verb> <main object>" (e.g. "pick up from radio"), not just the bare verb - see
pipeline_2026/download_extract.py for how the object is derived/normalized.

One episode per task (the last one, sorted) is held out for a genuine held-out eval.

Multi-GPU via `accelerate` (data-parallel): the full example list is built
IDENTICALLY on every process (same deterministic file-read order + a fixed shuffle
seed), then each process takes its own disjoint stride-slice as its local shard - no
DistributedSampler/DataLoader needed since we're already doing a manual per-example
loop. Gradients sync on every micro-step's backward (no accelerator.accumulate()
no_sync optimization) - simpler and correct; the extra all-reduces are cheap since
this model is small and LoRA-only gradients are tiny.

Launch:
  single GPU:  python3 lora_train_v3.py
  multi GPU:   accelerate launch --multi_gpu --num_processes=N lora_train_v3.py
"""
import glob
import json
import os
import random

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

ROOT = "/shared_work/markhsp/behavior1k-statuschecker"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
ADAPTER_OUT = os.environ.get("ADAPTER_OUT", f"{ROOT}/lora_adapter_v3")
DATA_ROOT = f"{ROOT}/data_2026"
SHARD_SEED = 1234  # fixed so every process builds/shuffles the identical example list before slicing
# optional comma-separated task-chunk list (e.g. "task-0004,task-0010,...") to train on
# only a subset of the 100 tasks - for smaller/faster experiments
TASK_SUBSET = os.environ.get("TASK_SUBSET")
TASK_SUBSET = set(TASK_SUBSET.split(",")) if TASK_SUBSET else None

# Resume support: if RESUME_ADAPTER is set, load that checkpoint instead of starting
# a fresh LoRA adapter, and run EPOCHS_TO_RUN epochs starting at EPOCH_OFFSET (for
# checkpoint naming/shuffle-seed continuity) rather than always starting at epoch 0.
# Used when a training job gets cut off by the SLURM time limit mid-run - resume from
# the last completed epoch's checkpoint instead of redoing already-finished epochs.
RESUME_ADAPTER = os.environ.get("RESUME_ADAPTER")
EPOCH_OFFSET = int(os.environ.get("EPOCH_OFFSET", "0"))
EPOCHS_TO_RUN = int(os.environ.get("EPOCHS_TO_RUN", "2"))

SYSTEM_PROMPT = (
    "You are annotating individual frames from a fixed head-mounted camera on a "
    "mobile manipulator robot completing a household task. Classify the robot's "
    "current subskill using ONLY the visual evidence in this single frame."
)


def build_user_prompt(task_instruction, labels):
    label_list = "\n".join(f"{i+1}. {lab}" for i, lab in enumerate(labels))
    return (
        f'Task: "{task_instruction}"\n'
        f"The robot's subskill right now is always exactly one of:\n{label_list}\n"
        "Look only at what is visible in this frame. Respond with ONLY the label, "
        "nothing else."
    )


def load_examples():
    """Returns (examples, held_out_episode dict[task_chunk] -> episode id).
    Deterministic order across processes: sorted glob + in-file JSON order."""
    examples = []
    held_out_episode = {}
    task_dirs = sorted(glob.glob(f"{DATA_ROOT}/task-*/ground_truth.json"))
    for gt_path in task_dirs:
        with open(gt_path) as f:
            gt = json.load(f)
        task_chunk = gt["task_chunk"]
        if TASK_SUBSET and task_chunk not in TASK_SUBSET:
            continue
        episodes_sorted = sorted(gt["episodes"])
        if len(episodes_sorted) < 2:
            print(f"{task_chunk}: only {len(episodes_sorted)} episode(s), skipping (need >=2 for holdout)")
            continue
        held_out_ep = episodes_sorted[-1]
        held_out_episode[task_chunk] = held_out_ep

        user_prompt = build_user_prompt(gt["task_instruction"], gt["distinct_labels"])
        n_train = 0
        for item in gt["per_second"]:
            if item["episode"] == held_out_ep:
                continue
            examples.append({
                "task_chunk": task_chunk, "sec": item["sec"], "episode": item["episode"],
                "frame_path": item["frame_path"], "user_prompt": user_prompt,
                "target": item["target"],
            })
            n_train += 1
        print(f"{task_chunk} ({gt['task_name']}): {n_train} training examples (held out {held_out_ep})")

    return examples, held_out_episode


def build_training_tensors(processor, device, example):
    img = Image.open(example["frame_path"]).convert("RGB")
    messages_prompt_only = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": example["user_prompt"]},
        ]},
    ]
    prompt_inputs = processor.apply_chat_template(
        messages_prompt_only, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    prompt_len = prompt_inputs["input_ids"].shape[-1]

    messages_full = messages_prompt_only + [
        {"role": "assistant", "content": [{"type": "text", "text": example["target"]}]}
    ]
    full_inputs = processor.apply_chat_template(
        messages_full, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt"
    )

    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100  # loss only on the label tokens

    full_inputs["labels"] = labels
    return {k: v.to(device) for k, v in full_inputs.items()}


def main():
    accelerator = Accelerator()
    set_seed(SHARD_SEED)
    is_main = accelerator.is_main_process

    if is_main:
        print(f"World size: {accelerator.num_processes}  device: {accelerator.device}")
        print("Loading examples...")
    examples, held_out_episode = load_examples()
    if is_main:
        print(f"Total training examples: {len(examples)} across {len(held_out_episode)} tasks")
        heldout_out = os.environ.get("HELDOUT_OUT", f"{ROOT}/results/lora_v3_heldout_episodes.json")
        with open(heldout_out, "w") as f:
            json.dump(held_out_episode, f, indent=2)

    # Every process builds the identical list above, then shuffles it identically
    # (same seed) before taking its own disjoint stride-slice as a local shard.
    rng = random.Random(SHARD_SEED)
    rng.shuffle(examples)
    local_examples = examples[accelerator.process_index::accelerator.num_processes]
    print(f"[rank {accelerator.process_index}] local shard: {len(local_examples)} examples")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16)

    if RESUME_ADAPTER:
        if is_main:
            print(f"Resuming from checkpoint: {RESUME_ADAPTER}")
        model = PeftModel.from_pretrained(model, RESUME_ADAPTER, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    model, optimizer = accelerator.prepare(model, optimizer)

    GRAD_ACCUM = 8
    step = EPOCH_OFFSET * (len(local_examples) // GRAD_ACCUM)
    for epoch in range(EPOCH_OFFSET, EPOCH_OFFSET + EPOCHS_TO_RUN):
        rng2 = random.Random(SHARD_SEED + 1000 * (epoch + 1) + accelerator.process_index)
        rng2.shuffle(local_examples)
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()
        for i, ex in enumerate(local_examples):
            inputs = build_training_tensors(processor, accelerator.device, ex)
            out = model(**inputs)
            loss = out.loss / GRAD_ACCUM
            accelerator.backward(loss)
            total_loss += out.loss.item()
            n_steps += 1
            if (i + 1) % GRAD_ACCUM == 0 or i == len(local_examples) - 1:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if is_main and step % 50 == 0:
                    print(f"epoch {epoch} step {step}/{len(local_examples)//GRAD_ACCUM} avg_loss={total_loss/n_steps:.4f}")
        if is_main:
            print(f"=== epoch {epoch} done (rank 0 local avg_loss={total_loss/n_steps:.4f}) ===")

        accelerator.wait_for_everyone()
        if is_main:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(f"{ADAPTER_OUT}_epoch{epoch}")
            print(f"Saved checkpoint to {ADAPTER_OUT}_epoch{epoch}")
        # non-main ranks must not race ahead into the next epoch's collective ops
        # (gradient sync on first backward) while rank 0 is still writing to disk -
        # that mismatch is what caused a 600s NCCL watchdog timeout crash before.
        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if is_main:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(ADAPTER_OUT)
        print(f"Saved final LoRA adapter to {ADAPTER_OUT}")


if __name__ == "__main__":
    main()
