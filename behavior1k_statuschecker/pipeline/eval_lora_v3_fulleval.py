"""Evaluate lora_adapter_v3 on the FULL disjoint eval set (data_2026_eval/, episodes
21-40/task - none of these episodes were used in training, unlike the single
held-out-episode check in eval_lora_v3_heldout.py). Much larger and more reliable
signal: ~744 eval examples/task on average vs. ~1 held-out episode's worth.

The prompt's label list must match what the model was actually FINE-TUNED on - i.e.
the TRAINING set's distinct_labels (data_2026/task-XXXX/ground_truth.json), not the
eval set's own distinct_labels (a different sample of episodes could contain a
slightly different set of verb+object combos). Any eval-only label the model never
saw during training is counted as an automatic miss for that frame, which is fair.
"""
import glob
import json
import os

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

ROOT = "/shared_work/markhsp/behavior1k-statuschecker"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
# override to point at an intermediate checkpoint, e.g. lora_adapter_v3_epoch0
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", f"{ROOT}/lora_adapter_v3")
# distinguishes output filenames so an epoch0 eval doesn't get overwritten by the
# final-adapter eval later (or vice versa)
RESULT_TAG = os.environ.get("RESULT_TAG", "fulleval")
RESULTS_DIR = f"{ROOT}/results"
TRAIN_DATA_ROOT = f"{ROOT}/data_2026"
EVAL_DATA_ROOT = f"{ROOT}/data_2026_eval"

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


def parse_answer(raw_text, labels):
    text = raw_text.strip().lower().strip('."\'')
    for lab in labels:
        if text == lab.lower():
            return lab
    for lab in sorted(labels, key=len, reverse=True):
        if lab.lower() in text:
            return lab
    return None


def run_task(model, processor, task_chunk):
    with open(f"{TRAIN_DATA_ROOT}/{task_chunk}/ground_truth.json") as f:
        train_gt = json.load(f)
    eval_gt_path = f"{EVAL_DATA_ROOT}/{task_chunk}/ground_truth.json"
    import os
    if not os.path.exists(eval_gt_path):
        print(f"SKIP {task_chunk}: no eval data")
        return None
    with open(eval_gt_path) as f:
        eval_gt = json.load(f)

    train_labels = train_gt["distinct_labels"]  # what the model was fine-tuned on
    prompt = build_user_prompt(train_gt["task_instruction"], train_labels)
    eval_items = eval_gt["per_second"]

    predictions = []
    correct = 0
    unseen_label_count = 0
    for item in eval_items:
        img = Image.open(item["frame_path"]).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]},
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        raw = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        pred = parse_answer(raw, train_labels)

        gt_label = item["target"]
        if gt_label not in train_labels:
            unseen_label_count += 1
        is_correct = pred == gt_label
        correct += int(is_correct)
        predictions.append({
            "sec": item["sec"], "episode": item["episode"], "gt": gt_label, "pred": pred,
            "raw": raw.strip(), "correct": is_correct,
        })

    total = len(predictions)
    acc = correct / total if total else 0.0
    by_class = {}
    for p in predictions:
        by_class.setdefault(p["gt"], [0, 0])
        by_class[p["gt"]][1] += 1
        if p["correct"]:
            by_class[p["gt"]][0] += 1
    macro = sum(c / t for c, t in by_class.values()) / len(by_class) if by_class else 0.0

    print(f"=== {task_chunk} ({train_gt['task_name']}) FULL-EVAL (v3): acc={acc:.3f} ({correct}/{total})  "
          f"macro={macro:.3f}  n_classes={len(by_class)}  unseen_gt_labels={unseen_label_count} ===")

    with open(f"{RESULTS_DIR}/lora_v3_{RESULT_TAG}_{task_chunk}.json", "w") as f:
        json.dump({
            "task_chunk": task_chunk, "task_name": train_gt["task_name"],
            "accuracy": acc, "macro_recall": macro, "correct": correct, "total": total,
            "n_classes": len(by_class), "unseen_gt_label_frames": unseen_label_count,
            "predictions": predictions,
        }, f, indent=2)
    return acc, macro


def main():
    import sys
    task_dirs = sorted(glob.glob(f"{TRAIN_DATA_ROOT}/task-*/ground_truth.json"))
    task_chunks = [p.split("/")[-2] for p in task_dirs]

    task_subset_env = os.environ.get("TASK_SUBSET")
    if task_subset_env:
        subset = set(task_subset_env.split(","))
        task_chunks = [tc for tc in task_chunks if tc in subset]

    # optional [start_idx] [end_idx] (both inclusive) to run only a shard of the 100
    # tasks - lets this be split across a SLURM job array (1 GPU/task, independent,
    # no NCCL needed since each task's eval is fully separate). Results still land in
    # per-task JSON files; run with no args afterwards to just re-aggregate the summary.
    if len(sys.argv) >= 3:
        start_idx, end_idx = int(sys.argv[1]), int(sys.argv[2])
        task_chunks = task_chunks[start_idx:end_idx + 1]
        print(f"Running shard: tasks[{start_idx}:{end_idx+1}] = {task_chunks}")

    if not task_chunks:
        print("No tasks in this shard, nothing to do.")
        return

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0"
    )
    if os.environ.get("NO_ADAPTER"):
        model = base_model  # eval the raw untrained model, no LoRA
    else:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    for task_chunk in task_chunks:
        run_task(model, processor, task_chunk)

    print(f"\nShard done ({len(task_chunks)} tasks). Run aggregate_fulleval_summary.py "
          f"once all shards finish to build the combined summary.")


if __name__ == "__main__":
    main()
