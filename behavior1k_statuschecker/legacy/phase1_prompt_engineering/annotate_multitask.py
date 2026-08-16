"""Generic single-frame annotator across all 10 tasks, using Qwen3.5-2B. No per-task
hand-tuned visual criteria (unlike v14 for the radio task) - just the task instruction
and that episode's own distinct skill labels, to measure how the baseline approach
generalizes before investing in per-task refinement.
"""
import json
import os
import re

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

ROOT = "/shared_work/markhsp/behavior1k-statuschecker"
MODEL_ID = "Qwen/Qwen3.5-2B"
RESULTS_DIR = f"{ROOT}/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TASKS = {
    "task-0000": "Turn on the radio receiver that's on the table in the living room.",
    "task-0001": "Pick up the trash (soda cans) from the floor and put them in the trash can.",
    "task-0005": "Set the mousetraps near the sink and the toilet.",
    "task-0020": "Sort the vegetables from the baskets into the mixing bowls.",
    "task-0022": "Put the shoes on the shoe rack.",
    "task-0031": "Wash the two dusty boxing gloves in the washer until they are no longer covered with dust.",
    "task-0034": "Hang the poster on the wall nail.",
    "task-0040": "Make microwave popcorn: put the popcorn bag in the microwave and turn it on.",
    "task-0042": "Take the onion out of the sink, dice it on the chopping board, and put it in the bowl.",
    "task-0049": "Make a pizza: chop vegetables, add toppings to the dough, and bake it in the oven.",
}

SYSTEM_PROMPT = (
    "You are annotating individual frames from a fixed head-mounted camera on a "
    "mobile manipulator robot completing a household task. Classify the robot's "
    "current subskill using ONLY the visual evidence in this single frame."
)


def build_prompt(task_instruction, labels):
    label_list = ", ".join(f'"{l}"' for l in labels)
    return (
        f'Task: "{task_instruction}"\n'
        f"The robot's current subskill is always exactly one of: {label_list}.\n"
        "Look only at what is visible in this frame. Respond with EXACTLY one of "
        "those label strings, nothing else."
    )


def parse_label(raw_text, labels):
    text = raw_text.strip().lower().strip('."\'')
    for lab in labels:
        if text == lab.lower():
            return lab
    for lab in sorted(labels, key=len, reverse=True):  # longest match first
        if lab.lower() in text:
            return lab
    return None


def run_task(model, processor, task_chunk, task_instruction):
    with open(f"{ROOT}/multitask_data/{task_chunk}/ground_truth.json") as f:
        gt = json.load(f)
    labels = gt["distinct_labels"]
    prompt = build_prompt(task_instruction, labels)

    predictions = []
    correct = 0
    for item in gt["per_second"]:
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
            out = model.generate(**inputs, max_new_tokens=12, do_sample=False)
        raw = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        pred = parse_label(raw, labels)

        gt_label = item["skill_description"]
        is_correct = pred == gt_label
        correct += int(is_correct)
        predictions.append({
            "sec": item["sec"], "gt": gt_label, "pred": pred,
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

    print(f"\n=== {task_chunk} ({gt['task_name']}): acc={acc:.3f} ({correct}/{total})  macro={macro:.3f} ===")
    for k, (c, t) in sorted(by_class.items()):
        print(f"    {k:20s} {c}/{t}")

    with open(f"{RESULTS_DIR}/multitask_{task_chunk}.json", "w") as f:
        json.dump({
            "task_chunk": task_chunk, "task_name": gt["task_name"], "labels": labels,
            "accuracy": acc, "macro_recall": macro, "correct": correct, "total": total,
            "predictions": predictions,
        }, f, indent=2)
    return acc, macro


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0"
    )

    summary = []
    for task_chunk, instruction in TASKS.items():
        acc, macro = run_task(model, processor, task_chunk, instruction)
        summary.append((task_chunk, acc, macro))

    print("\n\n===== SUMMARY ACROSS ALL TASKS =====")
    print(f"{'task':12s} {'raw_acc':>8s} {'macro':>8s}")
    for tc, acc, macro in summary:
        print(f"{tc:12s} {acc:8.3f} {macro:8.3f}")
    avg_acc = sum(a for _, a, _ in summary) / len(summary)
    avg_macro = sum(m for _, _, m in summary) / len(summary)
    print(f"{'AVERAGE':12s} {avg_acc:8.3f} {avg_macro:8.3f}")


if __name__ == "__main__":
    main()
