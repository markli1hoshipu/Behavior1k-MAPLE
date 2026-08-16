"""Run Qwen3.5-2B across all 10 tasks using the Opus-derived, task-specific ACTION_LIST
rules (results/multitask_rules_task-*.md) instead of the generic label-list baseline.
"""
import glob
import json
import os
import re
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

ROOT = "/shared_work/markhsp/behavior1k-statuschecker"
MODEL_ID = "Qwen/Qwen3.5-2B"
RESULTS_DIR = f"{ROOT}/results"

# Which rules round to test: "" for round 1 (multitask_rules_task-*.md), "v2" for round 2
# (multitask_rules_v2_task-*.md), "v3" for round 3, etc, "best" for multitask_rules_best_*.md.
# Pass as first CLI arg, e.g. `python3 annotate_multitask_v2.py v3`.
RULES_ROUND = sys.argv[1] if len(sys.argv) > 1 else ""
_OUT_TAG_MAP = {"": "v2", "v2": "v3", "v3": "v4", "v4": "v5", "best": "bestcheck"}
OUT_TAG = _OUT_TAG_MAP.get(RULES_ROUND, f"{RULES_ROUND}_out")

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


def parse_rules(md_text):
    """Parse 'N. label - rule' lines into an ordered {digit: label} map and clean ACTION_LIST text."""
    numbered = {}
    lines = []
    for line in md_text.strip().split("\n"):
        m = re.match(r"^\s*(\d+)\.\s*([^-]+?)\s*-\s*(.+)$", line)
        if m:
            n, label, rule = m.groups()
            numbered[n] = label.strip()
            lines.append(f"{n}. {label.strip()} - {rule.strip()}")
    return numbered, "\n".join(lines)


def build_prompt(task_instruction, action_list_text):
    return (
        f'Task: "{task_instruction}"\n'
        f"The robot's subskill right now is always exactly one of:\n{action_list_text}\n"
        "Look only at what is visible in this frame. Which number best matches? "
        "Respond with ONLY the digit."
    )


def parse_pred(raw_text, numbered):
    m = re.search(r"\d+", raw_text)
    return numbered.get(m.group(0)) if m else None


def run_task(model, processor, task_chunk, task_instruction):
    with open(f"{ROOT}/multitask_data/{task_chunk}/ground_truth.json") as f:
        gt = json.load(f)
    rules_suffix = f"{RULES_ROUND}_{task_chunk}" if RULES_ROUND else task_chunk
    try:
        with open(f"{RESULTS_DIR}/multitask_rules_{rules_suffix}.md") as f:
            rules_md = f.read()
    except FileNotFoundError:
        print(f"SKIP {task_chunk}: no rules file for round {RULES_ROUND!r} yet")
        return None
    numbered, action_list_text = parse_rules(rules_md)
    if not numbered:
        print(f"SKIP {task_chunk}: could not parse rules")
        return None
    prompt = build_prompt(task_instruction, action_list_text)

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
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        raw = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        pred = parse_pred(raw, numbered)

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

    with open(f"{RESULTS_DIR}/multitask_{OUT_TAG}_{task_chunk}.json", "w") as f:
        json.dump({
            "task_chunk": task_chunk, "task_name": gt["task_name"],
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
        result = run_task(model, processor, task_chunk, instruction)
        if result:
            summary.append((task_chunk, *result))

    print(f"\n\n===== SUMMARY: {OUT_TAG} (this round) vs earlier rounds =====")
    print(f"{'task':12s} {OUT_TAG+'_acc':>8s} {OUT_TAG+'_macro':>9s} {'base_acc':>9s} {'base_macro':>11s} {'r1_acc':>8s} {'r1_macro':>9s}")
    for tc, acc, macro in summary:
        try:
            base = json.load(open(f"{RESULTS_DIR}/multitask_{tc}.json"))
            base_acc, base_macro = base["accuracy"], base["macro_recall"]
        except FileNotFoundError:
            base_acc, base_macro = float("nan"), float("nan")
        try:
            r1 = json.load(open(f"{RESULTS_DIR}/multitask_v2_{tc}.json"))
            r1_acc, r1_macro = r1["accuracy"], r1["macro_recall"]
        except FileNotFoundError:
            r1_acc, r1_macro = float("nan"), float("nan")
        print(f"{tc:12s} {acc:8.3f} {macro:9.3f} {base_acc:9.3f} {base_macro:11.3f} {r1_acc:8.3f} {r1_macro:9.3f}")


if __name__ == "__main__":
    main()
