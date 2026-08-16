"""Run Qwen3.5-2B across all 10 tasks using the best-so-far rules per task, but asking
for a brief reasoning trace before the digit answer - this improved the radio task
(53.6% -> 55.8% macro) and is cheap to test broadly before further rule iteration.
"""
import json
import re

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

ROOT = "/shared_work/markhsp/behavior1k-statuschecker"
MODEL_ID = "Qwen/Qwen3.5-2B"
RESULTS_DIR = f"{ROOT}/results"

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
        "Look only at what is visible in this frame. Respond in EXACTLY this format:\n"
        "Reasoning: <one short sentence on what you actually see that supports your answer>\n"
        "Answer: <digit>"
    )


def parse_pred(raw_text, numbered):
    a = re.search(r"Answer:\s*(\d+)", raw_text, re.IGNORECASE)
    if a:
        return numbered.get(a.group(1))
    m = re.search(r"\d+", raw_text)
    return numbered.get(m.group(0)) if m else None


def run_task(model, processor, task_chunk, task_instruction):
    with open(f"{ROOT}/multitask_data/{task_chunk}/ground_truth.json") as f:
        gt = json.load(f)
    with open(f"{RESULTS_DIR}/multitask_rules_best_{task_chunk}.md") as f:
        rules_md = f.read()
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
            out = model.generate(**inputs, max_new_tokens=70, do_sample=False)
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

    with open(f"{RESULTS_DIR}/multitask_reasoning_{task_chunk}.json", "w") as f:
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

    print("\n\n===== SUMMARY: reasoning-trace vs best-so-far =====")
    for tc, acc, macro in summary:
        best_r1 = json.load(open(f"{RESULTS_DIR}/multitask_v2_{tc}.json"))["macro_recall"]
        best_r2 = json.load(open(f"{RESULTS_DIR}/multitask_v3_{tc}.json"))["macro_recall"]
        best_so_far = max(best_r1, best_r2)
        print(f"{tc:12s} reasoning_macro={macro:.3f}  best_so_far={best_so_far:.3f}  {'IMPROVED' if macro > best_so_far else ''}")


if __name__ == "__main__":
    main()
