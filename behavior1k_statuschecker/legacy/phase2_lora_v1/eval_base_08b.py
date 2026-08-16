"""Evaluate the UNTRAINED base Qwen3.5-0.8B (no LoRA adapter) on the exact same
prompts/data as eval_lora.py and eval_lora_heldout.py, for a fair three-way comparison:
base 0.8B vs LoRA-finetuned 0.8B vs prompt-engineered 2B.
"""
import json
import re

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

ROOT = "/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
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

TRAINING_EPISODE = {
    "task-0000": "episode_00000010", "task-0001": "episode_00010010",
    "task-0005": "episode_00050020", "task-0020": "episode_00200010",
    "task-0022": "episode_00220020", "task-0031": "episode_00310010",
    "task-0034": "episode_00340020", "task-0040": "episode_00400010",
    "task-0042": "episode_00420010", "task-0049": "episode_00490010",
}

SYSTEM_PROMPT = (
    "You are annotating individual frames from a fixed head-mounted camera on a "
    "mobile manipulator robot completing a household task. Classify the robot's "
    "current subskill using ONLY the visual evidence in this single frame."
)


def parse_rules(md_text):
    lines = []
    for line in md_text.strip().split("\n"):
        m = re.match(r"^\s*(\d+)\.\s*([^-]+?)\s*-\s*(.+)$", line)
        if m:
            n, label, rule = m.groups()
            lines.append(f"{n}. {label.strip()} - {rule.strip()}")
    return "\n".join(lines)


def build_user_prompt(task_instruction, action_list_text):
    return (
        f'Task: "{task_instruction}"\n'
        f"The robot's subskill right now is always exactly one of:\n{action_list_text}\n"
        "Look only at what is visible in this frame. Respond in EXACTLY this format:\n"
        "Reasoning: <one short sentence on what you actually see that supports your answer>\n"
        "Answer: <label>"
    )


def parse_answer(raw_text, labels):
    m = re.search(r"Answer:\s*(.+)", raw_text, re.IGNORECASE)
    text = (m.group(1) if m else raw_text).strip().lower().strip('."\'')
    for lab in labels:
        if text == lab.lower():
            return lab
    for lab in sorted(labels, key=len, reverse=True):
        if lab.lower() in text:
            return lab
    return None


def run_items(model, processor, items, task_instruction, action_list_text, labels, tag, task_chunk, gt_task_name):
    prompt = build_user_prompt(task_instruction, action_list_text)
    predictions = []
    correct = 0
    for item in items:
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
        pred = parse_answer(raw, labels)

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

    print(f"\n=== {task_chunk} ({gt_task_name}) [{tag}]: acc={acc:.3f} ({correct}/{total})  macro={macro:.3f} ===")
    for k, (c, t) in sorted(by_class.items()):
        print(f"    {k:20s} {c}/{t}")

    with open(f"{RESULTS_DIR}/multitask_base08b_{tag}_{task_chunk}.json", "w") as f:
        json.dump({
            "task_chunk": task_chunk, "tag": tag,
            "accuracy": acc, "macro_recall": macro, "correct": correct, "total": total,
            "predictions": predictions,
        }, f, indent=2)
    return acc, macro


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()

    summary_train = []
    summary_heldout = []
    for task_chunk, instruction in TASKS.items():
        with open(f"{RESULTS_DIR}/multitask_rules_best_{task_chunk}.md") as f:
            action_list_text = parse_rules(f.read())

        # training-set-equivalent (original single-episode benchmark)
        with open(f"{ROOT}/multitask_data/{task_chunk}/ground_truth.json") as f:
            gt_small = json.load(f)
        labels_small = gt_small["distinct_labels"]
        acc, macro = run_items(model, processor, gt_small["per_second"], instruction,
                                action_list_text, labels_small, "trainset", task_chunk, gt_small["task_name"])
        summary_train.append((task_chunk, acc, macro))

        # held-out (5 unseen episodes) from the v2 dataset
        with open(f"{ROOT}/multitask_data_v2/{task_chunk}/ground_truth.json") as f:
            gt_v2 = json.load(f)
        held_out_items = [it for it in gt_v2["per_second"] if it["episode"] != TRAINING_EPISODE[task_chunk]]
        labels_v2 = sorted(set(it["skill_description"] for it in held_out_items))
        acc_h, macro_h = run_items(model, processor, held_out_items, instruction,
                                    action_list_text, labels_v2, "heldout", task_chunk, gt_v2["task_name"])
        summary_heldout.append((task_chunk, acc_h, macro_h))

    print("\n\n===== SUMMARY: BASE (untrained) Qwen3.5-0.8B =====")
    print(f"{'task':12s} {'train_macro':>12s} {'heldout_macro':>14s}")
    for (tc, _, m1), (_, _, m2) in zip(summary_train, summary_heldout):
        print(f"{tc:12s} {m1:12.3f} {m2:14.3f}")
    avg1 = sum(m for _, _, m in summary_train) / len(summary_train)
    avg2 = sum(m for _, _, m in summary_heldout) / len(summary_heldout)
    print(f"{'AVERAGE':12s} {avg1:12.3f} {avg2:14.3f}")


if __name__ == "__main__":
    main()
