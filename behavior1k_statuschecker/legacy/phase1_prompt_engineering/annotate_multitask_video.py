"""Native video input (2-frame window) for the two tasks whose oracle ceiling is below
70% with single-frame methods - both have Opus-confirmed motion-direction-dependent
confusions (chop vs push_to, open vs close door) that a single static frame structurally
cannot resolve. Native video worked passably for the radio task at 2B scale (37.3% raw
on 2b_v13b) so it's worth testing here specifically where the failure mode matches.
"""
import json
import re

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.video_utils import VideoMetadata

ROOT = "/shared_work/markhsp/behavior1k-statuschecker"
MODEL_ID = "Qwen/Qwen3.5-2B"
RESULTS_DIR = f"{ROOT}/results"

TASKS = {
    "task-0042": "Take the onion out of the sink, dice it on the chopping board, and put it in the bowl.",
    "task-0049": "Make a pizza: chop vegetables, add toppings to the dough, and bake it in the oven.",
}

SYSTEM_PROMPT = (
    "You are annotating a robot's head-mounted camera video. You will see the last "
    "two frames, one second apart, ending at the current moment - use the earlier "
    "frame only to judge motion direction, and classify the CURRENT (last) frame."
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
        "Which number best matches what's happening in the CURRENT (most recent) frame? "
        "Respond with ONLY the digit."
    )


def parse_pred(raw_text, numbered):
    m = re.search(r"\d+", raw_text)
    return numbered.get(m.group(0)) if m else None


def run_task(model, processor, task_chunk, task_instruction):
    with open(f"{ROOT}/multitask_data/{task_chunk}/ground_truth.json") as f:
        gt = json.load(f)
    with open(f"{RESULTS_DIR}/multitask_rules_best_{task_chunk}.md") as f:
        rules_md = f.read()
    numbered, action_list_text = parse_rules(rules_md)
    prompt = build_prompt(task_instruction, action_list_text)

    items = gt["per_second"]
    predictions = []
    correct = 0
    for idx, item in enumerate(items):
        window = items[max(0, idx - 1):idx + 1]
        if len(window) < 2:
            window = [window[0], window[0]]
        frames = [Image.open(w["frame_path"]).convert("RGB") for w in window]
        video_metadata = [VideoMetadata(total_num_frames=2, fps=1.0, duration=2.0)]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": prompt},
            ]},
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", video_metadata=video_metadata
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

    with open(f"{RESULTS_DIR}/multitask_video_{task_chunk}.json", "w") as f:
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
    for task_chunk, instruction in TASKS.items():
        run_task(model, processor, task_chunk, instruction)


if __name__ == "__main__":
    main()
