"""Qwen3.5-0.8B subskill annotator with TEXT-based long-term memory.

Unlike annotate.py (which tried feeding multiple raw *images* as temporal context
and found the model collapses/degenerates with 2+ images), this keeps every model
call single-image, and instead carries the model's own short natural-language
summary of the previous second forward as part of the *text* prompt. That gives
it episode-level continuity without ever showing it more than one image at a time.

Each step the model is asked to (1) briefly describe what it sees and (2) guess
the current action number, in a fixed two-line format. The description is fed
back in as "memory" for the next second's prompt.

Edit PROMPT_VERSION / SYSTEM_PROMPT / build_user_prompt() between runs.
"""
import glob
import json
import os
import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

EPISODE = "00000010"
DATA_DIR = f"/shared_work/markhsp/behavior1k-statuschecker/data/episode_{EPISODE}"
RESULTS_DIR = "/shared_work/markhsp/behavior1k-statuschecker/results"
ANNOT_GLOB = "/shared_work/DATASETS/behavior-1k-embodiedAI-rollouts/annotations/task-0000/*.json"
os.makedirs(RESULTS_DIR, exist_ok=True)

TASK_INSTRUCTION = "Turn on the radio receiver that's on the table in the living room."
LABELS = ["move to", "pick up from", "press", "place on"]
NUMBERED_LABELS = {"1": "move to", "2": "pick up from", "3": "press", "4": "place on"}


def derive_action_order(pattern=ANNOT_GLOB):
    """Scan annotated demos and return the (single, if consistent) action order."""
    orders = {}
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        seq = tuple(seg["skill_description"][0] for seg in d["skill_annotation"])
        orders[seq] = orders.get(seq, 0) + 1
    # majority order across all annotated demos
    return max(orders.items(), key=lambda kv: kv[1])[0]


ACTION_ORDER = derive_action_order()
ORDER_STR = " -> ".join(
    f"{i+1} {name}" for i, name in enumerate(ACTION_ORDER)
)

# ==================== EDIT THIS BLOCK BETWEEN ITERATIONS ====================
PROMPT_VERSION = "v11"
# v10 (free-text description carried as memory) got macro_recall=0.361: worse than the
# 0-shot v6 baseline (0.444), because the model's "Description" field turned out to be
# constant boilerplate on every frame regardless of image content - it can't reliably
# caption an image at this size, so the memory it produced was uninformative. Here we
# still ask it to describe (for logging), but carry forward only the PREVIOUS PREDICTED
# ACTION as memory instead - something we know is reliably extracted, unlike free text.
MEMORY_MODE = "action"  # "action" | "description"

SYSTEM_PROMPT = (
    "You are annotating a robot's head-camera video, one frame per second, for an "
    "entire episode. You only ever see ONE frame at a time, but you keep short "
    "written notes across seconds so you remember what already happened."
)

ACTION_LIST = (
    "The robot's subskill right now is always exactly one of:\n"
    "1. move to - the robot's arms/grippers are NOT visible in frame, or are "
    "resting/idle; the robot is still navigating across the room toward the object.\n"
    "2. pick up from - an arm/gripper IS visible, reaching toward, closing around, "
    "or carrying the red object while it is held UP off the table surface, away from any button.\n"
    "3. press - a gripper's fingertip is directly ON TOP OF the round button in the "
    "CENTER of the red object, pressing down on it.\n"
    "4. place on - an arm/gripper IS visible and the red object is flat/resting DOWN "
    "on the table or mat surface (not raised in the air), with the gripper releasing "
    "or already let go of it."
)


def build_user_prompt(memory):
    if memory:
        memory_line = f"1 second ago, the action was: {memory}\n"
    else:
        memory_line = "This is the first frame of the episode - there is no prior action.\n"
    return (
        f'Task: "{TASK_INSTRUCTION}"\n'
        f"{ACTION_LIST}\n"
        f"Based on annotated demonstrations of this task, actions always happen in this "
        f"order and never repeat or go backward: {ORDER_STR}.\n"
        f"{memory_line}"
        "Look at the current frame. Reply in EXACTLY this two-line format:\n"
        "Description: <one short sentence on what you see now>\n"
        "Action: <digit 1-4>"
    )
# ==============================================================================


def parse_response(raw_text):
    desc_m = re.search(r"Description:\s*(.*)", raw_text, re.IGNORECASE)
    act_m = re.search(r"Action:\s*([1-4])", raw_text, re.IGNORECASE)
    if not act_m:
        act_m = re.search(r"[1-4]", raw_text)
    description = desc_m.group(1).strip() if desc_m else raw_text.strip()
    description = description.split("\n")[0][:150]
    action_num = act_m.group(0)[-1] if act_m else None
    pred = NUMBERED_LABELS.get(action_num)
    return description, pred


def main():
    with open(f"{DATA_DIR}/ground_truth.json") as f:
        gt = json.load(f)

    model_id = "Qwen/Qwen3.5-0.8B"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0"
    )

    items = [it for it in gt["per_second"] if it["skill_description"] is not None]

    predictions = []
    correct = 0
    total = 0
    memory = None
    for item in items:
        img = Image.open(item["frame_path"]).convert("RGB")
        user_prompt = build_user_prompt(memory)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_prompt},
            ]},
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
        raw = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        description, pred = parse_response(raw)
        if MEMORY_MODE == "action" and pred is not None:
            action_num = [k for k, v in NUMBERED_LABELS.items() if v == pred][0]
            memory = f"{action_num} ({pred})"
        else:
            memory = description  # carry forward into next step's prompt

        gt_label = item["skill_description"]
        is_correct = pred == gt_label
        correct += int(is_correct)
        total += 1
        predictions.append({
            "sec": item["sec"], "gt": gt_label, "pred": pred,
            "description": description, "raw": raw.strip(), "correct": is_correct,
        })
        mark = "OK " if is_correct else "XX "
        print(f"{mark} sec={item['sec']:3d} gt={gt_label:15s} pred={str(pred):15s} desc={description!r}")

    acc = correct / total if total else 0.0
    by_class = {}
    for p in predictions:
        by_class.setdefault(p["gt"], [0, 0])
        by_class[p["gt"]][1] += 1
        if p["correct"]:
            by_class[p["gt"]][0] += 1
    macro = sum(c / t for c, t in by_class.values()) / len(by_class)
    print(f"\n=== {PROMPT_VERSION}: accuracy {correct}/{total} = {acc:.3f}  macro_recall={macro:.3f} ===")
    for k, (c, t) in by_class.items():
        print(f"    {k:15s} {c}/{t}")

    with open(f"{RESULTS_DIR}/{PROMPT_VERSION}.json", "w") as f:
        json.dump({
            "version": PROMPT_VERSION, "action_order": list(ACTION_ORDER),
            "system_prompt": SYSTEM_PROMPT, "accuracy": acc, "macro_recall": macro,
            "correct": correct, "total": total, "predictions": predictions,
        }, f, indent=2)


if __name__ == "__main__":
    main()
