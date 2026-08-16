"""Run the same v10 (single-frame + text memory) annotation setup used for
Qwen3.5-0.8B, but through Claude Haiku 4.5 via the `claude -p` CLI, for a
direct comparison on the same episode/ground-truth.
"""
import glob
import json
import os
import re
import subprocess
import time

EPISODE = "00000010"
ROOT = "/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker"
DATA_DIR = f"{ROOT}/data/episode_{EPISODE}"
RESULTS_DIR = f"{ROOT}/results"
ANNOT_GLOB = "/shared_work/DATASETS/behavior-1k-embodiedAI-rollouts/annotations/task-0000/*.json"

TASK_INSTRUCTION = "Turn on the radio receiver that's on the table in the living room."
NUMBERED_LABELS = {"1": "move to", "2": "pick up from", "3": "press", "4": "place on"}

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


def derive_action_order(pattern=ANNOT_GLOB):
    orders = {}
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        seq = tuple(seg["skill_description"][0] for seg in d["skill_annotation"])
        orders[seq] = orders.get(seq, 0) + 1
    return max(orders.items(), key=lambda kv: kv[1])[0]


ACTION_ORDER = derive_action_order()
ORDER_STR = " -> ".join(f"{i+1} {name}" for i, name in enumerate(ACTION_ORDER))


def build_user_prompt(frame_path, memory):
    memory_line = (
        f'Your notes from 1 second ago: "{memory}"\n'
        if memory else "This is the first frame of the episode - you have no prior notes.\n"
    )
    return (
        f"Read/view the image at {frame_path}\n\n"
        f'Task: "{TASK_INSTRUCTION}"\n'
        f"{ACTION_LIST}\n"
        f"Based on annotated demonstrations of this task, actions always happen in this "
        f"order and never repeat or go backward: {ORDER_STR}.\n"
        f"{memory_line}"
        "Look at the current frame. Reply in EXACTLY this two-line format, nothing else:\n"
        "Description: <one short sentence on what you see now>\n"
        "Action: <digit 1-4>"
    )


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


def call_haiku(prompt):
    result = subprocess.run(
        ["claude", "-p", prompt, "--model", "claude-haiku-4-5-20251001",
         "--allowedTools", "Read", "--permission-mode", "bypassPermissions",
         "--output-format", "text"],
        capture_output=True, text=True, timeout=60,
    )
    return result.stdout.strip()


def main():
    with open(f"{DATA_DIR}/ground_truth.json") as f:
        gt = json.load(f)
    items = [it for it in gt["per_second"] if it["skill_description"] is not None]

    predictions = []
    correct = 0
    total = 0
    memory = None
    t0 = time.time()
    for item in items:
        prompt = build_user_prompt(item["frame_path"], memory)
        raw = call_haiku(prompt)
        description, pred = parse_response(raw)
        memory = description

        gt_label = item["skill_description"]
        is_correct = pred == gt_label
        correct += int(is_correct)
        total += 1
        predictions.append({
            "sec": item["sec"], "gt": gt_label, "pred": pred,
            "description": description, "raw": raw, "correct": is_correct,
        })
        mark = "OK " if is_correct else "XX "
        print(f"{mark} sec={item['sec']:3d} gt={gt_label:15s} pred={str(pred):15s} desc={description!r}", flush=True)

    acc = correct / total if total else 0.0
    by_class = {}
    for p in predictions:
        by_class.setdefault(p["gt"], [0, 0])
        by_class[p["gt"]][1] += 1
        if p["correct"]:
            by_class[p["gt"]][0] += 1
    macro = sum(c / t for c, t in by_class.values()) / len(by_class)
    elapsed = time.time() - t0
    print(f"\n=== haiku4.5 (v10-style): accuracy {correct}/{total} = {acc:.3f}  macro_recall={macro:.3f}  ({elapsed:.0f}s) ===")
    for k, (c, t) in by_class.items():
        print(f"    {k:15s} {c}/{t}")

    with open(f"{RESULTS_DIR}/haiku45_v10.json", "w") as f:
        json.dump({
            "version": "haiku45_v10", "model": "claude-haiku-4-5-20251001",
            "action_order": list(ACTION_ORDER), "accuracy": acc, "macro_recall": macro,
            "correct": correct, "total": total, "elapsed_s": elapsed, "predictions": predictions,
        }, f, indent=2)


if __name__ == "__main__":
    main()
