"""Structured chain-of-thought, single call, on Qwen3.5-2B.

Neither free-form reasoning (v14_reasoning) nor splitting into two separate calls
(twostage) beat v14 outright. This tries a third structure: one call, but the model must
answer explicit sub-questions in sequence before its final digit, so the intermediate
judgments are forced and structured (not freely worded) while staying in one generation
(so later answers can be conditioned on earlier ones within the same forward pass,
unlike the two-call split which can't see its own stage-1 answer as context).
"""
import json
import os
import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

EPISODE = "00000010"
DATA_DIR = f"/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker/data/episode_{EPISODE}"
RESULTS_DIR = "/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TASK_INSTRUCTION = "Turn on the radio receiver that's on the table in the living room."
MODEL_ID = "Qwen/Qwen3.5-2B"
PROMPT_VERSION = "2b_structured_cot_v2"
NUMBERED_LABELS = {"1": "move to", "2": "pick up from", "3": "press", "4": "place on"}

SYSTEM_PROMPT = (
    "You are annotating individual frames from a fixed head-mounted camera on a "
    "mobile manipulator robot. Classify the robot's current subskill using ONLY "
    "the visual evidence in this single frame."
)

# v2: Q2's 3-way choice (body/dial/none) never once produced "body" across 59 frames -
# "dial" and "none" won every time even on true pick-up-from frames, the same attractor
# pattern seen throughout this whole session. Splitting it into two INDEPENDENT binary
# questions instead, since every yes/no question we've tried all session (arm-visible)
# has been reliable, while every 3+-way choice has shown some kind of default-option bias.
USER_PROMPT = f"""Task: "{TASK_INSTRUCTION}"
The robot's subskill right now is always exactly one of:
1. move to - no robot arm is visible.
2. pick up from - an arm/gripper is gripping/squeezing the radio's body (not its round dial).
3. press - the gripper is directly touching the radio's round dial face.
4. place on - an arm is visible near the radio but the gripper is not gripping the body and not touching the dial (pulling away, releasing, or hovering with a gap).

Answer these questions IN ORDER, then give your final answer. Respond in EXACTLY this
format, one line each:
Q1 (is any part of a robot arm/gripper visible anywhere in the frame?): <yes/no>
Q2 (is the gripper touching/pressing the round dial face specifically?): <yes/no>
Q3 (is the gripper gripping/squeezing the radio's plain body, elsewhere from the dial?): <yes/no>
Q4 (final action number, using your Q1/Q2/Q3 answers above - if Q1 is no, answer 1; else if Q2 is yes, answer 3; else if Q3 is yes, answer 2; else answer 4): <1/2/3/4>"""


def parse(raw_text):
    # Q4's answer comes after the LAST "):" in the Q4 line - search only past that point
    # so we never accidentally match the literal digit inside the "Q4" label itself.
    q_label = raw_text.rfind("Q4")
    if q_label == -1:
        return None
    after_label = raw_text[q_label:]
    colon = after_label.find("):")
    search_region = after_label[colon + 2:] if colon != -1 else after_label[len("Q4"):]
    m = re.search(r"[1-4]", search_region)
    return NUMBERED_LABELS.get(m.group(0)) if m else None


def main():
    with open(f"{DATA_DIR}/ground_truth.json") as f:
        gt = json.load(f)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0"
    )

    items = [it for it in gt["per_second"] if it["skill_description"] is not None]

    predictions = []
    correct = 0
    total = 0
    for item in items:
        img = Image.open(item["frame_path"]).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=140, do_sample=False)
        raw = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        pred = parse(raw)

        gt_label = item["skill_description"]
        is_correct = pred == gt_label
        correct += int(is_correct)
        total += 1
        predictions.append({
            "sec": item["sec"], "gt": gt_label, "pred": pred,
            "raw": raw, "correct": is_correct,
        })
        mark = "OK " if is_correct else "XX "
        print(f"{mark} sec={item['sec']:3d} gt={gt_label:15s} pred={str(pred):15s} raw={raw!r}")

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
            "version": PROMPT_VERSION, "accuracy": acc, "macro_recall": macro,
            "correct": correct, "total": total, "predictions": predictions,
        }, f, indent=2)


if __name__ == "__main__":
    main()
