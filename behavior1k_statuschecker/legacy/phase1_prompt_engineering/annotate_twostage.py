"""Two-stage classification, on Qwen3.5-2B.

Every single-shot 4-way prompt we tried (v1-v23) shared one failure pattern: "move to"
(no arm visible) kept winning frames it shouldn't, because it's the only class scored on
absence rather than positive evidence, and it competes directly against classes 2-4 in
the same decision. This decomposes the problem:

  Stage 1 (cheap, reliable): is an arm/gripper visible at all? This alone was 9/9 in
  nearly every prior test - it's not the hard part.
  Stage 2 (only runs if stage 1 says yes): given an arm IS visible, which of the 3
  remaining classes (pick up from / press / place on) fits? Class 1 never competes here.

Uses v14's exact per-class descriptive language for stage 2, just narrowed to 3 options.
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
PROMPT_VERSION = "2b_twostage"

STAGE1_PROMPT = (
    "Is a robot arm or gripper visible anywhere in this frame - even a small part of "
    "one at the edge of the picture? Answer with ONLY yes or no."
)

STAGE2_PROMPT = (
    f'Task: "{TASK_INSTRUCTION}"\n'
    "A robot arm/gripper is visible in this frame, interacting with a red radio "
    "receiver. Which of these best describes what's happening?\n"
    "2. pick up from - the gripper is reaching toward or holding the red radio, but "
    "the radio's round dial is NOT facing the camera (you only see the plain "
    "red-and-white striped body, not the dial face).\n"
    "3. press - the radio's flat face with its round dial IS facing the camera, and "
    "the gripper is directly on that dial. The radio does not need to be lifted for "
    "this - it can happen while the radio still sits on the table.\n"
    "4. place on - the radio is resting back down on the table (or being lowered onto "
    "it) and the gripper is pulling away rather than touching the dial.\n"
    "Respond with ONLY the digit (2, 3, or 4)."
)

NUMBERED_LABELS = {"2": "pick up from", "3": "press", "4": "place on"}


def parse_yesno(raw_text):
    return "yes" in raw_text.strip().lower()[:5]


def parse_234(raw_text):
    m = re.search(r"[234]", raw_text)
    return NUMBERED_LABELS.get(m.group(0)) if m else None


def ask(model, processor, img, prompt_text, max_new_tokens=8):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt_text},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


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

        raw1 = ask(model, processor, img, STAGE1_PROMPT)
        arm_visible = parse_yesno(raw1)

        if not arm_visible:
            pred = "move to"
            raw2 = None
        else:
            raw2 = ask(model, processor, img, STAGE2_PROMPT)
            pred = parse_234(raw2)

        gt_label = item["skill_description"]
        is_correct = pred == gt_label
        correct += int(is_correct)
        total += 1
        predictions.append({
            "sec": item["sec"], "gt": gt_label, "pred": pred,
            "stage1_raw": raw1, "stage2_raw": raw2, "correct": is_correct,
        })
        mark = "OK " if is_correct else "XX "
        print(f"{mark} sec={item['sec']:3d} gt={gt_label:15s} pred={str(pred):15s} "
              f"stage1={raw1!r} stage2={raw2!r}")

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
