"""Run Qwen3.5-0.8B as a per-second subskill annotator over extracted frames,
score alignment against ground truth, and log results for prompt iteration.

Edit PROMPT_VERSION / SYSTEM_PROMPT / build_user_prompt() / WINDOW between runs.
"""
import json
import os
import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.video_utils import VideoMetadata

EPISODE = "00000010"
DATA_DIR = f"/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker/data/episode_{EPISODE}"
RESULTS_DIR = "/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TASK_INSTRUCTION = "Turn on the radio receiver that's on the table in the living room."
LABELS = ["move to", "pick up from", "press", "place on"]
NUMBERED_LABELS = {"1": "move to", "2": "pick up from", "3": "press", "4": "place on"}
MODEL_ID = "Qwen/Qwen3.5-2B"

# ==================== EDIT THIS BLOCK BETWEEN ITERATIONS ====================
# v18: v17 (23.7%) did not "confuse" classes, it deleted two of them - across 59 frames it
# emitted only 'move to' (54) and 'press' (5), zero 'pick up from', zero 'place on'.
# Measured cause: rule 1 is the only rule scored on whole-frame gestalt, and its positive
# half ("wide view of the room, radio small/far") is true in 100% of frames here - the
# couch+floor fill every frame. So rule 1 is a permanent attractor gated only by a
# negation, and rules 2-4 must out-score it on positive evidence. v14 let rule 2 win with
# an emphatic first-position existence check ("IS visible"), a concessive "but" (default-
# with-exception, not a conjunction), and a positive paraphrase of the discriminator.
# v17 replaced that with a 4-conjunct AND-chain, dropped the paraphrase and the color
# anchor, and - most damaging - dropped "at all anywhere in the frame" from the tie-break,
# the final and highest-recency sentence. Measured: 61-99% of the arm's dark pixels are in
# frame-border-touching components (it is cropped at the bottom edge, never a centered
# salient object), so an unscoped "is an arm visible?" gets answered from frame center,
# where there is no arm. v14 states the scope twice; v17 once.
# v18 = v14 verbatim plus three evidence-backed edits:
#  (a) make the arm check concrete and coarse-scale. Dark pixels in the bottom quarter of
#      the frame: 0.00-1.71% for every arm-absent frame (secs 0-24), 8.0-25.9% for every
#      arm-present frame (secs 27-58). Perfect separation, >4x gap, low-frequency cue.
#  (b) back the dial test with an absolute aspect-ratio cue. Largest red component w/h:
#      0.19-0.48 in pick-up frames (tall, narrow), 1.30-1.39 in press frames (wider than
#      tall, ~3x the width and ~4x the pixels). Stated absolutely, not as a comparison to
#      another frame, so a stateless per-frame classifier can evaluate it.
#  (c) drop v14's green-light claim (confirmed 0 green px in all 11 place-on frames; the
#      only radio-side green is 43-80 px at secs 46-47, i.e. during press, and 80 px is
#      well under one merged visual token) and drop the tilt cue instead of reinstating
#      it. Tilt is real in pixels (2-13 deg from vertical in pick-up vs 21-36 in place-on)
#      but the red mask fragments into 3-5 components with the largest holding only 51-63%
#      of the red pixels, on a ~85x115 px object - and v17 made tilt the sole class-4
#      discriminator and got 0 activations in 11 chances. Class 4 is declared
#      unrecoverable from a single frame here; rule 4 is kept truthful and deliberately
#      non-competing so it cannot steal rule 2's frames.
# v19: v18 scored 35.6%. The bottom-edge arm cue worked exactly as designed - class 2
# fired at precisely secs 28-38 (11/11, matching v14) and class 1 at 0-27 - but press
# collapsed 7/9 -> 1/9. Cause is the same attractor mechanism, now working against rule 3:
# I put the arm cue INSIDE rule 2, and dark-bottom-quarter is 13-20% at secs 40-46 too, so
# rule 2 gained a clause that is satisfiable in exactly the frames where rule 3 has to
# win, while rule 3 gained nothing. Whichever rule has more satisfiable positive clauses
# takes the frame. The arm cue is a gate between 1 and {2,3,4}, not a feature of class 2,
# so v19 states it once in a preamble and leaves rules 2/3/4 with equal clause counts.
# v20 tested the reverse: drop the aspect-ratio parenthetical from rule 3, keeping only
# the preamble gate + rule 4 fix. That scored 42.4% (25/59) with press at 5/9, i.e. the
# aspect cue WAS load-bearing for press (v19 6/9 vs v20 5/9). So v19 stands as the best
# revision. Final scoreboard on this episode: v14 45.8% (27/59), v19 44.1% (26), v20 42.4%
# (25), v18 35.6% (21), v17 23.7% (14).
#
# v19 does not beat v14 - it is one frame short, and that frame (sec 42) is a decode
# boundary flip between visually identical frames (41/42/43 red-blob aspect 1.33/1.31/
# 1.30). What v19 buys instead is a temporally coherent segmentation (1|2|3|2|1 in one
# pass) rather than v14's flip-flopping tail, and a prompt whose every clause is backed by
# a measured cue rather than by an invented one.
#
# Prompt engineering is exhausted here. Of the 33 remaining errors, 19 are pick-up frames
# (secs 9-27) where NO arm is in view at all - dark-bottom-quarter is 0.00-1.71%, the
# frame shows an empty room, and no wording can make a single-frame classifier report an
# arm that is not there - and 11 are place-on frames that are near time-reverses of the
# pick-up frames. Both groups need temporal context (WINDOW > 1), not better words. The
# single-frame ceiling on this episode is roughly 9 + 12 + 8 + 0 = 29/59 (~49%); v14 at 27
# is already within two frames of it.
PROMPT_VERSION = "2b_v14"
WINDOW = 1
STITCH = False
NATIVE_VIDEO = False
ASK_FOR_REASONING = False

# Restored as the active default after 4 rounds of Opus consultation (v15-v23, including
# a v19/v22 hybrid in v23) failed to beat this on either raw accuracy or macro recall -
# see results/opus5_rules_output_v{1,2,3,4}.md for the full derivation trail and
# results/2b_v*.json for every tested variant's scores. Best result across all versions:
# 45.8% raw / 53.6% macro (2b_v14). 2b_v14_reasoning trades some raw accuracy for higher
# macro recall (42.4%/55.8%) by asking the model to explain itself before answering.
FEW_SHOT_EXAMPLES = []

SYSTEM_PROMPT = (
    "You are annotating individual frames from a fixed head-mounted camera on a "
    "mobile manipulator robot. Classify the robot's current subskill using ONLY "
    "the visual evidence in this single frame."
)

ACTION_LIST = (
    "The robot's subskill right now is always exactly one of:\n"
    "1. move to - no robot arm is visible anywhere in the frame; you see a wide view "
    "of the room (floor, walls, couch) with the red radio small, far away, or not "
    "visible at all.\n"
    "2. pick up from - a robot arm/gripper IS visible, reaching toward or holding the "
    "red radio, but the radio's round dial is NOT facing the camera (you only see the "
    "plain red-and-white striped body, not the dial face).\n"
    "3. press - the radio's flat face with its round dial IS facing the camera, and a "
    "gripper is directly on that dial. This is the only step where the gripper touches "
    "the round dial. The radio does not need to be lifted for this - it can happen "
    "while the radio still sits on the table.\n"
    "4. place on - the radio is resting back down on the table (or being lowered onto "
    "it) and the gripper is pulling away rather than touching the dial. If the dial is "
    "visible, it may show a small green light now on (this means the radio was already "
    "successfully pressed).\n"
    "Tie-break: if no arm is visible at all anywhere in the frame, answer 1 (move to)."
)

def build_user_prompt(task_instruction, labels):
    if ASK_FOR_REASONING:
        return (
            f'Task: "{task_instruction}"\n'
            f"{ACTION_LIST}\n"
            "Look only at what is visible in this frame. Respond in EXACTLY this format:\n"
            "Reasoning: <one short sentence on what you actually see that supports your answer>\n"
            "Answer: <digit 1-4>"
        )
    return (
        f'Task: "{task_instruction}"\n'
        f"{ACTION_LIST}\n"
        "Look only at what is visible in this frame. Which number (1-4) best matches? "
        "Respond with ONLY the digit."
    )
# ==============================================================================


def parse_label(raw_text, labels):
    text = raw_text.strip().lower()
    m = re.search(r"[1-4]", text)
    if m and m.group(0) in NUMBERED_LABELS:
        return NUMBERED_LABELS[m.group(0)]
    for lab in labels:
        if text == lab.lower():
            return lab
    for lab in labels:
        if lab.lower() in text:
            return lab
    return None


def parse_reasoning_and_label(raw_text, labels):
    reasoning_m = re.search(r"Reasoning:\s*(.*)", raw_text, re.IGNORECASE)
    answer_m = re.search(r"Answer:\s*([1-4])", raw_text, re.IGNORECASE)
    reasoning = reasoning_m.group(1).strip().split("\n")[0] if reasoning_m else ""
    if answer_m:
        pred = NUMBERED_LABELS.get(answer_m.group(1))
    else:
        pred = parse_label(raw_text, labels)  # fallback if format wasn't followed
    return reasoning, pred


def main():
    with open(f"{DATA_DIR}/ground_truth.json") as f:
        gt = json.load(f)

    model_id = MODEL_ID
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0"
    )

    user_prompt = build_user_prompt(TASK_INSTRUCTION, LABELS)
    items = [it for it in gt["per_second"] if it["skill_description"] is not None]

    predictions = []
    correct = 0
    total = 0
    for idx, item in enumerate(items):
        window_items = items[max(0, idx - WINDOW + 1):idx + 1]
        while len(window_items) < WINDOW:
            window_items = [window_items[0]] + window_items  # pad with first frame at episode start
        content = []
        video_metadata = None
        if NATIVE_VIDEO:
            frames = [Image.open(w["frame_path"]).convert("RGB") for w in window_items]
            content.append({"type": "video", "video": frames})
            # tell the model these frames are really 1s apart, not a default-assumed 24fps clip
            video_metadata = [VideoMetadata(total_num_frames=len(frames), fps=1.0, duration=float(len(frames)))]
        elif STITCH:
            imgs = [Image.open(w["frame_path"]).convert("RGB") for w in window_items]
            h = min(im.height for im in imgs)
            imgs = [im.resize((int(im.width * h / im.height), h)) for im in imgs]
            total_w = sum(im.width for im in imgs) + 4 * (len(imgs) - 1)
            stitched = Image.new("RGB", (total_w, h), (255, 255, 0))
            x = 0
            for im in imgs:
                stitched.paste(im, (x, 0))
                x += im.width + 4
            content.append({"type": "image", "image": stitched})
        else:
            for w in window_items:
                tag = "Now" if w is item else "Before"
                content.append({"type": "text", "text": f"[{tag}]"})
                content.append({"type": "image", "image": Image.open(w["frame_path"]).convert("RGB")})
        content.append({"type": "text", "text": user_prompt})

        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        for i, ex in enumerate(FEW_SHOT_EXAMPLES):
            prefix = f'Task: "{TASK_INSTRUCTION}"\n{ACTION_LIST}\n' if i == 0 else ""
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prefix}Example frame - which number (1-4)? Respond with ONLY the digit."},
                    {"type": "image", "image": Image.open(ex["frame_path"]).convert("RGB")},
                ],
            })
            messages.append({"role": "assistant", "content": [{"type": "text", "text": ex["label_num"]}]})
        lead_in = [{"type": "text", "text": "Now classify a new frame the same way."}] if FEW_SHOT_EXAMPLES else []
        messages.append({"role": "user", "content": lead_in + content})
        template_kwargs = {"video_metadata": video_metadata} if video_metadata else {}
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", **template_kwargs
        )
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        max_new = 80 if ASK_FOR_REASONING else 16
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        raw = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        if ASK_FOR_REASONING:
            reasoning, pred = parse_reasoning_and_label(raw, LABELS)
        else:
            reasoning, pred = "", parse_label(raw, LABELS)

        gt_label = item["skill_description"]
        is_correct = pred == gt_label
        correct += int(is_correct)
        total += 1
        predictions.append({
            "sec": item["sec"], "gt": gt_label, "pred": pred,
            "reasoning": reasoning, "raw": raw.strip(), "correct": is_correct,
        })
        mark = "OK " if is_correct else "XX "
        if ASK_FOR_REASONING:
            print(f"{mark} sec={item['sec']:3d} gt={gt_label:15s} pred={str(pred):15s} reasoning={reasoning!r}")
        else:
            print(f"{mark} sec={item['sec']:3d} gt={gt_label:15s} pred={str(pred):15s} raw={raw.strip()!r}")

    acc = correct / total if total else 0.0
    print(f"\n=== {PROMPT_VERSION}: accuracy {correct}/{total} = {acc:.3f} ===")

    with open(f"{RESULTS_DIR}/{PROMPT_VERSION}.json", "w") as f:
        json.dump({
            "version": PROMPT_VERSION, "window": WINDOW,
            "system_prompt": SYSTEM_PROMPT, "user_prompt": user_prompt,
            "accuracy": acc, "correct": correct, "total": total,
            "predictions": predictions,
        }, f, indent=2)


if __name__ == "__main__":
    main()
