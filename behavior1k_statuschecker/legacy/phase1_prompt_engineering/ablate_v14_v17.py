"""One-factor-at-a-time ablation between v14 (45.8%) and v17 (23.7%).

Each variant swaps exactly one rule of v14 for the corresponding v17 rule (or vice
versa), so any accuracy/prediction delta is attributable to that single rule. All
variants share one model load and are scored against the same ground truth.
"""
import json
import os
import re
import sys
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

DATA_DIR = "/shared_work/markhsp/behavior1k-statuschecker/data/episode_00000010"
RESULTS_DIR = "/shared_work/markhsp/behavior1k-statuschecker/results"
TASK_INSTRUCTION = "Turn on the radio receiver that's on the table in the living room."
LABELS = ["move to", "pick up from", "press", "place on"]
NUMBERED_LABELS = {"1": "move to", "2": "pick up from", "3": "press", "4": "place on"}
MODEL_ID = "Qwen/Qwen3.5-2B"

SYSTEM_PROMPT = (
    "You are annotating individual frames from a fixed head-mounted camera on a "
    "mobile manipulator robot. Classify the robot's current subskill using ONLY "
    "the visual evidence in this single frame."
)

# --- v14 rules (the working baseline) ---
R1_14 = ("1. move to - no robot arm is visible anywhere in the frame; you see a wide view "
         "of the room (floor, walls, couch) with the red radio small, far away, or not "
         "visible at all.\n")
R2_14 = ("2. pick up from - a robot arm/gripper IS visible, reaching toward or holding the "
         "red radio, but the radio's round dial is NOT facing the camera (you only see the "
         "plain red-and-white striped body, not the dial face).\n")
R3_14 = ("3. press - the radio's flat face with its round dial IS facing the camera, and a "
         "gripper is directly on that dial. This is the only step where the gripper touches "
         "the round dial. The radio does not need to be lifted for this - it can happen "
         "while the radio still sits on the table.\n")
R4_14 = ("4. place on - the radio is resting back down on the table (or being lowered onto "
         "it) and the gripper is pulling away rather than touching the dial. If the dial is "
         "visible, it may show a small green light now on (this means the radio was already "
         "successfully pressed).\n")
TB_14 = "Tie-break: if no arm is visible at all anywhere in the frame, answer 1 (move to)."

# --- v17 rules (the regression) ---
R1_17 = ("1. move to - no robot arm is visible anywhere in the frame; you see a wide view "
         "of the room with the red radio small, far away, or not visible at all.\n")
R2_17 = ("2. pick up from - an arm/gripper is visible and reaching toward or holding the "
         "plain body of the radio, standing upright, its round dial not facing the camera.\n")
R3_17 = ("3. press - either the radio is lifted up in a gripper, or its round dial IS facing "
         "the camera with a gripper on that dial.\n")
R4_17 = ("4. place on - an arm is at the radio near the table, and the radio itself is "
         "tilted or leaning to one side rather than standing straight up and vertical.\n")
TB_17 = ("Tie-break: if no arm is visible, answer 1. If an arm is at the radio on the table "
         "and you can't tell whether it's grabbing or releasing, answer 4.")

# v14 rule 2 with every colour word stripped, structure otherwise identical
R2_14_NOCOLOR = ("2. pick up from - a robot arm/gripper IS visible, reaching toward or "
                 "holding the radio, but the radio's round dial is NOT facing the camera "
                 "(you only see the plain body, not the dial face).\n")

# --- round 2: decompose rule 2 one lexical feature at a time ---
# v14 r2 with the emphatic capitals removed (IS -> is, NOT -> not), nothing else changed
R2_14_LOWER = ("2. pick up from - a robot arm/gripper is visible, reaching toward or holding "
               "the red radio, but the radio's round dial is not facing the camera (you only "
               "see the plain red-and-white striped body, not the dial face).\n")
# v14 r2 with the restating parenthetical deleted
R2_14_NOPAREN = ("2. pick up from - a robot arm/gripper IS visible, reaching toward or "
                 "holding the red radio, but the radio's round dial is NOT facing the "
                 "camera.\n")
# v14 r2 with v17's extra "standing upright" conjunct bolted on
R2_14_UPRIGHT = ("2. pick up from - a robot arm/gripper IS visible, reaching toward or "
                 "holding the red radio, standing upright, but the radio's round dial is NOT "
                 "facing the camera (you only see the plain red-and-white striped body, not "
                 "the dial face).\n")
# v17 r2 with only the emphatic capitals added back
R2_17_CAPS = ("2. pick up from - an arm/gripper IS visible and reaching toward or holding the "
              "plain body of the radio, standing upright, its round dial NOT facing the "
              "camera.\n")
# v17 r2 with only "standing upright" removed
R2_17_NOUPRIGHT = ("2. pick up from - an arm/gripper is visible and reaching toward or "
                   "holding the plain body of the radio, its round dial not facing the "
                   "camera.\n")

# --- round 2: can ANY phrasing recover place-on? (v14 base, rule 4 only) ---
# shape/aspect phrasing instead of an angle
R4_ASPECT = ("4. place on - the radio is back down on the mat and its red shape is now about "
             "as wide as it is tall, a squat blob, instead of the tall narrow upright shape "
             "it has while being picked up.\n")
# blunt "fallen over" phrasing
R4_LYING = ("4. place on - the radio is back down on the mat and has fallen over onto its "
            "side; it is lying down, not standing up like a little tower.\n")
# contact/pose phrasing with an explicit contrast against rule 2
R4_LEAN = ("4. place on - a gripper is at the radio down on the mat, and the radio is "
           "leaning hard against that gripper at a slant. If the radio is standing straight "
           "up and down instead, it is 2 (pick up from), not 4.\n")
# truthful minimal description, no invented cue
R4_MINIMAL = ("4. place on - the radio has been set back down on the mat after the dial was "
              "already pressed, and a gripper is still beside it.\n")

VARIANTS = {
    # controls
    "A_v14_baseline":  (R1_14, R2_14, R3_14, R4_14, TB_14),
    "B_v17_baseline":  (R1_17, R2_17, R3_17, R4_17, TB_17),
    # v14 with exactly one v17 rule swapped in
    "C_v14_r1of17":    (R1_17, R2_14, R3_14, R4_14, TB_14),
    "D_v14_r2of17":    (R1_14, R2_17, R3_14, R4_14, TB_14),
    "E_v14_r3of17":    (R1_14, R2_14, R3_17, R4_14, TB_14),
    "F_v14_r4of17":    (R1_14, R2_14, R3_14, R4_17, TB_14),
    "G_v14_tbof17":    (R1_14, R2_14, R3_14, R4_14, TB_17),
    # v17 with exactly one v14 rule swapped back in
    "H_v17_r2of14":    (R1_17, R2_14, R3_17, R4_17, TB_17),
    "I_v17_r3of14":    (R1_17, R2_17, R3_14, R4_17, TB_17),
    "J_v17_r4of14":    (R1_17, R2_17, R3_17, R4_14, TB_17),
    # colour-token isolation: v14 rule 2 minus the words "red"/"red-and-white striped"
    "K_v14_r2nocolor": (R1_14, R2_14_NOCOLOR, R3_14, R4_14, TB_14),
    # round 2a: which lexical feature of rule 2 carries the effect
    "L_r2_lowercase":  (R1_14, R2_14_LOWER, R3_14, R4_14, TB_14),
    "M_r2_noparen":    (R1_14, R2_14_NOPAREN, R3_14, R4_14, TB_14),
    "N_r2_plusupright": (R1_14, R2_14_UPRIGHT, R3_14, R4_14, TB_14),
    "O_v17_r2caps":    (R1_17, R2_17_CAPS, R3_17, R4_17, TB_17),
    "P_v17_r2nouprght": (R1_17, R2_17_NOUPRIGHT, R3_17, R4_17, TB_17),
    # round 2b: can any phrasing of rule 4 recover place-on at all?
    "Q_r4_aspect":     (R1_14, R2_14, R3_14, R4_ASPECT, TB_14),
    "R_r4_lying":      (R1_14, R2_14, R3_14, R4_LYING, TB_14),
    "S_r4_lean":       (R1_14, R2_14, R3_14, R4_LEAN, TB_14),
    "T_r4_minimal":    (R1_14, R2_14, R3_14, R4_MINIMAL, TB_14),
}


def build_prompt(rules):
    r1, r2, r3, r4, tb = rules
    return (
        f'Task: "{TASK_INSTRUCTION}"\n'
        "The robot's subskill right now is always exactly one of:\n"
        f"{r1}{r2}{r3}{r4}{tb}\n"
        "Look only at what is visible in this frame. Which number (1-4) best matches? "
        "Respond with ONLY the digit."
    )


def parse_label(raw_text):
    text = raw_text.strip().lower()
    m = re.search(r"[1-4]", text)
    if m:
        return NUMBERED_LABELS[m.group(0)]
    for lab in LABELS:
        if lab.lower() in text:
            return lab
    return None


def macro_recall(preds, gts):
    rec = {}
    for lab in LABELS:
        idx = [i for i, g in enumerate(gts) if g == lab]
        rec[lab] = sum(preds[i] == lab for i in idx) / len(idx) if idx else float("nan")
    return rec, sum(rec.values()) / len(rec)


def main():
    names = sys.argv[1:] or list(VARIANTS)
    with open(f"{DATA_DIR}/ground_truth.json") as f:
        gt = json.load(f)
    items = [it for it in gt["per_second"] if it["skill_description"] is not None]
    images = [Image.open(it["frame_path"]).convert("RGB") for it in items]
    gts = [it["skill_description"] for it in items]

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0"
    )

    out_all = {}
    for name in names:
        prompt = build_prompt(VARIANTS[name])
        preds = []
        for img in images:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "text", "text": "[Now]"},  # matches annotate.py WINDOW=1
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ]},
            ]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
                      for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            raw = processor.decode(out[0][inputs["input_ids"].shape[-1]:],
                                   skip_special_tokens=True)
            preds.append(parse_label(raw))

        acc = sum(p == g for p, g in zip(preds, gts)) / len(gts)
        rec, macro = macro_recall(preds, gts)
        short = {"move to": "M", "pick up from": "P", "press": "R", "place on": "L", None: "?"}
        print(f"{name:18s} acc={acc:.3f} macro={macro:.3f} "
              f"[M {rec['move to']:.2f} P {rec['pick up from']:.2f} "
              f"R {rec['press']:.2f} L {rec['place on']:.2f}]  "
              f"{''.join(short[p] for p in preds)}", flush=True)
        out_all[name] = {"prompt": prompt, "accuracy": acc, "macro_recall": macro,
                         "per_class_recall": rec,
                         "predictions": [{"sec": it["sec"], "gt": g, "pred": p}
                                         for it, g, p in zip(items, gts, preds)]}

    with open(f"{RESULTS_DIR}/ablation_v14_v17.json", "w") as f:
        json.dump(out_all, f, indent=2)


if __name__ == "__main__":
    main()
