"""Same demo format as build_demo_video.py (original frame on top, input prompt
left / VLM output right, green-on-black, one update per second), but for the
Claude Haiku 4.5 run (results/haiku45_v10.json) instead of Qwen3.5-0.8B.
Writes to a separate file so the Qwen demo is untouched.
"""
import glob
import json
import os
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EPISODE = "00000010"
FPS = 30
ROOT = "/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker"
DEMO_DIR = f"{ROOT}/demo"
os.makedirs(DEMO_DIR, exist_ok=True)

TASK_INSTRUCTION = "Turn on the radio receiver that's on the table in the living room."
ANNOT_GLOB = "/shared_work/DATASETS/behavior-1k-embodiedAI-rollouts/annotations/task-0000/*.json"

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


def build_user_prompt_haiku(frame_path, memory):
    """Exact reconstruction of run_haiku_annotator.py's build_user_prompt()."""
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


# ---------------------------------------------------------------- rendering ----

W, H = 1800, 1500
PANEL_W = W // 2
VIDEO_SIZE = 640
VIDEO_TOP = 20
TEXT_TOP = VIDEO_TOP + VIDEO_SIZE + 50
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT = ImageFont.truetype(FONT_PATH, 20)
FONT_HDR = ImageFont.truetype(FONT_PATH, 26)
GREEN = (60, 255, 60)
DIM_GREEN = (30, 140, 30)


def wrap(text, width_chars):
    out = []
    for line in text.split("\n"):
        out.extend(textwrap.wrap(line, width_chars) or [""])
    return out


def render_slide(sec, frame_bgr, prompt_text, output_text):
    img = Image.new("RGB", (W, H), (0, 0, 0))

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb).resize((VIDEO_SIZE, VIDEO_SIZE))
    img.paste(frame_pil, ((W - VIDEO_SIZE) // 2, VIDEO_TOP))

    d = ImageDraw.Draw(img)
    d.rectangle(
        [(W - VIDEO_SIZE) // 2 - 2, VIDEO_TOP - 2,
         (W + VIDEO_SIZE) // 2 + 2, VIDEO_TOP + VIDEO_SIZE + 2],
        outline=GREEN, width=2,
    )
    d.text((W // 2 - 60, VIDEO_TOP + VIDEO_SIZE + 8), f"t = {sec}s", font=FONT_HDR, fill=GREEN)

    d.line([(PANEL_W, TEXT_TOP), (PANEL_W, H)], fill=DIM_GREEN, width=2)
    d.line([(0, TEXT_TOP - 10), (W, TEXT_TOP - 10)], fill=DIM_GREEN, width=2)

    d.text((20, TEXT_TOP), "INPUT PROMPT (sent to Claude Haiku 4.5)", font=FONT_HDR, fill=GREEN)
    d.text((PANEL_W + 20, TEXT_TOP), "VLM OUTPUT (raw)", font=FONT_HDR, fill=GREEN)

    y = TEXT_TOP + 44
    for line in wrap(prompt_text, 62):
        d.text((20, y), line, font=FONT, fill=GREEN)
        y += 26

    y = TEXT_TOP + 44
    for line in wrap(output_text, 62):
        d.text((PANEL_W + 20, y), line, font=FONT, fill=GREEN)
        y += 26

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    with open(f"{ROOT}/results/haiku45_v10.json") as f:
        haiku = json.load(f)
    with open(f"{ROOT}/data/episode_{EPISODE}/ground_truth.json") as f:
        gt = json.load(f)
    preds = haiku["predictions"]
    frame_by_sec = {it["sec"]: it["frame_path"] for it in gt["per_second"]}
    n_seconds = len(preds)

    ann_path = f"{DEMO_DIR}/annotated_haiku.mp4"
    writer = cv2.VideoWriter(ann_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    memory = None
    for p in preds:
        prompt_text = build_user_prompt_haiku(frame_by_sec[p["sec"]], memory)
        output_text = p["raw"]
        memory = p["description"]  # matches the actual memory update rule used at run time
        frame_bgr = cv2.imread(frame_by_sec[p["sec"]])
        frame = render_slide(p["sec"], frame_bgr, prompt_text, output_text)
        for _ in range(FPS):
            writer.write(frame)
    writer.release()
    print(f"wrote {ann_path} ({n_seconds}s @ {FPS}fps)")


if __name__ == "__main__":
    main()
