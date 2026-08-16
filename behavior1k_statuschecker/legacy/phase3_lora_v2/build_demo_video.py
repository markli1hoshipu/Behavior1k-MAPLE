"""Build a side-by-side demo: for the v10 (memory) run on episode_00000010,
render a text video (LEFT = exact input prompt sent that second, RIGHT = raw
VLM output that second, both green-on-black), timed 1:1 with a trimmed copy
of the original video so both can be played together in sync.

Outputs into demo/:
  origin.mp4     - source video, trimmed to the scored window (59s @ 30fps)
  annotated.mp4  - text panels, same duration/fps, aligned second-by-second
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
SRC_VIDEO = f"/shared_work/DATASETS/behavior-1k-embodiedAI-rollouts/videos/task-0000/observation.images.rgb.head/episode_{EPISODE}.mp4"
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


def build_user_prompt_v10(memory):
    """Exact reconstruction of annotate_memory.py's v10 build_user_prompt()."""
    memory_line = (
        f'Your notes from 1 second ago: "{memory}"\n'
        if memory else "This is the first frame of the episode - you have no prior notes.\n"
    )
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


# ---------------------------------------------------------------- rendering ----

W, H = 1800, 1500
PANEL_W = W // 2
VIDEO_SIZE = 640  # original frames are 720x720, scaled down slightly to fit
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

    # actual original-video frame, centered at the top
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

    d.text((20, TEXT_TOP), "INPUT PROMPT (sent to Qwen3.5-0.8B)", font=FONT_HDR, fill=GREEN)
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
    with open(f"{ROOT}/results/v10.json") as f:
        v10 = json.load(f)
    with open(f"{ROOT}/data/episode_{EPISODE}/ground_truth.json") as f:
        gt = json.load(f)
    preds = v10["predictions"]
    frame_by_sec = {it["sec"]: it["frame_path"] for it in gt["per_second"]}
    n_seconds = len(preds)

    # single self-contained video: original frame burned in, annotation text below it,
    # one "slide" held for exactly 1s (FPS frames)
    ann_path = f"{DEMO_DIR}/annotated.mp4"
    writer = cv2.VideoWriter(ann_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    memory = None
    for p in preds:
        prompt_text = build_user_prompt_v10(memory)
        output_text = p["raw"]
        memory = p["description"]  # matches v10's actual memory update rule
        frame_bgr = cv2.imread(frame_by_sec[p["sec"]])
        frame = render_slide(p["sec"], frame_bgr, prompt_text, output_text)
        for _ in range(FPS):
            writer.write(frame)
    writer.release()
    print(f"wrote {ann_path} ({n_seconds}s @ {FPS}fps)")


if __name__ == "__main__":
    main()
