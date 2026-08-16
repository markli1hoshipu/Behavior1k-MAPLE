"""Build demo videos (same style as the earlier radio demo) for the tasks where
LoRA v2 (0.8B, held-out) scored above 80% macro recall: sorting_vegetables (82.5%),
putting_shoes_on_rack (83.3%), hanging_pictures (97.2%), make_microwave_popcorn (87.2%).

Frames are non-consecutive (per-class-capped subsample), so each is shown for exactly
1 second regardless of the real time gap between them - a highlight-reel style demo,
not a continuous playback.
"""
import json
import os
import re
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/shared_work/markhsp/behavior1k-statuschecker"
DEMO_DIR = f"{ROOT}/demo"
os.makedirs(DEMO_DIR, exist_ok=True)
FPS = 30

TASKS_ABOVE_80 = {
    "task-0020": ("Sort the vegetables from the baskets into the mixing bowls.", 0.825),
    "task-0022": ("Put the shoes on the shoe rack.", 0.833),
    "task-0034": ("Hang the poster on the wall nail.", 0.972),
    "task-0040": ("Make microwave popcorn: put the popcorn bag in the microwave and turn it on.", 0.872),
}


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
        "Look only at what is visible in this frame. Respond with ONLY the label, "
        "nothing else."
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
RED = (255, 80, 80)


def wrap(text, width_chars):
    out = []
    for line in text.split("\n"):
        out.extend(textwrap.wrap(line, width_chars) or [""])
    return out


def render_slide(sec, frame_bgr, prompt_text, gt, pred, is_correct):
    img = Image.new("RGB", (W, H), (0, 0, 0))
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb).resize((VIDEO_SIZE, VIDEO_SIZE))
    img.paste(frame_pil, ((W - VIDEO_SIZE) // 2, VIDEO_TOP))

    d = ImageDraw.Draw(img)
    box_color = GREEN if is_correct else RED
    d.rectangle(
        [(W - VIDEO_SIZE) // 2 - 2, VIDEO_TOP - 2,
         (W + VIDEO_SIZE) // 2 + 2, VIDEO_TOP + VIDEO_SIZE + 2],
        outline=box_color, width=3,
    )
    d.text((W // 2 - 60, VIDEO_TOP + VIDEO_SIZE + 8), f"t = {sec}s", font=FONT_HDR, fill=GREEN)

    d.line([(PANEL_W, TEXT_TOP), (PANEL_W, H)], fill=DIM_GREEN, width=2)
    d.line([(0, TEXT_TOP - 10), (W, TEXT_TOP - 10)], fill=DIM_GREEN, width=2)

    d.text((20, TEXT_TOP), "INPUT PROMPT (sent to LoRA-finetuned Qwen3.5-0.8B)", font=FONT_HDR, fill=GREEN)
    d.text((PANEL_W + 20, TEXT_TOP), "MODEL OUTPUT", font=FONT_HDR, fill=GREEN)

    y = TEXT_TOP + 44
    for line in wrap(prompt_text, 62):
        d.text((20, y), line, font=FONT, fill=GREEN)
        y += 26

    y = TEXT_TOP + 44
    out_color = GREEN if is_correct else RED
    d.text((PANEL_W + 20, y), f"Predicted: {pred}", font=FONT, fill=out_color)
    y += 34
    d.text((PANEL_W + 20, y), f"Ground truth: {gt}", font=FONT, fill=GREEN)
    y += 34
    d.text((PANEL_W + 20, y), f"{'CORRECT' if is_correct else 'WRONG'}", font=FONT_HDR, fill=out_color)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def build_task_video(task_chunk, instruction):
    with open(f"{ROOT}/results/multitask_lora_v2_heldout_{task_chunk}.json") as f:
        result = json.load(f)
    with open(f"{ROOT}/results/multitask_rules_best_{task_chunk}.md") as f:
        action_list_text = parse_rules(f.read())

    prompt_text = build_user_prompt(instruction, action_list_text)
    held_out_ep = result["held_out_episode"]
    preds = sorted(result["predictions"], key=lambda p: p["sec"])

    out_path = f"{DEMO_DIR}/lora_v2_{task_chunk}_{result['task_name'].replace(' ', '_')}.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for p in preds:
        frame_path = f"{ROOT}/multitask_data_v2/{task_chunk}/frames_{held_out_ep}/sec_{p['sec']:04d}.jpg"
        frame_bgr = cv2.imread(frame_path)
        if frame_bgr is None:
            continue
        slide = render_slide(p["sec"], frame_bgr, prompt_text, p["gt"], p["pred"], p["correct"])
        for _ in range(FPS):
            writer.write(slide)
    writer.release()
    print(f"wrote {out_path} ({len(preds)}s @ {FPS}fps, macro_recall={result['macro_recall']:.3f})")


if __name__ == "__main__":
    for task_chunk, (instruction, macro) in TASKS_ABOVE_80.items():
        build_task_video(task_chunk, instruction)
