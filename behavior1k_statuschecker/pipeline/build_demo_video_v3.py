"""Build demo videos (same green-text-on-black style as before) for the 10-task
4-epoch experiment's tasks that hit >=70% verb+object macro recall: moving_boxes
(76.7%), clean_a_trumpet (74.4%), re_shelving_library_books (73.6%),
cook_a_frozen_pie (77.1%), installing_a_fax_machine (71.8%).

Unlike the earlier single-held-out-episode demos, this eval set spans MANY episodes
per task (episodes 21-40), so frames are sorted by (episode, sec) - each episode's
frames play consecutively, then move to the next episode - rather than interleaved.
Each frame is shown for 1 second regardless of gaps (highlight-reel style, matching
the per-class-capped subsample).
"""
import json
import os
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/shared_work/markhsp/behavior1k-statuschecker"
DEMO_DIR = f"{ROOT}/demo"
TRAIN_DATA_ROOT = f"{ROOT}/data_2026"
EVAL_DATA_ROOT = f"{ROOT}/data_2026_eval"
os.makedirs(DEMO_DIR, exist_ok=True)
FPS = 30
MAX_FRAMES = 90  # cap video length to a ~90s highlight reel, evenly sampled

TASKS_ABOVE_70 = {
    "task-0016": 0.767,
    "task-0037": 0.744,
    "task-0055": 0.736,
    "task-0072": 0.771,
    "task-0089": 0.718,
}


def build_user_prompt(task_instruction, labels):
    label_list = "\n".join(f"{i+1}. {lab}" for i, lab in enumerate(labels))
    return (
        f'Task: "{task_instruction}"\n'
        f"The robot's subskill right now is always exactly one of:\n{label_list}\n"
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


def render_slide(sec, episode, frame_bgr, prompt_text, gt, pred, is_correct):
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
    d.text((W // 2 - 160, VIDEO_TOP + VIDEO_SIZE + 8), f"{episode}  t = {sec}s", font=FONT_HDR, fill=GREEN)

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


def build_task_video(task_chunk, macro):
    with open(f"{ROOT}/results/lora_v3_10task_4ep_{task_chunk}.json") as f:
        result = json.load(f)
    with open(f"{TRAIN_DATA_ROOT}/{task_chunk}/ground_truth.json") as f:
        train_gt = json.load(f)

    prompt_text = build_user_prompt(train_gt["task_instruction"], train_gt["distinct_labels"])
    preds = sorted(result["predictions"], key=lambda p: (p["episode"], p["sec"]))
    if len(preds) > MAX_FRAMES:
        step = len(preds) / MAX_FRAMES
        idxs = sorted(set(int(i * step) for i in range(MAX_FRAMES)))
        preds = [preds[i] for i in idxs]

    out_path = f"{DEMO_DIR}/lora_v3_10task4ep_{task_chunk}_{result['task_name'].replace(' ', '_')}.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n_written = 0
    for p in preds:
        frame_path = f"{EVAL_DATA_ROOT}/{task_chunk}/frames_{p['episode']}/sec_{p['sec']:04d}.jpg"
        frame_bgr = cv2.imread(frame_path)
        if frame_bgr is None:
            continue
        slide = render_slide(p["sec"], p["episode"], frame_bgr, prompt_text, p["gt"], p["pred"], p["correct"])
        for _ in range(FPS):
            writer.write(slide)
        n_written += 1
    writer.release()
    print(f"wrote {out_path} ({n_written} frames @ {FPS}fps, macro_recall={result['macro_recall']:.3f})")


if __name__ == "__main__":
    for task_chunk, macro in TASKS_ABOVE_70.items():
        build_task_video(task_chunk, macro)
