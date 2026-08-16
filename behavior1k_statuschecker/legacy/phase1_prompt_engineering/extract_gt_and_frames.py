"""Extract 1fps frames + per-second ground-truth skill labels for one episode.

Usage: python extract_gt_and_frames.py <episode_id, e.g. 00000010>
"""
import json
import sys
import os
import cv2

EPISODE = sys.argv[1] if len(sys.argv) > 1 else "00000010"
TASK_CHUNK = "task-0000"
ROOT = "/shared_work/DATASETS/behavior-1k-embodiedAI-rollouts"
ANNOT_PATH = f"{ROOT}/annotations/{TASK_CHUNK}/episode_{EPISODE}.json"
VIDEO_PATH = f"{ROOT}/videos/{TASK_CHUNK}/observation.images.rgb.head/episode_{EPISODE}.mp4"
OUT_DIR = f"/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker/data/episode_{EPISODE}"
FRAMES_DIR = f"{OUT_DIR}/frames_1fps"

os.makedirs(FRAMES_DIR, exist_ok=True)

with open(ANNOT_PATH) as f:
    annot = json.load(f)

fps = 30  # dataset-wide, from meta/info.json
valid_start, valid_end = annot["meta_data"]["valid_duration"]
duration_s = valid_end / fps

segments = []
for seg in annot["skill_annotation"]:
    start_f, end_f = seg["frame_duration"]
    segments.append({
        "start_s": start_f / fps,
        "end_s": end_f / fps,
        "skill_description": seg["skill_description"][0] if seg["skill_description"] else None,
        "skill_type": seg["skill_type"][0] if seg["skill_type"] else None,
        "object_id": seg["object_id"][0] if seg["object_id"] else [],
        "manipulating_object_id": seg.get("manipulating_object_id", []),
    })

def label_at(sec):
    for seg in segments:
        if seg["start_s"] <= sec < seg["end_s"]:
            return seg
    # last segment's end is often inclusive of final second
    if segments and sec >= segments[-1]["end_s"] - 1e-6:
        return segments[-1]
    return None

n_seconds = int(duration_s)  # 0 .. n_seconds-1 full seconds within valid range
gt = []
cap = cv2.VideoCapture(VIDEO_PATH)
for sec in range(n_seconds):
    frame_idx = round(sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        print(f"WARN: could not read frame {frame_idx} (sec {sec})")
        continue
    frame_path = f"{FRAMES_DIR}/sec_{sec:04d}.jpg"
    cv2.imwrite(frame_path, frame)
    lab = label_at(sec)
    gt.append({
        "sec": sec,
        "frame_idx": frame_idx,
        "frame_path": frame_path,
        "skill_description": lab["skill_description"] if lab else None,
        "skill_type": lab["skill_type"] if lab else None,
        "object_id": lab["object_id"] if lab else [],
    })
cap.release()

with open(f"{OUT_DIR}/ground_truth.json", "w") as f:
    json.dump({
        "episode": EPISODE,
        "task_name": annot["task_name"],
        "fps": fps,
        "duration_s": duration_s,
        "segments": segments,
        "per_second": gt,
    }, f, indent=2)

print(f"Wrote {len(gt)} frames to {FRAMES_DIR}")
print(f"Wrote ground truth to {OUT_DIR}/ground_truth.json")
print(f"Task: {annot['task_name']}")
print(f"Segments ({len(segments)}):")
for s in segments:
    print(f"  [{s['start_s']:.1f}-{s['end_s']:.1f}s] {s['skill_description']} ({s['skill_type']}) obj={s['object_id']}")
