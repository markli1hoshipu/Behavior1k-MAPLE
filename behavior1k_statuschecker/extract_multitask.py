"""Generic per-second frame + ground-truth extraction for any task/episode, sampling
episodes down to a manageable frame budget for long tasks (e.g. make_pizza is 522s)."""
import glob
import json
import os

import cv2

ROOT = "/shared_work/markhsp/behavior1k-statuschecker"
HF_SNAP = glob.glob(
    "/shared_work/markhsp/behavior1k-statuschecker/hf_cache/hub/"
    "datasets--behavior-1k--2025-challenge-demos/snapshots/*"
)[0]
FPS = 30
MAX_FRAMES = 60  # cap per episode so long tasks stay tractable

TASKS = {
    "task-0000": ("episode_00000010", "turning_on_radio"),
    "task-0001": ("episode_00010010", "picking_up_trash"),
    "task-0005": ("episode_00050020", "setting_mousetraps"),
    "task-0020": ("episode_00200010", "sorting_vegetables"),
    "task-0022": ("episode_00220020", "putting_shoes_on_rack"),
    "task-0031": ("episode_00310010", "clean_boxing_gloves"),
    "task-0034": ("episode_00340020", "hanging_pictures"),
    "task-0040": ("episode_00400010", "make_microwave_popcorn"),
    "task-0042": ("episode_00420010", "chop_an_onion"),
    "task-0049": ("episode_00490010", "make_pizza"),
}


def extract(task_chunk, episode):
    annot_path = f"{HF_SNAP}/annotations/{task_chunk}/{episode}.json"
    video_path = f"{HF_SNAP}/videos/{task_chunk}/observation.images.rgb.head/{episode}.mp4"
    if not os.path.exists(annot_path):
        print(f"SKIP {task_chunk}: annotation not found at {annot_path}")
        return None
    if not os.path.exists(video_path):
        print(f"SKIP {task_chunk}: video not found at {video_path}")
        return None

    with open(annot_path) as f:
        annot = json.load(f)

    valid_start, valid_end = annot["meta_data"]["valid_duration"]
    duration_s = (valid_end - valid_start) / FPS
    n_seconds_total = int(duration_s)

    segments = []
    for seg in annot["skill_annotation"]:
        start_f, end_f = seg["frame_duration"]
        segments.append({
            "start_s": start_f / FPS,
            "end_s": end_f / FPS,
            "skill_description": seg["skill_description"][0] if seg["skill_description"] else None,
        })

    def label_at(sec):
        for seg in segments:
            if seg["start_s"] <= sec < seg["end_s"]:
                return seg["skill_description"]
        return None

    # sample down to MAX_FRAMES if the episode is longer than that
    if n_seconds_total > MAX_FRAMES:
        step = n_seconds_total / MAX_FRAMES
        sample_secs = sorted(set(int(i * step) for i in range(MAX_FRAMES)))
    else:
        sample_secs = list(range(n_seconds_total))

    out_dir = f"{ROOT}/multitask_data/{task_chunk}"
    frames_dir = f"{out_dir}/frames"
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    per_second = []
    for sec in sample_secs:
        frame_idx = round((valid_start / FPS + sec) * FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        label = label_at(sec)
        if label is None:
            continue
        frame_path = f"{frames_dir}/sec_{sec:04d}.jpg"
        cv2.imwrite(frame_path, frame)
        per_second.append({"sec": sec, "skill_description": label, "frame_path": frame_path})
    cap.release()

    distinct_labels = sorted(set(p["skill_description"] for p in per_second))

    with open(f"{out_dir}/ground_truth.json", "w") as f:
        json.dump({
            "task_chunk": task_chunk, "episode": episode,
            "task_name": annot["task_name"], "duration_s": duration_s,
            "n_seconds_total": n_seconds_total, "n_sampled": len(per_second),
            "distinct_labels": distinct_labels, "per_second": per_second,
        }, f, indent=2)

    print(f"{task_chunk} ({annot['task_name']}): {n_seconds_total}s total, "
          f"sampled {len(per_second)} frames, {len(distinct_labels)} distinct labels: {distinct_labels}")
    return out_dir


if __name__ == "__main__":
    for task_chunk, (episode, name) in TASKS.items():
        extract(task_chunk, episode)
