"""Extract training frames for LoRA fine-tuning: every second (1fps) from every episode
in the episode plan (no per-episode frame cap, unlike extract_multitask.py which capped
at 60/episode for the fast prompt-testing loop). To avoid the long-duration classes
(mostly "move to") dominating the combined dataset, a per-class cap is applied AFTER
aggregating across all episodes for a task, not per-episode - so rare/short classes
(chop, pour, hang) keep every frame they have, while abundant ones are evenly subsampled
down to the cap.
"""
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
PER_CLASS_CAP = 60  # after aggregating all episodes for a task, cap any single class here

with open(f"{ROOT}/results/episode_plan.json") as f:
    EPISODE_PLAN = json.load(f)


def extract_episode_frames(task_chunk, episode, out_dir):
    """Every second, no cap. Returns list of {sec, skill_description, frame_path, episode}."""
    annot_path = f"{HF_SNAP}/annotations/{task_chunk}/{episode}.json"
    video_path = f"{HF_SNAP}/videos/{task_chunk}/observation.images.rgb.head/{episode}.mp4"
    if not (os.path.exists(annot_path) and os.path.exists(video_path)):
        print(f"  SKIP {task_chunk}/{episode}: files not found")
        return []

    with open(annot_path) as f:
        annot = json.load(f)
    valid_start, valid_end = annot["meta_data"]["valid_duration"]
    duration_s = (valid_end - valid_start) / FPS
    n_seconds = int(duration_s)

    segments = []
    for seg in annot["skill_annotation"]:
        fd = seg["frame_duration"]
        # frame_duration is normally [start, end], but some entries are a list of
        # disjoint [start, end] sub-ranges for the same label (e.g. two separate
        # "move to" bursts) - normalize both shapes to a list of ranges.
        ranges = fd if (isinstance(fd[0], list)) else [fd]
        label = seg["skill_description"][0] if seg["skill_description"] else None
        for start_f, end_f in ranges:
            segments.append({"start_s": start_f / FPS, "end_s": end_f / FPS, "skill_description": label})

    def label_at(sec):
        for seg in segments:
            if seg["start_s"] <= sec < seg["end_s"]:
                return seg["skill_description"]
        return None

    frames_dir = f"{out_dir}/frames_{episode}"
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    items = []
    for sec in range(n_seconds):
        label = label_at(sec)
        if label is None:
            continue
        frame_idx = round((valid_start / FPS + sec) * FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_path = f"{frames_dir}/sec_{sec:04d}.jpg"
        cv2.imwrite(frame_path, frame)
        items.append({"sec": sec, "skill_description": label, "frame_path": frame_path, "episode": episode})
    cap.release()
    return items


def cap_per_class(items, cap):
    by_class = {}
    for it in items:
        by_class.setdefault(it["skill_description"], []).append(it)
    out = []
    for label, group in by_class.items():
        if len(group) <= cap:
            out.extend(group)
        else:
            step = len(group) / cap
            idxs = sorted(set(int(i * step) for i in range(cap)))
            out.extend(group[i] for i in idxs)
    return out, {label: (min(len(g), cap), len(g)) for label, g in by_class.items()}


def main():
    for task_chunk, episodes in EPISODE_PLAN.items():
        out_dir = f"{ROOT}/multitask_data_v2/{task_chunk}"
        os.makedirs(out_dir, exist_ok=True)
        all_items = []
        for ep in episodes:
            items = extract_episode_frames(task_chunk, ep, out_dir)
            all_items.extend(items)
            print(f"  {task_chunk}/{ep}: {len(items)} frames extracted")

        capped_items, class_stats = cap_per_class(all_items, PER_CLASS_CAP)

        with open(annot_path := f"{HF_SNAP}/annotations/{task_chunk}/{episodes[0]}.json") as f:
            task_name = json.load(f)["task_name"]

        distinct_labels = sorted(set(it["skill_description"] for it in capped_items))
        with open(f"{out_dir}/ground_truth.json", "w") as f:
            json.dump({
                "task_chunk": task_chunk, "task_name": task_name, "episodes": episodes,
                "n_episodes": len(episodes), "n_raw_frames": len(all_items),
                "n_capped_frames": len(capped_items), "distinct_labels": distinct_labels,
                "per_second": capped_items,
            }, f, indent=2)

        print(f"{task_chunk} ({task_name}): {len(episodes)} episodes, "
              f"{len(all_items)} raw -> {len(capped_items)} after per-class cap={PER_CLASS_CAP}")
        for label, (kept, total) in sorted(class_stats.items()):
            print(f"    {label:20s} {kept}/{total}")
        print()


if __name__ == "__main__":
    main()
