"""Download + extract 1fps frames/ground-truth for task-0029 ("clean up your desk") -
a task NOT in the 10-task training set, for a genuine zero-shot generalization test of
lora_adapter_v2. Mirrors extract_multitask_v2.py's per-class cap logic, just scoped to
one new task and a handful of episodes (no training will happen on this data)."""
import glob
import json
import os

import cv2
from huggingface_hub import hf_hub_download

ROOT = "/shared_work/markhsp/behavior1k-statuschecker"
REPO = "behavior-1k/2025-challenge-demos"
TASK_CHUNK = "task-0029"
N_EPISODES = 3
FPS = 30
PER_CLASS_CAP = 60


def main():
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = fs.ls(f"datasets/{REPO}/annotations/{TASK_CHUNK}", detail=False)
    episodes = sorted(f.split("/")[-1].replace(".json", "") for f in files if f.endswith(".json"))[:N_EPISODES]
    print(f"episodes: {episodes}")

    for ep in episodes:
        hf_hub_download(REPO, f"annotations/{TASK_CHUNK}/{ep}.json", repo_type="dataset")
        hf_hub_download(REPO, f"videos/{TASK_CHUNK}/observation.images.rgb.head/{ep}.mp4", repo_type="dataset")

    hf_snap = glob.glob(f"{ROOT}/hf_cache/hub/datasets--behavior-1k--2025-challenge-demos/snapshots/*")[0]
    out_dir = f"{ROOT}/multitask_data_v2/{TASK_CHUNK}"
    os.makedirs(out_dir, exist_ok=True)

    all_items = []
    task_name = None
    for ep in episodes:
        annot_path = f"{hf_snap}/annotations/{TASK_CHUNK}/{ep}.json"
        video_path = f"{hf_snap}/videos/{TASK_CHUNK}/observation.images.rgb.head/{ep}.mp4"
        with open(annot_path) as f:
            annot = json.load(f)
        task_name = task_name or annot["task_name"]
        valid_start, valid_end = annot["meta_data"]["valid_duration"]
        n_seconds = int((valid_end - valid_start) / FPS)

        segments = []
        for seg in annot["skill_annotation"]:
            fd = seg["frame_duration"]
            ranges = fd if isinstance(fd[0], list) else [fd]
            label = seg["skill_description"][0] if seg["skill_description"] else None
            for start_f, end_f in ranges:
                segments.append({"start_s": start_f / FPS, "end_s": end_f / FPS, "skill_description": label})

        def label_at(sec):
            for seg in segments:
                if seg["start_s"] <= sec < seg["end_s"]:
                    return seg["skill_description"]
            return None

        frames_dir = f"{out_dir}/frames_{ep}"
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
            items.append({"sec": sec, "skill_description": label, "frame_path": frame_path, "episode": ep})
        cap.release()
        all_items.extend(items)
        print(f"  {ep}: {len(items)} frames")

    by_class = {}
    for it in all_items:
        by_class.setdefault(it["skill_description"], []).append(it)
    capped = []
    for label, group in by_class.items():
        if len(group) <= PER_CLASS_CAP:
            capped.extend(group)
        else:
            step = len(group) / PER_CLASS_CAP
            idxs = sorted(set(int(i * step) for i in range(PER_CLASS_CAP)))
            capped.extend(group[i] for i in idxs)

    distinct_labels = sorted(set(it["skill_description"] for it in capped))
    with open(f"{out_dir}/ground_truth.json", "w") as f:
        json.dump({
            "task_chunk": TASK_CHUNK, "task_name": task_name, "episodes": episodes,
            "n_episodes": len(episodes), "n_raw_frames": len(all_items),
            "n_capped_frames": len(capped), "distinct_labels": distinct_labels,
            "per_second": capped,
        }, f, indent=2)

    print(f"\n{TASK_CHUNK} ({task_name}): {len(episodes)} episodes, "
          f"{len(all_items)} raw -> {len(capped)} after cap={PER_CLASS_CAP}")
    for label in distinct_labels:
        n = len(by_class[label])
        print(f"    {label:20s} {min(n, PER_CLASS_CAP)}/{n}")


if __name__ == "__main__":
    main()
