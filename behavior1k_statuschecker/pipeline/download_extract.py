"""Download + extract 1fps frames from the BEHAVIOR-2026 challenge dataset
(behavior-1k/2026-challenge-demos), which uses LeRobot v3 packed-video format:
many episodes' video frames are concatenated into a small number of per-task mp4
files (videos/observation.rgb.zed_link_camera_0/chunk-XXX/file-XXX.mp4), and each
episode occupies a [from_timestamp, to_timestamp) slice of one such file (looked up
via meta/episodes/chunk-{task_index:03d}/file-000.parquet).

For each task we only download the first video file(s) that cover the first
N_EPISODES_PER_TASK episodes (by episode_index) - usually just 1 file, since each
file packs ~35-45 episodes of a ~1-6 minute task.

Training target is now "verb-norm": <skill_description> <main_object>, not just the
bare verb - e.g. "pick up from radio" instead of just "pick up from". The main object
is manipulating_object_id[0] if present (the thing actually being acted on), else
falling back to object_id[0][0] (e.g. the navigation target for "move to"). Object ids
in this dataset are "<category>_<6charmodelid>_<instanceidx>" or plain
"<category>_<instanceidx>" - normalized to just "<category>" (spaces for underscores)
by matching the longest prefix against the real BEHAVIOR category list, rather than a
regex guess (a regex confuses categories that happen to be 6 letters, like
"pillar_candle", with a random model-id suffix).

Usage: python3 download_extract.py [start_task_idx] [end_task_idx]  (both inclusive, default 0 99)

Episode range/output dir are overridable via env vars so this same (validated) logic
can build a second, disjoint eval set without duplicating code, e.g.:
  EPISODE_START=20 N_EPISODES_PER_TASK=20 OUT_ROOT=.../data_2026_eval python3 download_extract.py
"""
import json
import os
import sys

import cv2
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = "/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker"
OUT_ROOT = os.environ.get("OUT_ROOT", f"{ROOT}/data_2026")
REPO = "behavior-1k/2026-challenge-demos"
CAM = "observation.rgb.zed_link_camera_0"
FPS = 30
EPISODE_START = int(os.environ.get("EPISODE_START", "0"))
N_EPISODES_PER_TASK = int(os.environ.get("N_EPISODES_PER_TASK", "20"))
PER_CLASS_CAP = 60
CATEGORIES_PATH = "/shared_work/BEHAVIOR-2026/datasets/behavior-1k-assets/metadata/categories.txt"

os.makedirs(OUT_ROOT, exist_ok=True)

with open(CATEGORIES_PATH) as f:
    CATEGORY_SET = set(line.strip() for line in f if line.strip())


def normalize_object(object_id):
    """Strip the random-model-id/instance-index suffix off an object id by matching
    the longest prefix that's a real category name, e.g. 'coffee_table_koagbh_0' ->
    'coffee table', 'pillar_candle_90' -> 'pillar candle', 'radio_89' -> 'radio'."""
    tokens = object_id.split("_")
    for i in range(len(tokens) - 1, 0, -1):
        candidate = "_".join(tokens[:i])
        if candidate in CATEGORY_SET:
            return candidate.replace("_", " ")
    return object_id.replace("_", " ")


def _first_str(x):
    """Unwrap nested lists (e.g. multi-object manipulation like sweeping two log
    halves at once: manipulating_object_id == [['half_log_176_0', 'half_log_176_1']])
    down to the first plain object-id string."""
    while isinstance(x, list):
        if not x:
            return None
        x = x[0]
    return x


def main_object_for_segment(seg):
    manipulating = seg.get("manipulating_object_id") or []
    if manipulating:
        oid = _first_str(manipulating)
        if oid:
            return normalize_object(oid)
    object_id = seg.get("object_id") or []
    if object_id and object_id[0]:
        oid = _first_str(object_id[0])
        if oid:
            return normalize_object(oid)
    return None


def load_task_names():
    p = hf_hub_download(REPO, "meta/tasks.jsonl", repo_type="dataset")
    names = {}
    for line in open(p):
        d = json.loads(line)
        names[d["task_index"]] = (d["task_name"], d["task"])
    return names


def process_task(task_idx, task_name, task_instruction):
    task_chunk = f"task-{task_idx:04d}"
    out_dir = f"{OUT_ROOT}/{task_chunk}"
    if os.path.exists(f"{out_dir}/ground_truth.json"):
        print(f"[{task_chunk}] already done, skipping")
        return
    os.makedirs(out_dir, exist_ok=True)

    meta_p = hf_hub_download(REPO, f"meta/episodes/chunk-{task_idx:03d}/file-000.parquet", repo_type="dataset")
    df = pd.read_parquet(meta_p)
    df = df.sort_values("episode_index").iloc[EPISODE_START:EPISODE_START + N_EPISODES_PER_TASK]

    ck_col, fi_col = f"videos/{CAM}/chunk_index", f"videos/{CAM}/file_index"
    ft_col, tt_col = f"videos/{CAM}/from_timestamp", f"videos/{CAM}/to_timestamp"

    needed_files = sorted(set(zip(df[ck_col], df[fi_col])))
    video_paths = {}
    for ck, fi in needed_files:
        vp = hf_hub_download(REPO, f"videos/{CAM}/chunk-{ck:03d}/file-{fi:03d}.mp4", repo_type="dataset")
        video_paths[(ck, fi)] = vp

    all_items = []
    for _, row in df.iterrows():
        ep_idx = int(row["episode_index"])
        ann_path_rel = row["annotation_path"]
        try:
            ann_p = hf_hub_download(REPO, ann_path_rel, repo_type="dataset")
        except Exception as e:
            print(f"  SKIP ep {ep_idx}: annotation missing ({e})")
            continue
        annot = json.load(open(ann_p))
        valid_start, valid_end = annot["meta_data"]["valid_duration"]

        segments = []
        for seg in annot["skill_annotation"]:
            fd = seg["frame_duration"]
            ranges = fd if isinstance(fd[0], list) else [fd]
            verb = seg["skill_description"][0] if seg["skill_description"] else None
            obj = main_object_for_segment(seg)
            target = f"{verb} {obj}" if (verb and obj) else verb
            for start_f, end_f in ranges:
                segments.append({
                    "start_s": start_f / FPS, "end_s": end_f / FPS,
                    "verb": verb, "object": obj, "target": target,
                })

        def label_at(local_sec):
            for seg in segments:
                if seg["start_s"] <= local_sec < seg["end_s"]:
                    return seg
            return None

        from_ts = row[ft_col]
        video_path = video_paths[(int(row[ck_col]), int(row[fi_col]))]
        cap = cv2.VideoCapture(video_path)

        n_local_seconds = int(valid_end / FPS)
        ep_name = os.path.basename(ann_path_rel).replace(".json", "")
        frames_dir = f"{out_dir}/frames_{ep_name}"
        os.makedirs(frames_dir, exist_ok=True)

        items = []
        for local_sec in range(n_local_seconds):
            seg = label_at(local_sec)
            if seg is None:
                continue
            global_t = from_ts + local_sec
            frame_idx = round(global_t * FPS)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frame_path = f"{frames_dir}/sec_{local_sec:04d}.jpg"
            cv2.imwrite(frame_path, frame)
            items.append({
                "sec": local_sec, "verb": seg["verb"], "object": seg["object"], "target": seg["target"],
                "skill_description": seg["target"],  # kept for backward-compat field name
                "frame_path": frame_path, "episode": ep_name,
            })
        cap.release()
        all_items.extend(items)
        print(f"  [{task_chunk}] {ep_name}: {len(items)} frames")

    if not all_items:
        print(f"[{task_chunk}] NO DATA, skipping save")
        return

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
    episodes_used = sorted(set(it["episode"] for it in all_items))
    with open(f"{out_dir}/ground_truth.json", "w") as f:
        json.dump({
            "task_chunk": task_chunk, "task_name": task_name, "task_instruction": task_instruction,
            "episodes": episodes_used, "n_episodes": len(episodes_used),
            "n_raw_frames": len(all_items), "n_capped_frames": len(capped),
            "distinct_labels": distinct_labels, "per_second": capped,
        }, f, indent=2)

    print(f"[{task_chunk}] ({task_name}): {len(episodes_used)} episodes, "
          f"{len(all_items)} raw -> {len(capped)} after cap={PER_CLASS_CAP}")


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 99
    task_names = load_task_names()
    for task_idx in range(start, end + 1):
        if task_idx not in task_names:
            continue
        name, instr = task_names[task_idx]
        try:
            process_task(task_idx, name, instr)
        except Exception as e:
            print(f"[task-{task_idx:04d}] FAILED: {e}")


if __name__ == "__main__":
    main()
