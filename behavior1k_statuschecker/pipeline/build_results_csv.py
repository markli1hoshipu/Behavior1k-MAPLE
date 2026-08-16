"""Build model-comparison CSVs: one row per (model, task), columns for overall
(verb+object), skill (verb-only), and object (object-only) accuracy/macro recall.
Joins results/lora_v3_<tag>_task-*.json (overall) with the *_verbonly_summary.json
and *_objectonly_summary.json produced by compute_verb_only_metrics.py /
compute_object_only_metrics.py.

Writes into results/csv/, named by train/eval details (episode ranges, epoch
checkpoints compared) rather than the internal result tags.
"""
import csv
import glob
import json
import os

ROOT = "/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker"
RESULTS_DIR = f"{ROOT}/results"
CSV_DIR = f"{RESULTS_DIR}/csv"
os.makedirs(CSV_DIR, exist_ok=True)

# (result_tag, model_label) per comparison group, in the order they should appear
GROUPS = {
    "100task_train1-20ep_eval21-40ep_epoch0-1-2": [
        ("baseeval", "base (0 epochs, no LoRA)"),
        ("epoch0eval", "LoRA v3 (1 epoch)"),
        ("fulleval", "LoRA v3 (2 epochs, final)"),
    ],
    "10task_train1-20ep_eval21-40ep_epoch0-2-4": [
        ("10task_base", "base (0 epochs, no LoRA)"),
        ("10task_2ep", "LoRA v3 10task (2 epochs)"),
        ("10task_4ep", "LoRA v3 10task (4 epochs, final)"),
    ],
}


def load_overall(tag):
    """task_chunk -> {task_name, n_classes, accuracy, macro_recall}"""
    out = {}
    for fp in sorted(glob.glob(f"{RESULTS_DIR}/lora_v3_{tag}_task-*.json")):
        d = json.load(open(fp))
        out[d["task_chunk"]] = {
            "task_name": d["task_name"], "n_classes": d["n_classes"],
            "accuracy": d["accuracy"], "macro_recall": d["macro_recall"],
        }
    return out


def load_sub_summary(tag, kind):
    """kind: 'verbonly' or 'objectonly'. task_chunk -> {accuracy, macro_recall}"""
    fp = f"{RESULTS_DIR}/lora_v3_{tag}_{kind}_summary.json"
    if not os.path.exists(fp):
        return {}
    d = json.load(open(fp))
    return {tc: {"accuracy": v["accuracy"], "macro_recall": v["macro_recall"]} for tc, v in d["per_task"].items()}


def build_group_csv(out_name, tag_model_pairs):
    out_path = f"{CSV_DIR}/{out_name}.csv"
    rows = []
    for tag, model_label in tag_model_pairs:
        overall = load_overall(tag)
        verbonly = load_sub_summary(tag, "verbonly")
        objectonly = load_sub_summary(tag, "objectonly")
        for task_chunk in sorted(overall.keys()):
            o = overall[task_chunk]
            v = verbonly.get(task_chunk, {})
            j = objectonly.get(task_chunk, {})
            rows.append({
                "model": model_label,
                "task_chunk": task_chunk,
                "task_name": o["task_name"],
                "n_classes": o["n_classes"],
                "overall_acc": round(o["accuracy"], 4),
                "overall_macro": round(o["macro_recall"], 4),
                "skill_acc": round(v.get("accuracy", float("nan")), 4),
                "skill_macro": round(v.get("macro_recall", float("nan")), 4),
                "obj_acc": round(j.get("accuracy", float("nan")), 4),
                "obj_macro": round(j.get("macro_recall", float("nan")), 4),
            })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "task_chunk", "task_name", "n_classes",
            "overall_acc", "overall_macro", "skill_acc", "skill_macro", "obj_acc", "obj_macro",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    for out_name, tag_model_pairs in GROUPS.items():
        build_group_csv(out_name, tag_model_pairs)
