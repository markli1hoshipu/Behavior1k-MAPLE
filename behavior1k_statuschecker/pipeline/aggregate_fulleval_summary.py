"""Scan results/lora_v3_{RESULT_TAG}_task-*.json (written by sharded eval_lora_v3_fulleval.py
runs, one per SLURM array task) and build the combined summary across all 100 tasks."""
import glob
import json
import os
import sys

RESULTS_DIR = "/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker/results"
RESULT_TAG = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RESULT_TAG", "fulleval")


def main():
    files = sorted(glob.glob(f"{RESULTS_DIR}/lora_v3_{RESULT_TAG}_task-*.json"))
    summary = []
    for fp in files:
        d = json.load(open(fp))
        summary.append((d["task_chunk"], d["accuracy"], d["macro_recall"]))

    print(f"===== SUMMARY: LoRA v3 [{RESULT_TAG}] (100 tasks, verb+object) ({len(summary)}/100 tasks found) =====")
    total_macro = 0
    n_pass = 0
    for tc, acc, macro in summary:
        status = "PASS>=70%" if macro >= 0.70 else ""
        if macro >= 0.70:
            n_pass += 1
        total_macro += macro
        print(f"{tc:12s} acc={acc:.3f}  macro={macro:.3f}  {status}")
    avg = total_macro / len(summary) if summary else 0.0
    print(f"\nLoRA v3 [{RESULT_TAG}] average macro: {avg:.3f}   tasks >=70%: {n_pass}/{len(summary)}")

    missing = 100 - len(summary)
    if missing:
        print(f"WARNING: {missing} tasks missing - not all shards finished yet.")

    with open(f"{RESULTS_DIR}/lora_v3_{RESULT_TAG}_summary.json", "w") as f:
        json.dump({
            "n_tasks": len(summary), "avg_macro": avg, "n_pass_70": n_pass,
            "per_task": {tc: {"accuracy": acc, "macro_recall": macro} for tc, acc, macro in summary},
        }, f, indent=2)
    print(f"\nSaved results/lora_v3_{RESULT_TAG}_summary.json")


if __name__ == "__main__":
    main()
