"""Rescore the already-computed LoRA v3 predictions (results/lora_v3_fulleval_task-*.json)
using ONLY the verb (skill), ignoring the object - to see how much of the 62.2% combined
verb+object score is "getting the object right too" vs. "getting the skill right".

Ground-truth verb comes directly from data_2026_eval's per_second items (matched by
episode+sec). Predicted verb comes from a target->verb lookup built from the TRAINING
set's own per_second items (since predictions are always one of the training label
vocabulary, or None if unparsed).
"""
import glob
import json
import sys

ROOT = "/shared_work/markhsp/behavior1k-statuschecker"
RESULTS_DIR = f"{ROOT}/results"
TRAIN_DATA_ROOT = f"{ROOT}/data_2026"
EVAL_DATA_ROOT = f"{ROOT}/data_2026_eval"
RESULT_TAG = sys.argv[1] if len(sys.argv) > 1 else "fulleval"


def main():
    result_files = sorted(glob.glob(f"{RESULTS_DIR}/lora_v3_{RESULT_TAG}_task-*.json"))
    per_task_macro = {}

    for fp in result_files:
        result = json.load(open(fp))
        task_chunk = result["task_chunk"]

        with open(f"{TRAIN_DATA_ROOT}/{task_chunk}/ground_truth.json") as f:
            train_gt = json.load(f)
        target_to_verb = {it["target"]: it["verb"] for it in train_gt["per_second"]}

        with open(f"{EVAL_DATA_ROOT}/{task_chunk}/ground_truth.json") as f:
            eval_gt = json.load(f)
        gt_verb_by_key = {(it["episode"], it["sec"]): it["verb"] for it in eval_gt["per_second"]}

        by_class = {}
        correct = 0
        total = 0
        for p in result["predictions"]:
            key = (p["episode"], p["sec"])
            verb_gt = gt_verb_by_key.get(key)
            if verb_gt is None:
                continue
            verb_pred = target_to_verb.get(p["pred"]) if p["pred"] else None
            is_correct = verb_gt == verb_pred
            by_class.setdefault(verb_gt, [0, 0])
            by_class[verb_gt][1] += 1
            if is_correct:
                by_class[verb_gt][0] += 1
                correct += 1
            total += 1

        acc = correct / total if total else 0.0
        macro = sum(c / t for c, t in by_class.values()) / len(by_class) if by_class else 0.0
        per_task_macro[task_chunk] = {"accuracy": acc, "macro_recall": macro, "n_verb_classes": len(by_class)}

    avg_macro = sum(v["macro_recall"] for v in per_task_macro.values()) / len(per_task_macro)
    avg_acc = sum(v["accuracy"] for v in per_task_macro.values()) / len(per_task_macro)
    n_pass = sum(1 for v in per_task_macro.values() if v["macro_recall"] >= 0.70)

    print(f"===== VERB-ONLY rescoring of LoRA v3 [fulleval] ({len(per_task_macro)}/100 tasks) =====")
    for tc, v in sorted(per_task_macro.items()):
        status = "PASS>=70%" if v["macro_recall"] >= 0.70 else ""
        print(f"{tc:12s} acc={v['accuracy']:.3f}  macro={v['macro_recall']:.3f}  n_verbs={v['n_verb_classes']:2d}  {status}")

    print(f"\nVerb-only average accuracy: {avg_acc:.3f}")
    print(f"Verb-only average macro recall: {avg_macro:.3f}   tasks >=70%: {n_pass}/{len(per_task_macro)}")

    with open(f"{RESULTS_DIR}/lora_v3_{RESULT_TAG}_verbonly_summary.json", "w") as f:
        json.dump({
            "n_tasks": len(per_task_macro), "avg_accuracy": avg_acc, "avg_macro": avg_macro,
            "n_pass_70": n_pass, "per_task": per_task_macro,
        }, f, indent=2)
    print(f"\nSaved results/lora_v3_{RESULT_TAG}_verbonly_summary.json")


if __name__ == "__main__":
    main()
