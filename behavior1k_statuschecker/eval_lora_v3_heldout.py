"""Evaluate lora_adapter_v3 (trained on all 100 tasks, verb+object classification-only
loss) on the held-out episode reserved per task during training
(results/lora_v3_heldout_episodes.json)."""
import glob
import json

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

ROOT = "/shared_work/markhsp/behavior1k-statuschecker"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
ADAPTER_PATH = f"{ROOT}/lora_adapter_v3"
RESULTS_DIR = f"{ROOT}/results"
DATA_ROOT = f"{ROOT}/data_2026"

SYSTEM_PROMPT = (
    "You are annotating individual frames from a fixed head-mounted camera on a "
    "mobile manipulator robot completing a household task. Classify the robot's "
    "current subskill using ONLY the visual evidence in this single frame."
)


def build_user_prompt(task_instruction, labels):
    label_list = "\n".join(f"{i+1}. {lab}" for i, lab in enumerate(labels))
    return (
        f'Task: "{task_instruction}"\n'
        f"The robot's subskill right now is always exactly one of:\n{label_list}\n"
        "Look only at what is visible in this frame. Respond with ONLY the label, "
        "nothing else."
    )


def parse_answer(raw_text, labels):
    text = raw_text.strip().lower().strip('."\'')
    for lab in labels:
        if text == lab.lower():
            return lab
    for lab in sorted(labels, key=len, reverse=True):
        if lab.lower() in text:
            return lab
    return None


def run_task(model, processor, task_chunk, held_out_ep):
    with open(f"{DATA_ROOT}/{task_chunk}/ground_truth.json") as f:
        gt = json.load(f)
    held_out_items = [it for it in gt["per_second"] if it["episode"] == held_out_ep]
    if not held_out_items:
        print(f"SKIP {task_chunk}: no held-out frames for {held_out_ep}")
        return None
    labels = sorted(set(it["target"] for it in held_out_items))
    prompt = build_user_prompt(gt["task_instruction"], gt["distinct_labels"])

    predictions = []
    correct = 0
    for item in held_out_items:
        img = Image.open(item["frame_path"]).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]},
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        raw = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        pred = parse_answer(raw, gt["distinct_labels"])

        gt_label = item["target"]
        is_correct = pred == gt_label
        correct += int(is_correct)
        predictions.append({
            "sec": item["sec"], "episode": item["episode"], "gt": gt_label, "pred": pred,
            "raw": raw.strip(), "correct": is_correct,
        })

    total = len(predictions)
    acc = correct / total if total else 0.0
    by_class = {}
    for p in predictions:
        by_class.setdefault(p["gt"], [0, 0])
        by_class[p["gt"]][1] += 1
        if p["correct"]:
            by_class[p["gt"]][0] += 1
    macro = sum(c / t for c, t in by_class.values()) / len(by_class) if by_class else 0.0

    print(f"=== {task_chunk} ({gt['task_name']}) HELD-OUT (v3): acc={acc:.3f} ({correct}/{total})  macro={macro:.3f} ===")
    for k, (c, t) in sorted(by_class.items()):
        print(f"    {k:30s} {c}/{t}")

    with open(f"{RESULTS_DIR}/lora_v3_heldout_{task_chunk}.json", "w") as f:
        json.dump({
            "task_chunk": task_chunk, "task_name": gt["task_name"], "held_out_episode": held_out_ep,
            "accuracy": acc, "macro_recall": macro, "correct": correct, "total": total,
            "predictions": predictions,
        }, f, indent=2)
    return acc, macro


def main():
    with open(f"{RESULTS_DIR}/lora_v3_heldout_episodes.json") as f:
        held_out_episodes = json.load(f)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    summary = []
    for task_chunk in sorted(held_out_episodes.keys()):
        result = run_task(model, processor, task_chunk, held_out_episodes[task_chunk])
        if result:
            summary.append((task_chunk, *result))

    print("\n\n===== SUMMARY: LoRA v3 (100 tasks, verb+object) HELD-OUT =====")
    total_macro = 0
    n_pass = 0
    for tc, acc, macro in summary:
        status = "PASS>=70%" if macro >= 0.70 else ""
        if macro >= 0.70:
            n_pass += 1
        total_macro += macro
        print(f"{tc:12s} macro={macro:.3f}  {status}")
    avg = total_macro / len(summary) if summary else 0.0
    print(f"\nLoRA v3 held-out average macro: {avg:.3f}   tasks >=70%: {n_pass}/{len(summary)}")

    with open(f"{RESULTS_DIR}/lora_v3_summary.json", "w") as f:
        json.dump({
            "n_tasks": len(summary), "avg_macro": avg, "n_pass_70": n_pass,
            "per_task": {tc: {"accuracy": acc, "macro_recall": macro} for tc, acc, macro in summary},
        }, f, indent=2)


if __name__ == "__main__":
    main()
