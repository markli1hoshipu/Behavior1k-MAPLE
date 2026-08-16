"""Zero-shot test of lora_adapter_v2 on task-0029 ("clean up your desk") - a task NOT
in the 10-task training set (task-0000/0001/0005/0020/0022/0031/0034/0040/0042/0049).
No rules/prompt-engineering for this task: the label list is just the raw distinct
skill_description strings found in ground truth, unadorned - true zero-shot."""
import json

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

ROOT = "/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
ADAPTER_PATH = f"{ROOT}/lora_adapter_v2"
RESULTS_DIR = f"{ROOT}/results"
TASK_CHUNK = "task-0029"
TASK_INSTRUCTION = "Clean up your desk: put things away in their proper places."

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


def main():
    with open(f"{ROOT}/multitask_data_v2/{TASK_CHUNK}/ground_truth.json") as f:
        gt = json.load(f)
    labels = gt["distinct_labels"]
    prompt = build_user_prompt(TASK_INSTRUCTION, labels)
    print("PROMPT:\n" + prompt + "\n")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    predictions = []
    correct = 0
    for item in gt["per_second"]:
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
        pred = parse_answer(raw, labels)

        gt_label = item["skill_description"]
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

    print(f"\n=== task-0029 (clean up your desk) ZERO-SHOT (lora_adapter_v2): "
          f"acc={acc:.3f} ({correct}/{total})  macro={macro:.3f} ===")
    for k, (c, t) in sorted(by_class.items()):
        print(f"    {k:20s} {c}/{t}  ({c/t:.2f})")

    with open(f"{RESULTS_DIR}/task0029_zeroshot.json", "w") as f:
        json.dump({
            "task_chunk": TASK_CHUNK, "task_name": gt["task_name"], "zero_shot": True,
            "accuracy": acc, "macro_recall": macro, "correct": correct, "total": total,
            "prompt": prompt, "predictions": predictions,
        }, f, indent=2)
    print(f"\nSaved results/task0029_zeroshot.json")


if __name__ == "__main__":
    main()
