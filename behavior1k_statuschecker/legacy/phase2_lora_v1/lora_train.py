"""LoRA fine-tune Qwen3.5-0.8B on the Opus-generated CoT annotations across all 10
tasks. One shared adapter; task identity is carried by the prompt text (task
instruction + that task's ACTION_LIST), same as at inference time, so the model learns
to condition on the prompt rather than needing per-task adapters.
"""
import glob
import json
import re

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

ROOT = "/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
ADAPTER_OUT = f"{ROOT}/lora_adapter"

TASKS = {
    "task-0000": "Turn on the radio receiver that's on the table in the living room.",
    "task-0001": "Pick up the trash (soda cans) from the floor and put them in the trash can.",
    "task-0005": "Set the mousetraps near the sink and the toilet.",
    "task-0020": "Sort the vegetables from the baskets into the mixing bowls.",
    "task-0022": "Put the shoes on the shoe rack.",
    "task-0031": "Wash the two dusty boxing gloves in the washer until they are no longer covered with dust.",
    "task-0034": "Hang the poster on the wall nail.",
    "task-0040": "Make microwave popcorn: put the popcorn bag in the microwave and turn it on.",
    "task-0042": "Take the onion out of the sink, dice it on the chopping board, and put it in the bowl.",
    "task-0049": "Make a pizza: chop vegetables, add toppings to the dough, and bake it in the oven.",
}

SYSTEM_PROMPT = (
    "You are annotating individual frames from a fixed head-mounted camera on a "
    "mobile manipulator robot completing a household task. Classify the robot's "
    "current subskill using ONLY the visual evidence in this single frame."
)


def parse_rules(md_text):
    numbered = {}
    lines = []
    for line in md_text.strip().split("\n"):
        m = re.match(r"^\s*(\d+)\.\s*([^-]+?)\s*-\s*(.+)$", line)
        if m:
            n, label, rule = m.groups()
            numbered[n] = label.strip()
            lines.append(f"{n}. {label.strip()} - {rule.strip()}")
    return numbered, "\n".join(lines)


def build_user_prompt(task_instruction, action_list_text):
    return (
        f'Task: "{task_instruction}"\n'
        f"The robot's subskill right now is always exactly one of:\n{action_list_text}\n"
        "Look only at what is visible in this frame. Respond in EXACTLY this format:\n"
        "Reasoning: <one short sentence on what you actually see that supports your answer>\n"
        "Answer: <label>"
    )


def load_examples():
    examples = []
    for task_chunk, instruction in TASKS.items():
        with open(f"{ROOT}/multitask_data/{task_chunk}/ground_truth.json") as f:
            gt = json.load(f)
        frame_by_sec = {it["sec"]: it["frame_path"] for it in gt["per_second"]}
        with open(f"{ROOT}/results/multitask_rules_best_{task_chunk}.md") as f:
            _, action_list_text = parse_rules(f.read())

        cot_path = f"{ROOT}/results/cot_annotations_{task_chunk}.txt"
        with open(cot_path) as f:
            cot_lines = f.read().strip().split("\n")

        user_prompt = build_user_prompt(instruction, action_list_text)
        n_matched = 0
        for line in cot_lines:
            m = re.match(r"sec_(\d+):\s*Reasoning:\s*(.+?)\s*\|\s*Answer:\s*(.+)\s*$", line.strip())
            if not m:
                continue
            sec, reasoning, answer = m.groups()
            sec = int(sec)
            if sec not in frame_by_sec:
                continue
            target = f"Reasoning: {reasoning.strip()}\nAnswer: {answer.strip()}"
            examples.append({
                "task_chunk": task_chunk, "sec": sec,
                "frame_path": frame_by_sec[sec],
                "user_prompt": user_prompt, "target": target,
            })
            n_matched += 1
        print(f"{task_chunk}: {n_matched}/{len(cot_lines)} CoT lines matched to frames")
    return examples


def build_training_tensors(processor, model, example):
    img = Image.open(example["frame_path"]).convert("RGB")
    messages_prompt_only = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": example["user_prompt"]},
        ]},
    ]
    prompt_inputs = processor.apply_chat_template(
        messages_prompt_only, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    prompt_len = prompt_inputs["input_ids"].shape[-1]

    messages_full = messages_prompt_only + [
        {"role": "assistant", "content": [{"type": "text", "text": example["target"]}]}
    ]
    full_inputs = processor.apply_chat_template(
        messages_full, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt"
    )

    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100  # only compute loss on the assistant completion

    full_inputs["labels"] = labels
    return {k: v.to(model.device) for k, v in full_inputs.items()}


def main():
    print("Loading examples...")
    examples = load_examples()
    print(f"Total training examples: {len(examples)}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0"
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # needed alongside grad checkpointing when base weights are frozen (LoRA)
    model.train()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    EPOCHS = 3
    GRAD_ACCUM = 8
    step = 0
    for epoch in range(EPOCHS):
        import random
        random.shuffle(examples)
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()
        for i, ex in enumerate(examples):
            inputs = build_training_tensors(processor, model, ex)
            out = model(**inputs)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            total_loss += out.loss.item()
            n_steps += 1
            if (i + 1) % GRAD_ACCUM == 0 or i == len(examples) - 1:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step % 5 == 0:
                    print(f"epoch {epoch} step {step} avg_loss={total_loss/n_steps:.4f}")
        print(f"=== epoch {epoch} done, avg_loss={total_loss/n_steps:.4f} ===")

    model.save_pretrained(ADAPTER_OUT)
    print(f"Saved LoRA adapter to {ADAPTER_OUT}")


if __name__ == "__main__":
    main()
