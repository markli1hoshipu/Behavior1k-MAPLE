"""LoRA fine-tune Qwen3.5-0.8B on the larger dataset (6 episodes/task, ~3300 examples
after per-class capping), with a SIMPLER training objective: classification-only loss.
No Opus-written CoT reasoning this time - the prompt just asks for the label directly,
and the target is just the label text, so loss is computed purely on the classification
decision. This also means no Opus annotation step is needed at all: labels come straight
from BEHAVIOR-1K's own ground truth (multitask_data_v2/*/ground_truth.json).

One episode per task is held out (excluded from training) for a genuinely unseen
generalization test after training.
"""
import json
import random
import re

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

ROOT = "/shared_work/markhsp/behavior1k-statuschecker"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
ADAPTER_OUT = f"{ROOT}/lora_adapter_v2"

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
    lines = []
    for line in md_text.strip().split("\n"):
        m = re.match(r"^\s*(\d+)\.\s*([^-]+?)\s*-\s*(.+)$", line)
        if m:
            n, label, rule = m.groups()
            lines.append(f"{n}. {label.strip()} - {rule.strip()}")
    return "\n".join(lines)


def build_user_prompt(task_instruction, action_list_text):
    return (
        f'Task: "{task_instruction}"\n'
        f"The robot's subskill right now is always exactly one of:\n{action_list_text}\n"
        "Look only at what is visible in this frame. Respond with ONLY the label, "
        "nothing else."
    )


def load_examples(held_out_episode):
    """held_out_episode: dict[task_chunk] -> episode id to EXCLUDE from training."""
    examples = []
    for task_chunk, instruction in TASKS.items():
        with open(f"{ROOT}/multitask_data_v2/{task_chunk}/ground_truth.json") as f:
            gt = json.load(f)
        with open(f"{ROOT}/results/multitask_rules_best_{task_chunk}.md") as f:
            action_list_text = parse_rules(f.read())
        user_prompt = build_user_prompt(instruction, action_list_text)

        held_out_ep = held_out_episode[task_chunk]
        n_train = 0
        for item in gt["per_second"]:
            if item["episode"] == held_out_ep:
                continue
            examples.append({
                "task_chunk": task_chunk, "sec": item["sec"], "episode": item["episode"],
                "frame_path": item["frame_path"], "user_prompt": user_prompt,
                "target": item["skill_description"],
            })
            n_train += 1
        print(f"{task_chunk}: {n_train} training examples (held out episode {held_out_ep})")
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
    labels[:, :prompt_len] = -100  # loss only on the label tokens

    full_inputs["labels"] = labels
    return {k: v.to(model.device) for k, v in full_inputs.items()}


def main():
    with open(f"{ROOT}/results/episode_plan.json") as f:
        episode_plan = json.load(f)
    # hold out the LAST episode in each task's plan (a fresh one, never used in the
    # first LoRA run's training OR as the original single-episode benchmark)
    held_out_episode = {tc: eps[-1] for tc, eps in episode_plan.items()}
    with open(f"{ROOT}/results/lora_v2_heldout_episodes.json", "w") as f:
        json.dump(held_out_episode, f, indent=2)

    print("Loading examples...")
    examples = load_examples(held_out_episode)
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
    model.enable_input_require_grads()
    model.train()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    EPOCHS = 2  # larger dataset (~5-6x more examples) - fewer passes needed
    GRAD_ACCUM = 8
    step = 0
    for epoch in range(EPOCHS):
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
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step} avg_loss={total_loss/n_steps:.4f}")
        print(f"=== epoch {epoch} done, avg_loss={total_loss/n_steps:.4f} ===")

    model.save_pretrained(ADAPTER_OUT)
    print(f"Saved LoRA adapter to {ADAPTER_OUT}")


if __name__ == "__main__":
    main()
