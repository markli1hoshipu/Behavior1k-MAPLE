"""For a given task, sample a few labeled frames per skill class and ask Opus 5 to
derive task-specific visual rules for a single-frame VLM classifier - the same
technique that took the radio task from ~25% to ~54% macro recall, applied generically.
"""
import json
import subprocess
import sys

ROOT = "/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker"

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


def sample_frames(task_chunk, per_class=4):
    with open(f"{ROOT}/multitask_data/{task_chunk}/ground_truth.json") as f:
        gt = json.load(f)
    by_label = {}
    for item in gt["per_second"]:
        by_label.setdefault(item["skill_description"], []).append(item)
    samples = []
    for label, items in by_label.items():
        n = len(items)
        idxs = sorted(set(int(i * n / per_class) for i in range(min(per_class, n))))
        for i in idxs:
            samples.append(items[i])
    return gt, samples


def build_prompt(task_chunk, instruction):
    gt, samples = sample_frames(task_chunk)
    lines = "\n".join(f"- [{s['skill_description']}] sec {s['sec']}: {s['frame_path']}" for s in samples)
    labels = gt["distinct_labels"]
    return f"""You are helping design a prompt for a small (2B parameter) vision-language model
that will classify a robot's current subskill from a single camera frame, once per
second, for the household task: "{instruction}"

The possible subskill labels for this task are: {labels}

Below is a list of frames, each with its GROUND-TRUTH label and file path. Read every
image (use your Read tool) and figure out what is visually TRUE and RELIABLY DETECTABLE
in each label's frames, and what's easily confused between adjacent labels.

{lines}

For each label, write a SHORT (1-2 sentence) rule describing what is reliably visible
in-frame for that label, using plain, common-sense language (avoid precise spatial
jargon like "on top of / below / left / right" - it tends to confuse small models).
Where a distinction is genuinely not recoverable from a single static frame (e.g. it
depends on motion direction or on state before/after an off-screen event), say so
explicitly rather than inventing a rule that won't hold up.

Output ONLY the final rules in this exact format (one line per label, numbered to match
the label list above), nothing else - no preamble, no analysis section, no confidence
notes. Keep total output under 200 words:

1. <label> - <rule>
2. <label> - <rule>
...
"""


def main():
    task_chunk = sys.argv[1]
    instruction = TASKS[task_chunk]
    prompt = build_prompt(task_chunk, instruction)

    with open(f"{ROOT}/results/multitask_rules_prompt_{task_chunk}.txt", "w") as f:
        f.write(prompt)

    result = subprocess.run(
        ["claude", "-p", prompt, "--model", "claude-opus-5",
         "--allowedTools", "Read", "--output-format", "text"],
        capture_output=True, text=True, timeout=600,
    )
    output = result.stdout.strip()

    with open(f"{ROOT}/results/multitask_rules_{task_chunk}.md", "w") as f:
        f.write(output)
    print(f"=== {task_chunk} ===")
    print(output)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])


if __name__ == "__main__":
    main()
