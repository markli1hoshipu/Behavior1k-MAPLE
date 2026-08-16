"""Round 2: show Opus its own v1 rules PLUS the actual Qwen3.5-2B errors from running
them live (results/multitask_v2_task-*.json), so it can revise against real failures
instead of fresh frames - the technique that worked well for the radio task.
"""
import json
import subprocess
import sys

ROOT = "/shared_work/markhsp/behavior1k-statuschecker"

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


def build_prompt(task_chunk, instruction):
    with open(f"{ROOT}/results/multitask_rules_{task_chunk}.md") as f:
        prev_rules = f.read()
    with open(f"{ROOT}/results/multitask_v2_{task_chunk}.json") as f:
        run = json.load(f)
    with open(f"{ROOT}/multitask_data/{task_chunk}/ground_truth.json") as f:
        gt = json.load(f)
    frame_by_sec = {it["sec"]: it["frame_path"] for it in gt["per_second"]}

    preds = run["predictions"]
    errors = [p for p in preds if not p["correct"]]
    # cap error frames shown to keep the prompt a manageable size on complex tasks
    error_sample = errors[::max(1, len(errors) // 25)][:25]
    error_lines = "\n".join(
        f"- sec {p['sec']}: ground truth = {p['gt']}, model predicted = {p['pred']}  ({frame_by_sec[p['sec']]})"
        for p in error_sample
    )

    return f"""You previously derived rules (below) for classifying a robot's subskill from a
single camera frame, for the household task: "{instruction}"

=== YOUR PREVIOUS RULES ===
{prev_rules}

Those rules were dropped into a real prompt and run against Qwen3.5-2B (an actual small
vision-language model). Overall: {run['accuracy']:.1%} accuracy / {run['macro_recall']:.1%}
macro recall. Here is a sample of what it actually got WRONG (out of {len(errors)} total errors):

{error_lines}

Read the error frames (use your Read tool) and revise your rules to fix what's fixable.
If a specific confusion is genuinely not recoverable from a single static frame (e.g. it
depends on motion direction, or on a state that changed off-screen), say so explicitly in
your rule rather than inventing wording that won't hold up - a rule that admits the limit
is more useful than one that guesses.

Output ONLY the final revised rules in this exact format (one line per label, numbered to
match the label list from before), nothing else - no preamble, no analysis, no confidence
notes. Keep total output under 200 words:

1. <label> - <rule>
2. <label> - <rule>
...
"""


def main():
    task_chunk = sys.argv[1]
    instruction = TASKS[task_chunk]
    prompt = build_prompt(task_chunk, instruction)

    with open(f"{ROOT}/results/multitask_rules_prompt2_{task_chunk}.txt", "w") as f:
        f.write(prompt)

    result = subprocess.run(
        ["claude", "-p", prompt, "--model", "claude-opus-5",
         "--allowedTools", "Read", "--output-format", "text"],
        capture_output=True, text=True, timeout=600,
    )
    output = result.stdout.strip()

    with open(f"{ROOT}/results/multitask_rules_v2_{task_chunk}.md", "w") as f:
        f.write(output)
    print(f"=== {task_chunk} ===")
    print(output)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])


if __name__ == "__main__":
    main()
