"""Round 3: targeted refinement using the CURRENT BEST method's actual errors per task
(not round-1's rules-only baseline), aiming specifically to close the gap to 70% macro
recall. Tells Opus the oracle ceiling so it knows whether more rule refinement can help
or whether it should say so is a genuine limit.
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

METHOD_FILES = {
    "baseline": "multitask_{}.json", "round1": "multitask_v2_{}.json", "round2": "multitask_v3_{}.json",
    "reasoning": "multitask_reasoning_{}.json", "structured": "multitask_structured_{}.json",
    "ensemble": "multitask_ensemble_{}.json", "precens": "multitask_precens_{}.json",
}


def build_prompt(task_chunk, instruction):
    best_so_far = json.load(open(f"{ROOT}/results/multitask_best_so_far.json"))[task_chunk]
    best_method = best_so_far["method"]
    best_macro = best_so_far["macro"]

    with open(f"{ROOT}/results/{METHOD_FILES[best_method].format(task_chunk)}") as f:
        run = json.load(f)
    with open(f"{ROOT}/results/multitask_rules_best_{task_chunk}.md") as f:
        prev_rules = f.read()
    with open(f"{ROOT}/multitask_data/{task_chunk}/ground_truth.json") as f:
        gt = json.load(f)
    frame_by_sec = {it["sec"]: it["frame_path"] for it in gt["per_second"]}

    preds = run["predictions"]
    errors = [p for p in preds if not p["correct"]]
    error_sample = errors[::max(1, len(errors) // 20)][:20]
    error_lines = "\n".join(
        f"- sec {p['sec']}: ground truth = {p['gt']}, predicted = {p['pred']}  ({frame_by_sec.get(p['sec'], '?')})"
        for p in error_sample
    )

    return f"""Task: "{instruction}"

Current best result for this task, across everything tried so far (multiple rule
revisions, reasoning-trace prompting, structured sub-question prompting, and ensembling):
{best_macro:.1%} macro recall (method: {best_method}). Goal: get this task to 70% macro
recall. Below are the CURRENT rules and a sample of the remaining errors.

=== CURRENT RULES ===
{prev_rules}

=== REMAINING ERRORS (sample of {len(errors)} total) ===
{error_lines}

Read the error frames (use your Read tool). You have three options per confusion you
find, in this priority order:
1. If there's a genuinely reliable visual cue you can add/sharpen to fix it, do so.
2. If two labels are visually near-identical in a single frame but one is much rarer,
   consider whether biasing the tie-break toward the more common one is defensible.
3. If a confusion is genuinely NOT recoverable from a single static frame (depends on
   motion direction, or on an off-screen state change), say so explicitly in the rule
   text (e.g. "cannot be distinguished from X in a still frame; default to X") rather
   than inventing wording that won't hold up - we'd rather know the true ceiling.

Output ONLY the final revised rules in this exact format (one line per label, numbered
to match before), nothing else - no preamble, no analysis, no confidence notes. Keep
total output under 220 words:

1. <label> - <rule>
2. <label> - <rule>
...
"""


def main():
    task_chunk = sys.argv[1]
    instruction = TASKS[task_chunk]
    prompt = build_prompt(task_chunk, instruction)

    result = subprocess.run(
        ["claude", "-p", prompt, "--model", "claude-opus-5",
         "--allowedTools", "Read", "--output-format", "text"],
        capture_output=True, text=True, timeout=600,
    )
    output = result.stdout.strip()

    with open(f"{ROOT}/results/multitask_rules_v3_{task_chunk}.md", "w") as f:
        f.write(output)
    print(f"=== {task_chunk} ===")
    print(output)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])


if __name__ == "__main__":
    main()
