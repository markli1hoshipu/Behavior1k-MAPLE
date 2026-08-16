"""Round 4, surgical: only for the 6 tasks with oracle ceiling >=70% that haven't
crossed it yet. Explicitly instructs Opus to make a MINIMAL patch to the single weakest
class, not rewrite the whole rule set - round 3's whole-rewrite approach regressed 8/10
tasks (some severely), matching the same overfitting pattern seen on the radio task.
"""
import json
import subprocess
import sys

ROOT = "/shared_work/markhsp/behavior1k-statuschecker"

TASKS = {
    "task-0000": "Turn on the radio receiver that's on the table in the living room.",
    "task-0001": "Pick up the trash (soda cans) from the floor and put them in the trash can.",
    "task-0020": "Sort the vegetables from the baskets into the mixing bowls.",
    "task-0031": "Wash the two dusty boxing gloves in the washer until they are no longer covered with dust.",
    "task-0034": "Hang the poster on the wall nail.",
    "task-0040": "Make microwave popcorn: put the popcorn bag in the microwave and turn it on.",
}

METHOD_FILES = {
    "baseline": "multitask_{}.json", "round1": "multitask_v2_{}.json", "round2": "multitask_v3_{}.json",
    "reasoning": "multitask_reasoning_{}.json", "structured": "multitask_structured_{}.json",
    "ensemble": "multitask_ensemble_{}.json", "precens": "multitask_precens_{}.json",
    "round3": "multitask_v4_{}.json",
}

# The rules text that actually PRODUCED the current best score for each task (not always
# multitask_rules_best_*.md, since ensemble/precens methods reuse whichever underlying
# rules scored well in earlier rounds - map each best-method name to its rules file).
METHOD_TO_RULES = {
    "baseline": None,  # generic label list, no custom rules file
    "round1": "multitask_rules_{}.md",
    "round2": "multitask_rules_v2_{}.md",
    "round3": "multitask_rules_v3_{}.md",
    "reasoning": "multitask_rules_best_{}.md",
    "structured": "multitask_rules_best_{}.md",
    "ensemble": "multitask_rules_best_{}.md",
    "precens": "multitask_rules_best_{}.md",
}


def build_prompt(task_chunk, instruction):
    best_so_far = json.load(open(f"{ROOT}/results/multitask_best_so_far.json"))[task_chunk]
    best_method = best_so_far["method"]
    best_macro = best_so_far["macro"]

    rules_file = METHOD_TO_RULES.get(best_method) or "multitask_rules_best_{}.md"
    with open(f"{ROOT}/results/{rules_file.format(task_chunk)}") as f:
        prev_rules = f.read()
    with open(f"{ROOT}/results/{METHOD_FILES[best_method].format(task_chunk)}") as f:
        run = json.load(f)
    with open(f"{ROOT}/multitask_data/{task_chunk}/ground_truth.json") as f:
        gt = json.load(f)
    frame_by_sec = {it["sec"]: it["frame_path"] for it in gt["per_second"]}

    preds = run["predictions"]
    by_class_errors = {}
    by_class_total = {}
    for p in preds:
        by_class_total[p["gt"]] = by_class_total.get(p["gt"], 0) + 1
        if not p["correct"]:
            by_class_errors.setdefault(p["gt"], []).append(p)
    # find the single worst class by error count
    worst_class = max(by_class_errors, key=lambda k: len(by_class_errors[k]))
    worst_errors = by_class_errors[worst_class][:12]
    error_lines = "\n".join(
        f"- sec {p['sec']}: predicted \"{p['pred']}\" instead  ({frame_by_sec.get(p['sec'], '?')})"
        for p in worst_errors
    )
    worst_recall = 1 - len(by_class_errors[worst_class]) / by_class_total[worst_class]

    return f"""Task: "{instruction}"

Current best score for this task: {best_macro:.1%} macro recall. The single WORST class
is "{worst_class}" at only {worst_recall:.1%} recall ({len(by_class_errors[worst_class])} of
{by_class_total[worst_class]} frames wrong). Every OTHER class is doing reasonably well
with the current rules - do not touch them.

=== CURRENT RULES (do not rewrite these unless the "{worst_class}" rule itself is at fault) ===
{prev_rules}

=== ALL ERRORS FOR "{worst_class}" (what it got predicted as instead) ===
{error_lines}

Read these error frames (use your Read tool). Make the SMALLEST possible edit that fixes
"{worst_class}" specifically:
- If you can sharpen just the "{worst_class}" rule's wording, edit ONLY that line.
- If a competing class's rule is accidentally also matching these frames, add ONE
  narrowing clause to that competing rule, and nothing else.
- If this confusion is genuinely not recoverable from a single static frame, say so in
  the "{worst_class}" rule text (e.g. "cannot be distinguished from X in a still frame")
  rather than forcing a fix.

Do NOT rewrite rules for classes that were not mentioned as having errors above.

Output ONLY the complete revised rule set (same numbering, same labels, ALL classes
present even if you didn't change most of them), one line per label, nothing else - no
preamble, no analysis:

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
        capture_output=True, text=True, timeout=1200,
    )
    output = result.stdout.strip()

    with open(f"{ROOT}/results/multitask_rules_v4_{task_chunk}.md", "w") as f:
        f.write(output)
    print(f"=== {task_chunk} ===")
    print(output)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])


if __name__ == "__main__":
    main()
