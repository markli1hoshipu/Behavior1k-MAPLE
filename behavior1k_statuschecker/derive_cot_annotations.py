"""Use Opus 5 to write grounded chain-of-thought rationales for every already-labeled
training frame, to use as LoRA fine-tuning targets for Qwen3.5-0.8B. Opus already KNOWS
the correct answer (from BEHAVIOR-1K's ground truth) - its job is not to classify, but to
write a short, visually-grounded justification for that known answer, in the same
"Reasoning: ...\nAnswer: <label>" format the fine-tuned model will need to reproduce.
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
    with open(f"{ROOT}/multitask_data/{task_chunk}/ground_truth.json") as f:
        gt = json.load(f)
    labels = gt["distinct_labels"]
    items = gt["per_second"]

    lines = "\n".join(f"- sec {it['sec']} [KNOWN CORRECT LABEL: {it['skill_description']}]: {it['frame_path']}" for it in items)

    return f"""You are generating training data for fine-tuning a small vision-language model
to classify a robot's subskill from a single camera frame, for the household task:
"{instruction}"

Possible labels for this task: {labels}

Below is every frame from one episode, each tagged with its ALREADY-KNOWN CORRECT label
(from ground-truth motion-capture annotation - you are NOT being asked to guess or verify
it, it is certain). Your job is to read each frame (use your Read tool) and write a
SHORT, honest, visually-grounded justification for why that label fits THIS SPECIFIC
frame - point to what is actually visible (arm position, what's being touched, what's in
view), not generic task knowledge. If the frame genuinely doesn't show anything that
distinguishes this label from a neighboring one (e.g. it's an approach frame with no
robot parts visible), say so honestly in the reasoning rather than inventing a
visual detail that isn't there - the fine-tuned model needs to learn realistic reasoning,
not confident hallucination.

{lines}

Output EXACTLY one block per frame, in this format, nothing else (no preamble, no
summary):

sec_<NNNN>: Reasoning: <one short sentence, grounded in what's visible> | Answer: <label>
sec_<NNNN>: Reasoning: <one short sentence, grounded in what's visible> | Answer: <label>
...

(one line per frame, in the same order as listed above, using the exact 4-digit
zero-padded sec_NNNN identifier and the exact label text given)
"""


def main():
    task_chunk = sys.argv[1]
    instruction = TASKS[task_chunk]
    prompt = build_prompt(task_chunk, instruction)

    with open(f"{ROOT}/results/cot_prompt_{task_chunk}.txt", "w") as f:
        f.write(prompt)

    result = subprocess.run(
        ["claude", "-p", prompt, "--model", "claude-opus-5",
         "--allowedTools", "Read", "--output-format", "text"],
        capture_output=True, text=True, timeout=1500,
    )
    output = result.stdout.strip()

    with open(f"{ROOT}/results/cot_annotations_{task_chunk}.txt", "w") as f:
        f.write(output)
    print(f"=== {task_chunk} ===")
    print(f"{len(output.splitlines())} lines generated")
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])


if __name__ == "__main__":
    main()
