"""Sample a diverse set of labeled frames across multiple episodes and hand them
to Claude Opus 5 (via the `claude` CLI) to derive generalized visual rules for
distinguishing the 4 subskill classes - instead of the ad hoc "eyeball 3-4 frames"
approach used to hand-write ACTION_LIST in annotate.py.
"""
import json
import subprocess

ROOT = "/shared_work/markhsp/behavior1k-statuschecker"
EPISODES = ["00000010", "00000020", "00000030"]
SAMPLES_PER_SEGMENT = 3  # early / mid / late within each labeled segment


def sample_frames_for_episode(episode):
    with open(f"{ROOT}/data/episode_{episode}/ground_truth.json") as f:
        gt = json.load(f)
    per_second = {it["sec"]: it for it in gt["per_second"]}
    samples = []
    for seg in gt["segments"]:
        start, end = seg["start_s"], seg["end_s"]
        secs = sorted({
            int(start + 0.15 * (end - start)),
            int(start + 0.5 * (end - start)),
            int(start + 0.85 * (end - start)),
        })
        for sec in secs:
            if sec in per_second:
                samples.append({
                    "episode": episode,
                    "sec": sec,
                    "label": seg["skill_description"],
                    "frame_path": per_second[sec]["frame_path"],
                })
    return samples


def main():
    all_samples = []
    for ep in EPISODES:
        all_samples.extend(sample_frames_for_episode(ep))
    print(f"Sampled {len(all_samples)} labeled frames across {len(EPISODES)} episodes")

    lines = []
    for s in all_samples:
        lines.append(f"- [{s['label']}] episode {s['episode']} sec {s['sec']}: {s['frame_path']}")
    frame_list = "\n".join(lines)

    prompt = f"""You are helping design a prompt for a small (0.8B-2B parameter) vision-language
model that will classify a robot's current subskill from a single camera frame, once per
second, for the task "Turn on the radio receiver that's on the table in the living room."

There are exactly 4 possible subskill labels, always occurring in this fixed order within
an episode: move to -> pick up from -> press -> place on.

Below is a list of frames, each with its GROUND-TRUTH label and file path. Read every
image (use your Read tool) and study what is visually true across all examples of each
label, and what's confusable between adjacent labels.

{frame_list}

Known problems from our own manual analysis so far, which you should specifically address:
1. Early "pick up from" frames often show no visible robot arm yet (still navigating) -
   visually identical to "move to". Look for any subtler distinguishing signal (e.g. robot
   orientation, distance to object, anything else) that might separate these, or conclude
   honestly if no such signal exists in a single static frame.
2. "place on" frames span two very different visual regimes: (a) close-up, arm visible,
   lowering the object, and (b) after release, camera pulled back to a wide shot with no
   arm visible - which looks just like "move to". Figure out if there's a reliable way to
   tell late "place on" apart from "move to" in a single frame, or if it's genuinely
   ambiguous.
3. "press" and "pick up from" both often show an arm holding the red object up in the air.

For each of the 4 labels, write a SHORT (1-3 sentence) rule describing what is reliably
visible in-frame for that label, using **plain, common-sense language** (avoid overly
precise spatial jargon like "on top of / below / left / right" - that specific wording
tends to confuse small models). Where a distinction is genuinely not recoverable from a
single static frame (no motion/temporal info), say so explicitly rather than inventing a
rule that doesn't hold up.

End with a short "CONFIDENCE" note on which of the 4 boundaries you think are learnable
from single frames at all, and which ones fundamentally need temporal/motion context.
"""

    with open(f"{ROOT}/results/opus5_rules_prompt.txt", "w") as f:
        f.write(prompt)

    print("Calling claude -p --model claude-opus-5 ...")
    result = subprocess.run(
        ["claude", "-p", prompt, "--model", "claude-opus-5",
         "--allowedTools", "Read", "--permission-mode", "bypassPermissions",
         "--output-format", "text"],
        capture_output=True, text=True, timeout=600,
    )
    output = result.stdout.strip()
    print(output)

    with open(f"{ROOT}/results/opus5_rules_output.md", "w") as f:
        f.write(output)
    print(f"\nSaved to {ROOT}/results/opus5_rules_output.md")
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])


if __name__ == "__main__":
    main()
