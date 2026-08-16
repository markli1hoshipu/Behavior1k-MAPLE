"""Close the loop: show Opus 5 its own previous rules PLUS the actual Qwen3.5-2B
validation results (results/2b_v14.json) that used them, so it can revise the rules
against real model failures instead of fresh hand-picked frames.
"""
import json
import subprocess

ROOT = "/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker"
EPISODE = "00000010"

with open(f"{ROOT}/results/opus5_rules_output.md") as f:
    previous_rules = f.read()

with open(f"{ROOT}/results/2b_v14.json") as f:
    run = json.load(f)

with open(f"{ROOT}/data/episode_{EPISODE}/ground_truth.json") as f:
    gt = json.load(f)
frame_by_sec = {it["sec"]: it["frame_path"] for it in gt["per_second"]}

preds = run["predictions"]
errors = [p for p in preds if not p["correct"]]
correct = [p for p in preds if p["correct"]]

error_lines = "\n".join(
    f"- sec {p['sec']}: ground truth = {p['gt']}, Qwen3.5-2B predicted = {p['pred']}  ({frame_by_sec[p['sec']]})"
    for p in errors
)

# a few correctly-classified frames per class, for contrast
contrast_lines = []
by_class_correct = {}
for p in correct:
    by_class_correct.setdefault(p["gt"], []).append(p)
for cls, items in by_class_correct.items():
    for p in items[:2]:
        contrast_lines.append(f"- sec {p['sec']}: ground truth = {cls} (Qwen got this RIGHT)  ({frame_by_sec[p['sec']]})")
contrast_block = "\n".join(contrast_lines)

prompt = f"""You previously derived rules (below) for classifying a robot's subskill from a
single camera frame, for the 4-class task "move to -> pick up from -> press -> place on"
(turning on a radio). Those rules were then dropped into a real prompt and run against
Qwen3.5-2B (an actual small vision-language model, not a human/strong model) over all 59
seconds of episode {EPISODE}. Overall it scored 45.8% raw accuracy / 53.6% macro recall -
better than before, but far from your ~70-80% single-frame ceiling estimate, and it
completely whiffed one class.

=== YOUR PREVIOUS RULES ===
{previous_rules}

=== WHAT ACTUALLY WENT WRONG (Qwen3.5-2B's real errors on episode {EPISODE}) ===
{error_lines}

=== FOR CONTRAST: frames Qwen3.5-2B got RIGHT ===
{contrast_block}

Read the ERROR frames (use your Read tool) and compare them against the CONTRAST frames
of the same/adjacent classes. Two specific patterns to investigate:

1. "place on" scored 0/11 - EVERY place-on frame was misclassified (mostly as "pick up
   from", some as "move to"). Your rule for place on/pick up from hinged on the dial
   being hidden + gripper direction, and on a green light appearing after press. Look at
   the actual place-on frames: is the dial visible or hidden in each? Is the green light
   actually visible/detectable, or is it too small/subtle for a small VLM to reliably
   read even when technically present in the pixels? Is there a completely different,
   more obvious visual signal for "place on" that we're missing (radio's exact position
   relative to the mat center, gripper pose, anything)?

2. sec 9-27 (most of "pick up from") is STILL being called "move to" even with the new
   rules - this matches your original prediction that this boundary is unrecoverable
   from a single frame. Confirm or refute that with these specific frames.

Rewrite the ACTION_LIST rules (same 4 classes, same short/plain-language style, no
"on top of / below / left / right" spatial jargon) to fix what's fixable. If something
genuinely isn't recoverable from a single static frame even with a perfect rule, say so
explicitly rather than inventing a rule that won't hold up on the next model - we'd
rather know the true ceiling than chase noise.
"""

with open(f"{ROOT}/results/opus5_rules_prompt_v2.txt", "w") as f:
    f.write(prompt)

print("Calling claude -p --model claude-opus-5 (iteration 2) ...")
result = subprocess.run(
    ["claude", "-p", prompt, "--model", "claude-opus-5",
     "--allowedTools", "Read", "--permission-mode", "bypassPermissions",
     "--output-format", "text"],
    capture_output=True, text=True, timeout=600,
)
output = result.stdout.strip()
print(output)

with open(f"{ROOT}/results/opus5_rules_output_v2.md", "w") as f:
    f.write(output)
print(f"\nSaved to {ROOT}/results/opus5_rules_output_v2.md")
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
