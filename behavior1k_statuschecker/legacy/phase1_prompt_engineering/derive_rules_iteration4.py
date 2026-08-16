"""Fourth Opus consultation. This time give Opus the model's own stated reasoning traces
(not just pixel measurements or final digits) from a run where Qwen3.5-2B was asked to
explain itself before answering, using v14's exact rules. The traces make the failure
mode legible in the model's own words: it reasons "gripper is positioned over the round
dial, indicating press" for several place-on frames where the ground truth says otherwise,
and for a few early pick-up-from frames where the dial has only just come into view.
"""
import json
import subprocess

ROOT = "/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker"
EPISODE = "00000010"

V14_ACTION_LIST = """1. move to - no robot arm is visible anywhere in the frame; you see a wide view of the room (floor, walls, couch) with the red radio small, far away, or not visible at all.
2. pick up from - a robot arm/gripper IS visible, reaching toward or holding the red radio, but the radio's round dial is NOT facing the camera (you only see the plain red-and-white striped body, not the dial face).
3. press - the radio's flat face with its round dial IS facing the camera, and a gripper is directly on that dial. This is the only step where the gripper touches the round dial. The radio does not need to be lifted for this - it can happen while the radio still sits on the table.
4. place on - the radio is resting back down on the table (or being lowered onto it) and the gripper is pulling away rather than touching the dial. If the dial is visible, it may show a small green light now on (this means the radio was already successfully pressed).
Tie-break: if no arm is visible at all anywhere in the frame, answer 1 (move to)."""

with open(f"{ROOT}/data/episode_{EPISODE}/ground_truth.json") as f:
    gt = json.load(f)
frame_by_sec = {it["sec"]: it["frame_path"] for it in gt["per_second"]}

d = json.load(open(f"{ROOT}/results/2b_v14_reasoning.json"))
preds = d["predictions"]

lines = []
for p in preds:
    mark = "CORRECT" if p["correct"] else "WRONG"
    lines.append(
        f"- sec {p['sec']}: gt={p['gt']}, model answered={p['pred']} ({mark})\n"
        f"    model's stated reasoning: \"{p['reasoning']}\"\n"
        f"    frame: {frame_by_sec[p['sec']]}"
    )
trace_block = "\n".join(lines)

prompt = f"""Same 4-class robot-subskill task (move to -> pick up from -> press -> place on,
turning on a radio), same episode, same model (Qwen3.5-2B), using your best rules so far
(v14, unchanged - this is NOT a new rule set, just a new response format):

=== ACTION_LIST (v14, unchanged) ===
{V14_ACTION_LIST}

This time the model was asked to explain its reasoning BEFORE answering, so you can read
its actual stated logic instead of inferring intent from pixels alone. Result: 42.4%
accuracy, 55.8% macro recall (move to 9/9, pick up from 7/30, press 9/9, place on 0/11).
Press went to a perfect 9/9 this way - explaining itself first seems to help there. But
place on is now systematically absorbed into "press", and 4 pick-up-from frames near the
grasp point also flipped to "press".

Full per-second trace (reasoning + answer) below. Read it - you don't need to re-read the
raw images this time, the model's own words ARE the data:

{trace_block}

Two things to diagnose from the traces themselves:

1. For every place-on frame (sec 48-58), what does the model's reasoning actually say
   about dial visibility and gripper position? Is it applying rule 3 correctly given what
   it claims to see, or is it misreading the frame? If it's applying the rule correctly to
   what it (correctly or incorrectly) perceives, that's a RULE gap, not a perception gap -
   rule 3 currently has no negative case, nothing that says "gripper near/on the dial is
   NOT enough on its own, it also needs X" to keep place-on from being swallowed.

2. For sec 28/30/31/32 (pick-up-from, misclassified as press), what does the model see
   that differs from sec 33-38 (pick-up-from, correctly classified)? Is there a stated
   distinguishing detail in its own reasoning between these two groups, or does it use
   near-identical language for both (which would mean the frames themselves are genuinely
   on the rule 2/3 boundary)?

Rewrite the ACTION_LIST (same 4 classes, same plain-language style as v14) to close
whatever gap the traces reveal, particularly a way to keep rule 3 (press) from also
matching place-on frames. If the traces show this is a genuine boundary with no reliable
textual fix, say that explicitly instead of proposing a rule that reads well but won't
change behavior - you have three previous rounds of evidence that "plausible-sounding"
rules with no measurable cue behind them didn't move the model at all.
"""

with open(f"{ROOT}/results/opus5_rules_prompt_v4.txt", "w") as f:
    f.write(prompt)

print("Calling claude -p --model claude-opus-5 (iteration 4, reasoning-trace based) ...")
result = subprocess.run(
    ["claude", "-p", prompt, "--model", "claude-opus-5",
     "--allowedTools", "Read", "--permission-mode", "bypassPermissions",
     "--output-format", "text"],
    capture_output=True, text=True, timeout=1500,
)
output = result.stdout.strip()
print(output)

with open(f"{ROOT}/results/opus5_rules_output_v4.md", "w") as f:
    f.write(output)
print(f"\nSaved to {ROOT}/results/opus5_rules_output_v4.md")
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
