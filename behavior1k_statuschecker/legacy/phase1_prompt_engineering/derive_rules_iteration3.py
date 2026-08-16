"""Third Opus consultation. Iteration 2 (v15/v16/v17) had pixel-verified MORE ACCURATE
facts than v14, but scored WORSE on the actual Qwen3.5-2B model every time (v14: 45.8%/
53.6% macro; v15: 18.6%/25.0%; v16 and v17: 23.7%/38.9%, byte-identical to each other
despite wording fixes). Give Opus the actual A/B: v14's exact working prompt text next to
v16/v17's regressed prompt text, plus both result sets, and ask it to diagnose the actual
delta rather than have me keep guessing at wording blind.
"""
import json
import subprocess

ROOT = "/shared_work/markhsp/behavior1k-statuschecker"
EPISODE = "00000010"

V14_ACTION_LIST = """1. move to - no robot arm is visible anywhere in the frame; you see a wide view of the room (floor, walls, couch) with the red radio small, far away, or not visible at all.
2. pick up from - a robot arm/gripper IS visible, reaching toward or holding the red radio, but the radio's round dial is NOT facing the camera (you only see the plain red-and-white striped body, not the dial face).
3. press - the radio's flat face with its round dial IS facing the camera, and a gripper is directly on that dial. This is the only step where the gripper touches the round dial. The radio does not need to be lifted for this - it can happen while the radio still sits on the table.
4. place on - the radio is resting back down on the table (or being lowered onto it) and the gripper is pulling away rather than touching the dial. If the dial is visible, it may show a small green light now on (this means the radio was already successfully pressed).
Tie-break: if no arm is visible at all anywhere in the frame, answer 1 (move to)."""

V17_ACTION_LIST = """1. move to - no robot arm is visible anywhere in the frame; you see a wide view of the room with the red radio small, far away, or not visible at all.
2. pick up from - an arm/gripper is visible and reaching toward or holding the plain body of the radio, standing upright, its round dial not facing the camera.
3. press - either the radio is lifted up in a gripper, or its round dial IS facing the camera with a gripper on that dial.
4. place on - an arm is at the radio near the table, and the radio itself is tilted or leaning to one side rather than standing straight up and vertical.
Tie-break: if no arm is visible, answer 1. If an arm is at the radio on the table and you can't tell whether it's grabbing or releasing, answer 4."""

with open(f"{ROOT}/data/episode_{EPISODE}/ground_truth.json") as f:
    gt = json.load(f)
frame_by_sec = {it["sec"]: it["frame_path"] for it in gt["per_second"]}

d14_2b = json.load(open(f"{ROOT}/results/2b_v14.json"))
d17 = json.load(open(f"{ROOT}/results/2b_v17.json"))

flip_lines = []
for p14, p17 in zip(d14_2b["predictions"], d17["predictions"]):
    if p14["pred"] != p17["pred"]:
        flip_lines.append(
            f"- sec {p14['sec']} (gt={p14['gt']}): v14 said {p14['pred']!r} "
            f"({'correct' if p14['correct'] else 'WRONG'}) -> v17 said {p17['pred']!r} "
            f"({'correct' if p17['correct'] else 'WRONG'})  frame: {frame_by_sec[p14['sec']]}"
        )
flip_block = "\n".join(flip_lines)

prompt = f"""Same 4-class robot-subskill classification task as before, same model (Qwen3.5-2B),
same episode. Two ACTION_LIST prompt versions, A/B tested on the real model:

=== PROMPT V14 (WORKS: 45.8% accuracy, 53.6% macro recall) ===
{V14_ACTION_LIST}

=== PROMPT V17 (WORSE: 23.7% accuracy, 38.9% macro recall) ===
{V17_ACTION_LIST}

V17 was your own iteration-2 revision after you found v14's green-light claim was pixel-
wrong (0 green pixels in any place-on frame) and its "gripper pulling away" framing
mismatched 8/11 place-on frames. Those corrections were factually right. But V17 scores
WORSE on the real model, not better - and it's not a length problem (I already tried a
394-word version that also failed, then trimmed to about the same length as V14 and
still got the SAME predictions, byte-for-byte, as the 394-word version). I also tried
fixing two specific things I suspected (a self-contradiction in rule 2, and rule 4 asking
the model to compare against "how it looked earlier" which a single-frame classifier
can't do) - that produced ZERO change in output, identical predictions to before the fix.
So the actual cause is neither prompt length nor those two wording issues.

Here is every frame where V14 and V17 disagree on their prediction for the SAME frame:
{flip_block}

Read these frames and both prompt texts closely. I need you to explain the actual
mechanism, not just say "V17 is more complex." Specifically:
1. Is there a specific rule or phrase in V17 that a small model can't ground (e.g. "the
   radio is tilted/leaning" - can a 2B VLM actually perceive a ~20 degree tilt on a small
   object in a 720x720 frame, or is that below its effective visual resolution even
   though it's true in the pixels)?
2. Does V17's rule 3 change ("radio lifted up in a gripper" as an alternative sufficient
   condition for press) create new confusion with pick-up-from that V14 avoided by
   requiring the dial specifically?
3. Any other structural difference between V14 and V17 you can point to concretely.

Then produce ONE more revision: keep V14's proven structure/phrasing as the base, and
change ONLY what you can now show is both factually necessary AND perceivable by a small
model (i.e. don't reintroduce the tilt cue if you conclude it's below the model's
effective resolution - drop that class distinction as unrecoverable instead, explicitly,
rather than gesturing at an invisible cue).
"""

with open(f"{ROOT}/results/opus5_rules_prompt_v3.txt", "w") as f:
    f.write(prompt)

print("Calling claude -p --model claude-opus-5 (iteration 3) ...")
result = subprocess.run(
    ["claude", "-p", prompt, "--model", "claude-opus-5",
     "--allowedTools", "Read", "--permission-mode", "bypassPermissions",
     "--output-format", "text"],
    capture_output=True, text=True, timeout=1500,
)
output = result.stdout.strip()
print(output)

with open(f"{ROOT}/results/opus5_rules_output_v3.md", "w") as f:
    f.write(output)
print(f"\nSaved to {ROOT}/results/opus5_rules_output_v3.md")
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
