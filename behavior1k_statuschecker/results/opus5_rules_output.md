I read all 36 frames. Here's what actually holds up across them.

## What varies across the set

The camera is the robot's head view; arms enter from the near edge of frame when raised. The radio has one highly diagnostic face — a flat panel with a **big dark round dial** — and a **small green light** that turns on in that dial once it's been pressed.

| | arm in frame | radio distance | round dial facing camera | green light |
|---|---|---|---|---|
| move to (9/9) | never | far/small or absent | never | no |
| pick up from | 4/9 no arm (early), 5/9 arm | close, on table | never | no |
| press (9/9) | yes | very close | 8/9 yes | on in late frames |
| place on (9/9) | yes | very close, back on table | ep30 yes, ep10/20 no | yes wherever dial visible |

## The four rules

**move to** — No robot arm anywhere in the picture. You see a lot of the room at once — floor, walls, ceiling, the whole couch — and the red radio is small and far away, or not in the picture at all.

**pick up from** — A robot arm has come into the picture and a gripper is reaching toward or closing around the red radio while the radio is still sitting on the table. You only see the plain red-and-white body of the radio; the big round dial is not turned toward you.

**press** — The radio's flat face with the big round dial is turned toward the camera and a gripper is right on that dial, sometimes with the radio lifted up in the air. This is the only step where a gripper is touching the round dial itself.

**place on** — The radio is back down resting on the table, or being lowered the last bit onto it, and the grippers are beside it or pulling away instead of touching the dial. If the round dial is visible it now has a small green light on.

Useful tie-break to state in the prompt: *if no arm is visible at all, answer "move to."*

## On your three known problems

**1. Early "pick up from" vs "move to" — genuinely not recoverable.** Episode 10 sec 13 and episode 20 sec 16 are labeled "pick up from" but show ceiling, a wide room view, a small distant radio, and no arm. They are pixel-for-pixel the same kind of picture as the move-to frames. Radio size/proximity trends the right way on average (move-to frames are consistently farther), but it fails on exactly these frames, so it isn't a rule — it's a prior. Robot orientation doesn't help either: move-to frames already point at the table. Take the loss and default no-arm frames to "move to"; that's right on 9/9 move-to and wrong on ~4/9 pick-up.

**2. Late "place on" — regime (b) does not appear in this sample.** All 9 place-on frames have arms clearly in view; none is a pulled-back arm-free wide shot. So I can't confirm the failure mode from these examples. If it does occur elsewhere, the only surviving cue is the green light on the dial, which at wide-shot distance is a few pixels — not something a 0.8B–2B model will read. Treat wide arm-free late frames as genuinely ambiguous with "move to."

**3. "press" vs "pick up from" — this one is solidly learnable.** They're separated by the radio's pose, not by whether an arm is present: in every press frame the flat dial face is turned toward the camera with a gripper on it, and in every pick-up frame the dial is hidden and you see only the striped body. Note the framing "arm holding the object up in the air" is what misled the manual pass — in episode 30 the press happens with the radio standing on the table, never lifted. Anchor on **dial facing you + gripper on the dial**, not on lift.

**One problem you didn't list:** "pick up from" vs "place on" collide when the dial isn't visible. Episode 10 sec 34 (pick up) and sec 57 (place on) are both "arm near a radio lying on the table, dial hidden" and I don't think they're separable from one frame. Episode 30 escapes this only because its place-on frames happen to show the lit dial.

## CONFIDENCE

- **press vs everything else** — learnable from single frames. Clean, redundant visual signature (dial facing camera, gripper on dial, green light in the later ones).
- **move to vs mid/late pick up from** — learnable. Arm present or absent is a hard, reliable split.
- **move to vs early pick up from** — not learnable. Needs temporal context; the distinction is "has the robot finished arriving," which is a fact about the trajectory, not the frame.
- **pick up from vs place on** — half learnable. Recoverable when the green light is visible (after-press state), unrecoverable when the dial is hidden, which is most of episodes 10 and 20.

Realistic ceiling for a single-frame 0.8–2B model here is roughly 70–80%, with essentially all remaining error on the move-to/early-pick-up boundary and the dial-hidden pick-up/place-on collision. If you can afford it, the cheapest real fix is not a better prompt — it's feeding the model the second count or the previous frame's label, since the four steps are strictly ordered and monotonic.