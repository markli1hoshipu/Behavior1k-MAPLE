I read every error frame plus the surrounding segment. Three of my previous rules were factually wrong about this episode, and one new cue holds up cleanly.

## What I got wrong

**1. The green light does not exist during "place on."** I measured saturated-green pixels in all 59 frames: exactly **0** in every one of the 11 place-on frames (48–58). The only frames with any green at all are sec 46 (43 px) and sec 47 (71 px) — both labeled *press*. The dial faces away for the entire place-on segment. This wasn't "too subtle for a 2B model"; the pixels aren't there. That rule alone poisoned the whole class.

**2. "Grippers beside it or pulling away" is wrong for 8 of the 11 place-on frames.** At sec 48–55 the gripper is firmly wrapped around the radio, lowering it. Qwen reading that as "pick up from" was reading the image correctly — my rule mislabeled it. Only sec 56–58 show retracted arms.

**3. I had the label semantics backwards.** In this episode "pick up from" is mostly the *walk-up* (sec 9–24 have no robot parts in frame at all), and "press" *starts at sec 30 while the radio is still sitting on the mat* — the reach, grasp and lift are all labeled press. So my old "pick up from = gripper closing on the radio on the table" rule points squarely at frames labeled press.

**4. A distractor I never flagged:** the bright glowing white blob on the mat is a separate lit object present in *every* frame from sec 0 onward, including all move-to frames. It is the brightest thing in the picture and a small model can easily read it as "the radio is on."

## Confirmed unrecoverable

**sec 9–20 vs move to: confirmed, refuted nothing.** I put sec 0/4/8 (move to) next to 13/17 (pick up from) — same ceiling, same hanging lamp, same whole-room view, same tiny distant radio, no arms. The only measurable difference is couch area 12.5% vs 13.0% of frame. That's a prior, not a cue. **12 frames of guaranteed loss.**

**Grasp vs release is the same picture.** sec 28–37 (grasp) and sec 48–55 (lower/release) differ only in a ~25° pose change. There is no "is it closing or opening" signal.

## The one real new cue

The radio's **standing angle**. Principal-axis tilt from vertical:

| frames | phase | tilt |
|---|---|---|
| 26–39 | untouched + grasp + lift | 0.7°–12.7° (stands straight up) |
| 40–46 | dial to camera | 76°–89° (laid over facing you) |
| 48–58 | place on | 21°–34° (leaning) |

Zero overlap. The robot puts the radio back down **tipped over at a slant**, not neatly upright as it started. It's real and clean here — but it's a 20–25° difference on a small object, so I'd state it and not bank on it.

## Rewritten ACTION_LIST

**move to** — You can see the ceiling, or the big hanging ceiling lamp, or the tops of the walls with pictures on them. The whole room is in view at once, the red radio is a small distant object, and none of the robot's own arms are anywhere in the picture.

**pick up from** — None of the robot's arms are in the picture yet, or only the very tips of the grippers have just come into view. The ceiling and the hanging lamp are gone; the couch and bare wall fill the far part of the picture, the mat is large and close, and the red radio is still sitting there untouched with nothing reaching it.

**press** — Either the radio has been lifted clear of the table and is held in a gripper, or its flat face with the big dark round dial is turned toward the camera, filling a large part of the picture, with a gripper on that dial. Whenever the radio is still down near the table during this step, it is standing straight up — tall and narrow.

**place on** — The radio is back down at the table with a gripper still around it or drawing away from it. The giveaway is that the radio is no longer standing up straight the way it was at the start — it is leaning over at a slant. Also answer this when the radio is resting on the mat by itself with both arms drawn back.

**Tie-breaks, in order:**
1. No robot arms in the picture and the ceiling or hanging lamp is visible → **move to**.
2. No robot arms in the picture and the ceiling and lamp are gone → **pick up from**.
3. Radio held up in the air, or big red face with the dark round dial → **press**.
4. Radio down at the table with an arm right at it and you cannot tell whether it is being grabbed or let go → answer **place on**.
5. Ignore the bright white glowing blob on the mat. It is a separate lit object present from the very first frame; it is not the radio and it does not mean the radio is on.

Rule 4 is the load-bearing change. The justification is asymmetry, not evidence: **press has an unmistakable signature of its own (dial to camera, radio in the air), so it doesn't need the ambiguous state; place on has almost nothing else.** Ceding that state to pick-up-from is what produced 0/11.

## Expected outcome

| | accuracy | macro recall | place on |
|---|---|---|---|
| Qwen now | 45.8% | 53.6% | 0/11 |
| new rules, tilt cue **not** read | 62.7% | 68.8% | 11/11 |
| new rules, tilt cue read | 76.3% | 83.3% | 11/11 |

Downside is bounded because rule 4 catches the fallout when tilt fails.

**True single-frame ceiling: 47/59 ≈ 80%**, capped by sec 9–20. My earlier 70–80% estimate was right by accident — I attributed it to a pick-up/place-on collision that turns out to be separable, and missed that the real cost is a 12-second stretch of pick-up-from that is visually pure move-to. No prompt recovers those. Feeding the second count or the previous label is still the only fix, and now I can say exactly what it's worth: about 20 points.