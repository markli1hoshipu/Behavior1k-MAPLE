## The mechanism

V17 didn't confuse classes — it **deleted two of them**. Across 59 frames it emitted only `move to` (54) and `press` (5). Zero `pick up from`, zero `place on`. That's not a boundary shifting, that's two rules never activating.

The root cause is structural, and it's in **rule 1**, not in rules 2–4.

Rule 1 is the only rule in either prompt scored on whole-frame gestalt: *"a wide view of the room with the red radio small, far away."* In this episode the couch fills the top ~40% and floor/rug the rest of **every single frame**, including every pick-up and press frame. So rule 1's positive half is satisfiable at 100% of frames. Its only gate is a negation — "no robot arm is visible." A 2B VLM can't reliably compute `NOT ∃ arm`; what it does well is match positive descriptions to pixels. That makes rule 1 a permanent attractor, and rules 2–4 have to out-score it on positive evidence.

V14 wins that contest three ways V17 gave up:

| | V14 | V17 |
|---|---|---|
| rule 2 existence check | `IS visible` — capitalized, first position, own clause | buried mid-chain, unemphasized |
| rule 2 logic | `but` (concessive → default-with-exception) | `and` ×4 (conjunction → weakest-link scoring) |
| rule 2 restatement | `(you only see the plain red-and-white striped body)` — a positive paraphrase giving a second way to satisfy | deleted; "red" dropped entirely, the only color anchor |
| tie-break scope | "no arm visible **at all anywhere in the frame**" | "no arm is visible" |

That last row is the single most damaging edit. I measured it: **61–99% of the arm's dark pixels sit in frame-border-touching components** — the arm is cropped at the bottom edge, never a centered salient object. Asked "is an arm visible?" without a scope qualifier, a small VLM answers from frame center, where there's couch, rug, and a small red thing. "Anywhere in the frame" is the phrase that forces the peripheral scan. V14 states it twice, including in the final, highest-recency sentence. V17 states it once and drops it from the tie-break.

V17's added class-4 tie-break ("if you can't tell grabbing or releasing, answer 4") fired **zero times in 11 chances** — it's gated behind the same arm perception that already failed upstream.

## Your three questions

**1. The tilt cue.** Real in pixels, larger than you guessed — ~2–13° from vertical in pick-up frames vs **21–36°** in place-on. But unusable, for a reason other than angle size: the red mask **fragments into 3–5 connected components with the largest holding only 51–63% of the red pixels**, because the black gripper cuts across the body. On a ~85×115 px object that's ~12 merged visual tokens, largest coherent fragment ~6. Recovering a principal axis requires mentally reassembling occluded fragments. And the model settled it empirically: as V17's sole class-4 discriminator it produced 0 activations in 11 frames. **Dropped as unrecoverable, explicitly.**

Related: I confirmed your green-light finding and can add that it's doubly dead. 0 green px in all 11 place-on frames; the only radio-side green is **43–80 px at secs 46–47 — during press**. 80 px is under one merged token.

**2. Rule 3's "lifted in a gripper."** Yes, it collides with rule 2 — but it's not the main mechanism (press only went 7/9→5/9). The real loss is what it *replaced*. V14's dial-only definition maps onto a huge coarse-scale signature: red blob aspect ratio **0.19–0.48 in pick-up vs 1.30–1.39 in press**, ~3× the width and ~4× the pixels. V17 swapped a near-perfect low-frequency separator for a lift-state judgement requiring the occluded table plane. Note V17 answered `move to` at secs 47–51 where the radio *is* held aloft — the lift disjunct never fired either.

**3. Other structural difference.** The clause-count asymmetry above, plus rule 1 losing its "(floor, walls, couch)" anchor — which made rule 1 *more* permissive, not less.

## The revision

`annotate.py` now holds **v19** = V14 verbatim plus two changes, each pixel-backed:

- **The arm check moved out of the rules into a preamble gate.** Dark pixels in the bottom quarter: **0.00–1.71% in every arm-absent frame (secs 0–24), 8.0–25.9% in every arm-present frame (secs 27–58)** — perfect separation, >4× gap, coarse-scale.
- **Rule 4 rewritten** to drop the false green claim and the "pulling away" framing, stated truthfully and deliberately non-competing so it can't steal rule 2's frames.

I ran all of it rather than hand you untested text: **v14 45.8% · v19 44.1% · v20 42.4% · v18 35.6% · v17 23.7%.**

**v19 does not beat V14 — it's one frame short (26 vs 27).** Two failed intermediates are worth reporting because both confirmed the mechanism by breaking it: v18 (35.6%) put the arm cue *inside rule 2*, where it's also true at secs 40–46, so rule 2 gained a clause rule 3 didn't have and press collapsed 7→1. v20 (42.4%) removed the rule-3 aspect parenthetical on my guess that it wasn't earning its place; press dropped 6→5, so it was load-bearing and I put it back. v19 ties V14 on move-to (9/9), pick-up (11/30) and place-on (0/11), losing only sec 42 — and secs 41/42/43 are pixel-identical (aspect 1.33/1.31/1.30), so that's a greedy-decode boundary flip, not a perceptual failure. What v19 buys is a temporally coherent segmentation (1|2|3|2|1 in one pass) instead of V14's flip-flopping tail.

**The more useful finding: prompt engineering is done here.** Of 33 remaining errors, 19 are pick-up frames (secs 9–27) where **no arm is in view at all** — I confirmed this visually at sec 25 and numerically (0.00–1.71% dark) — and no wording makes a single-frame classifier report an arm that isn't there. The other 11 are place-on frames that are near time-reverses of pick-up frames. Single-frame ceiling is ~29/59 (≈49%); V14 at 27 is already within two frames of it. The remaining headroom is in `WINDOW > 1`, not in words.