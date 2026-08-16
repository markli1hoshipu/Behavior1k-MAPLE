# TODO

1. **Try multi-camera.** Only the head camera
   (`observation.rgb.zed_link_camera_0`) is used today. The raw dataset also
   ships left/right wrist realsense views - feeding more than one view to the
   VLM per frame could resolve the occlusion/object-identity ambiguity that's
   likely driving a chunk of the object-recall gap (see verb-only vs.
   verb+object numbers in the results tables).

2. **Try CoT data.** `legacy/phase2_lora_v1` used Opus-written CoT reasoning
   targets once (549 examples) before the project moved to classification-only
   loss for scale. Worth reintroducing reasoning targets at the current
   100-task scale, especially for the highest-class-count tasks
   (`clean_up_your_desk`, `canning_food`, `make_pizza`) where the model has to
   disambiguate among 30-46 verb+object combos from a single frame.

3. **Try adding logical chain constraints.** Skill sequences aren't
   unconstrained - e.g. "move to X" almost always precedes "pick up from X" for
   the same object. Encoding that structure (as a training signal, a decoding-
   time constraint, or a post-hoc consistency filter over a full episode's
   predictions) could clean up locally-plausible-but-sequence-inconsistent
   predictions that per-frame classification alone can't catch.
