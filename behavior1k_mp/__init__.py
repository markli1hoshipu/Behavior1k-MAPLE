"""behavior1k_mp — hybrid motion-planner + X-VLA policy pipeline for BEHAVIOR-1K."""
__version__ = "0.1.0"

# `third_party/` contains only `zexternal_utils.py` (Mark's IK function) +
# `r1pro.urdf`. We load `zexternal_utils.py` by file path inside
# `behavior1k_mp.ik`, so we don't need to manipulate sys.path here.
