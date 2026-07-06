"""Data-collection driver.

Wraps HybridPolicy and OmniGibson's `eval_data_gen_par_save_all.py` to produce
full-detail episodes: parquet actions, hdf5 raw trajectories, annotations, BDDL
transitions, phase segments, and videos.

Entry point: `maple collect --task <name> --instances <ids> [...]`.
"""
