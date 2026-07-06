"""Recovery eval entry — applies a recorded failure-frame restore state
right after env reset, then runs the standard policy step loop. Writes a
q_score metric JSON like the standard eval, plus the restore source so
downstream aggregators can attribute results to specific frames.

Usage (Hydra-style args):
    OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES \
    python eval_recovery.py task.name=putting_shoes_on_rack \
        log_path=/path/to/logs policy=websocket model.port=8765 \
        +restore_state=/path/to/episode_NNNNNNNN_stepNNNN.json

Self-contained: no FR (OG 5.1) imports. Uses challenge OG 4.5 APIs.
"""
from __future__ import annotations
import json
import logging
import os
import sys
from pathlib import Path

import hydra
import torch as th
from omegaconf import DictConfig

logger = logging.getLogger("eval_recovery")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def apply_failure_restore(evaluator, sem: dict, *, release_ag: bool = True) -> None:
    """Apply a recorded failure-frame state to the live sim (OG 4.5)."""
    import omnigibson as og
    robot = evaluator.robot

    # 1) release any leftover assisted-grasp
    if release_ag:
        for _arm in robot.arm_names:
            try:
                robot.release_grasp_immediately(arm=_arm)
            except Exception:
                pass

    # 2) joints
    jp = th.tensor(sem["joint_positions"], dtype=th.float32)
    robot.set_joint_positions(jp)

    # 3) base pose
    bpos = sem.get("base_position")
    bquat = sem.get("base_orientation")
    if bpos is not None and bquat is not None:
        robot.set_position_orientation(
            th.tensor(bpos, dtype=th.float32),
            th.tensor(bquat, dtype=th.float32),
        )

    # 4) object poses (scope-name -> {position, orientation})
    scope = evaluator.env.task.object_scope
    n_obj = 0
    for name, pose in (sem.get("object_poses") or {}).items():
        ent = scope.get(name)
        if ent is None:
            continue
        ent.set_position_orientation(
            th.tensor(pose["position"], dtype=th.float32),
            th.tensor(pose["orientation"], dtype=th.float32),
        )
        n_obj += 1

    # 5) pin controllers to current joints so they don't yank
    try:
        robot.apply_action(robot.q_to_action(jp))
    except Exception as e:
        logger.warning(f"controller-pin failed: {e}")

    # 6) one sim step so state takes effect
    og.sim.step()
    print(f"[restore] applied {jp.numel()} joints + base + {n_obj} objects", flush=True)

    # 7) re-engage AG if recorded — R1Pro uses private _establish_grasp
    # with ag_data=(object, link) and contact_pos.
    ag = sem.get("assisted_grasp") or {}
    for arm, obj_name in ag.items():
        if not obj_name:
            continue
        ent = scope.get(obj_name)
        if ent is None:
            print(f"[restore] WARN: AG obj '{obj_name}' not in scope; skipping", flush=True)
            continue
        try:
            # Pick a link on the object: prefer root_link; else first rigid link.
            ag_link = getattr(ent, "root_link", None)
            if ag_link is None and hasattr(ent, "links"):
                ag_link = next(iter(ent.links.values()))
            if ag_link is None:
                print(f"[restore] WARN: no link found on '{obj_name}'; skipping AG", flush=True)
                continue
            contact_pos = ent.get_position_orientation()[0]
            robot._establish_grasp(arm=arm, ag_data=(ent, ag_link), contact_pos=contact_pos)
            og.sim.step()  # stabilize the new constraint
            print(f"[restore] AG re-engaged: arm={arm} obj={obj_name}", flush=True)
        except Exception as e:
            print(f"[restore] WARN: AG re-engage failed for {arm}/{obj_name}: {type(e).__name__}: {e}", flush=True)


def _resolve_train_index(task_idx: int, literal_instance_id: int) -> int:
    """Map a literal instance_id to the eval_instance_ids index used by
    eval.py's eval_on_train_instances branch (order = first-seen in
    episodes.jsonl for this task)."""
    from omnigibson.macros import gm
    ep_path = os.path.join(gm.DATA_PATH,
                           "2025-challenge-task-instances",
                           "metadata", "episodes.jsonl")
    with open(ep_path) as f:
        episodes = [json.loads(l) for l in f]
    instances = []
    for ep in episodes:
        if ep["episode_index"] // 10000 == task_idx:
            instances.append(int((ep["episode_index"] // 10) % 1000))
    for i, v in enumerate(instances):
        if v == literal_instance_id:
            return i
    raise ValueError(
        f"instance_id={literal_instance_id} not in task {task_idx} train list "
        f"({len(instances)} instances)")


@hydra.main(version_base=None,
            config_path="/shared_work/BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs",
            config_name="base_config")
def main(cfg: DictConfig) -> None:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

    restore_path = getattr(cfg, "restore_state", None)
    if not restore_path:
        raise RuntimeError("must pass +restore_state=<path>")
    restore_path = Path(str(restore_path))
    if not restore_path.exists():
        raise FileNotFoundError(f"restore state not found: {restore_path}")
    sem = json.loads(restore_path.read_text())
    src = sem.get("source") or {}
    task_name = src.get("task_name")
    inst_id = src.get("instance_id")
    rollout_step = src.get("rollout_step")
    demo_id = src.get("demo_id")
    if task_name is None or inst_id is None:
        raise RuntimeError(f"restore JSON missing source.task_name or source.instance_id: {src}")

    if cfg.task.name != task_name:
        logger.warning(f"cfg.task.name='{cfg.task.name}' differs from JSON "
                       f"source.task_name='{task_name}' — using JSON value")
        cfg.task.name = task_name

    from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
    task_idx = TASK_NAMES_TO_INDICES[task_name]
    train_index = _resolve_train_index(task_idx, inst_id)
    logger.info(f"restore: task={task_name}(id={task_idx}) "
                f"literal_instance={inst_id} -> train_index={train_index} "
                f"demo_id={demo_id} rollout_step={rollout_step}")

    cfg.eval_on_train_instances = True
    cfg.eval_instance_ids = [train_index]

    from omnigibson.learning.eval import Evaluator
    from omnigibson.learning.utils.obs_utils import create_video_writer

    log_path = Path(cfg.log_path).expanduser()
    log_path.mkdir(parents=True, exist_ok=True)
    metrics_path = log_path / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)
    video_path = log_path / "videos"
    video_path.mkdir(parents=True, exist_ok=True)

    file_tag = f"{task_name}_{inst_id}_{rollout_step}"

    with Evaluator(cfg) as evaluator:
        logger.info("Evaluator booted; running recovery eval for one frame ...")
        evaluator.reset()
        evaluator.load_task_instance(train_index, test_hidden=False)
        evaluator.reset()
        # Apply the failure restore state to the live sim
        apply_failure_restore(evaluator, sem, release_ag=True)
        # CRITICAL: refresh evaluator.obs so the policy's first action is
        # computed from the POST-restore observation (not the stale cached
        # pre-restore one from evaluator.reset()).
        fresh_obs = evaluator.env.get_obs()[0]
        evaluator.obs = evaluator._preprocess_obs(fresh_obs)
        print("[restore] evaluator.obs refreshed post-restore", flush=True)

        if cfg.write_video:
            video_name = str(video_path / f"{file_tag}.mp4")
            evaluator.video_writer = create_video_writer(
                fpath=video_name, resolution=(448, 672))

        for metric in evaluator.metrics:
            metric.start_callback(evaluator.env)

        done = False
        terminated = False
        truncated = False
        while not done:
            terminated, truncated = evaluator.step()
            if terminated or truncated:
                done = True
            if cfg.write_video:
                evaluator._write_video()
            if evaluator.env._current_step % 1000 == 0:
                logger.info(f"Current step: {evaluator.env._current_step}")

        for metric in evaluator.metrics:
            metric.end_callback(evaluator.env)
        metrics = {}
        for metric in evaluator.metrics:
            metrics.update(metric.gather_results())
        metrics["_recovery_source"] = src
        # Load q_score at restore frame from the _goals.json companion file
        # (written by FR's failure-frame extractor alongside the state json)
        goals_path = restore_path.with_name(restore_path.stem + "_goals.json")
        q_start = None
        if goals_path.exists():
            try:
                _g = json.loads(goals_path.read_text())
                q_start = _g.get("q_absolute")
            except Exception:
                pass
        metrics["_q_start"] = q_start
        if isinstance(q_start, (int, float)) and metrics.get("q_score"):
            q_end = metrics["q_score"].get("final")
            if isinstance(q_end, (int, float)):
                metrics["_q_delta"] = q_end - q_start

        out_file = metrics_path / f"{file_tag}.json"
        with open(out_file, "w") as f:
            json.dump(metrics, f)
        logger.info(f"Wrote metrics: {out_file}")
        logger.info(f"terminated={terminated} truncated={truncated} "
                    f"n_trials={evaluator.n_trials} "
                    f"n_success={evaluator.n_success_trials}")


if __name__ == "__main__":
    main()
