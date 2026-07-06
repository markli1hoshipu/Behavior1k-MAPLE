"""Phase 1b — fine-grained closed-loop grasp of the radio.

Per-waypoint **closed-loop** control:
  1. Read current proprioceptive state.
  2. Solve IK from current state to wp_k (warm-started by current joints).
  3. Linearly interpolate in joint space from current → IK target over
     INTERP_STEPS_PER_WAYPOINT frames; emit those frames one at a time.
  4. After the chunk finishes, read state again and check:
        err = max_dim(|state_q − ik_q|)
     If err < JOINT_ERROR_THRESHOLD_RAD: move to wp_{k+1}.
     Otherwise: retry this waypoint (plan again from new state).
     If MAX_RETRIES_PER_WAYPOINT is exhausted, give up and move on.
  5. After all waypoints converge, emit CLOSE_GRIPPER_FRAMES holding the
     last commanded pose with R_grip = −1.

This replaces the previous "plan-everything-in-on_enter" pattern, which
suffered from cumulative controller lag — each waypoint started its
interpolation from the **previous IK target** (assumed reached) rather
than the **actually-achieved joint state**, so error compounded across
waypoints. In closed-loop, the interp always starts from the actual
state, so lag does not accumulate.

State machine in `forward()`:
  - mode=BRIDGE        emit pre-computed bridge frames (approach → grasp)
  - mode=PLAN          (no frame to emit) → solve IK, build sub-traj,
                       transition to EXEC
  - mode=EXEC          emit one frame from sub-traj per call
                       → when sub-traj exhausted, transition to CHECK
  - mode=CHECK         compute joint error vs IK target
                       → if converged or retries exhausted: advance wp_idx
                       → else: retry (back to PLAN for same wp)
  - mode=CLOSE_GRIP    emit close-gripper hold frames
  - mode=DONE          is_done() returns True
"""
from __future__ import annotations

import logging

import numpy as np
import torch as th
from scipy.spatial.transform import Rotation as R

from behavior1k_mp.core.action import Action, Executor
from behavior1k_mp.core.grasp import vertical_descent_waypoints
from behavior1k_mp.core.ik import DEFAULT_URDF_PATH, solve_ik_r1pro
from behavior1k_mp.core.utils.action_bridge import linear_bridge_to
from behavior1k_mp.core.utils.obs import extract_state_23d
from behavior1k_mp.core.utils.pose import apply_roll_180_deg, pose_in_frame

logger = logging.getLogger(__name__)


def _np1d(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


class PickUpGraspAction(Action):
    name = "pick_up_radio_grasp"
    expected_phase_id = 1
    executor = Executor.MOTION_PLANNER

    INTERP_STEPS_PER_WAYPOINT: int = 30
    CLOSE_GRIPPER_FRAMES: int = 15
    GRASP_HEIGHT_OFFSET_M: float = 0.16
    IK_MAX_ITERS: int = 400
    GRIPPER_YAW_OFFSET_RAD: float = -np.pi / 2

    # Closed-loop convergence params.
    MAX_RETRIES_PER_WAYPOINT: int = 2
    JOINT_ERROR_THRESHOLD_RAD: float = 0.05   # max per-joint |Δ| to consider converged

    # Variant-dependent grasp offset in the RADIO's local frame (m).
    # All-zero baseline (user request). A vs B now differ only via
    # `apply_roll_180_deg` (flip the gripper 180° around its approach axis).
    GRASP_OFFSET_A_LOCAL = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    GRASP_OFFSET_B_LOCAL = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def __init__(self, env, robot, target_obj, *, active_arm: str = "right", **kwargs):
        super().__init__(env, robot, target_obj, **kwargs)
        if active_arm != "right":
            raise NotImplementedError("only right-arm grasp implemented")
        self.active_arm = active_arm
        # closed-loop state (set in on_enter / mutated in forward)
        self._waypoints: list[np.ndarray] | None = None
        self._held_state23: np.ndarray | None = None
        self._wp_idx: int = 0
        self._retry_count: int = 0
        self._sub_traj: np.ndarray | None = None
        self._sub_traj_idx: int = 0
        self._last_ik_q: np.ndarray | None = None
        self._bridge_traj: np.ndarray | None = None
        self._bridge_idx: int = 0
        self._closing: bool = False
        self._close_frames_emitted: int = 0
        self._done_internal: bool = False
        self._last_emitted_action: np.ndarray | None = None

    # ───────────────────────── helpers ─────────────────────────
    def _radio_in_base(self) -> np.ndarray:
        radio_pos, radio_quat = self.target_obj.get_position_orientation()
        robot_pos, robot_quat = self.robot.get_position_orientation()
        radio_world = np.concatenate([_np1d(radio_pos)[:3], _np1d(radio_quat)[:4]])
        robot_world = np.concatenate([_np1d(robot_pos)[:3], _np1d(robot_quat)[:4]])
        return pose_in_frame(radio_world, robot_world)

    def _top_down_grasp_pose(self, radio_in_base: np.ndarray, variant: str = "A") -> np.ndarray:
        from behavior1k_mp.core.utils.pose import quat_xyzw_to_yaw
        radio_yaw = quat_xyzw_to_yaw(radio_in_base[3:7])
        grasp_yaw = radio_yaw + self.GRIPPER_YAW_OFFSET_RAD
        grasp_quat_xyzw = R.from_euler("xyz", [0.0, 0.0, grasp_yaw]).as_quat()

        # Variant-dependent xy offset in radio's local frame → base frame
        # via the radio's yaw rotation.
        local_offset = (self.GRASP_OFFSET_A_LOCAL if variant == "A"
                        else self.GRASP_OFFSET_B_LOCAL)
        c, s = np.cos(radio_yaw), np.sin(radio_yaw)
        R_yaw_only = np.array([[c, -s, 0.0],
                               [s,  c, 0.0],
                               [0.0, 0.0, 1.0]], dtype=np.float64)
        base_offset = R_yaw_only @ local_offset

        grasp_pos = (radio_in_base[:3]
                     + base_offset
                     + np.array([0.0, 0.0, self.GRASP_HEIGHT_OFFSET_M]))
        return np.concatenate([grasp_pos, grasp_quat_xyzw]).astype(np.float64)

    def _assemble_action(self, q_torso_arm_11: np.ndarray, state23: np.ndarray, grip_value: float) -> np.ndarray:
        action = np.zeros(23, dtype=np.float32)
        action[3:7] = q_torso_arm_11[0:4]
        action[7:14] = state23[7:14]
        action[14] = state23[14]
        action[15:22] = q_torso_arm_11[4:11]
        action[22] = float(grip_value)
        return action

    def _current_q11(self, obs) -> np.ndarray:
        s23 = extract_state_23d(obs["robot_r1::proprio"])
        return np.concatenate([s23[3:7], s23[15:22]]).astype(np.float64)

    def _plan_sub_traj_for_current_wp(self, current_q11: np.ndarray) -> None:
        """Solve IK to self._waypoints[self._wp_idx] and build a 30-frame interp.

        Stores the IK solution in self._last_ik_q and the interp buffer in
        self._sub_traj. Resets self._sub_traj_idx to 0.
        """
        wp = self._waypoints[self._wp_idx]
        res = solve_ik_r1pro(
            mode="right_torso",
            target_pos_m=wp[:3],
            target_quat_xyzw=wp[3:7],
            urdf_path=str(DEFAULT_URDF_PATH),
            initial_guess=current_q11,
            max_iters=self.IK_MAX_ITERS,
            verbose=False,
        )
        self._last_ik_q = res["q_sol"].astype(np.float64)
        logger.info(
            "PickUpGrasp IK wp %d/%d (try %d): err_pos=%.4f m, err_quat=%.4f rad",
            self._wp_idx + 1, len(self._waypoints), self._retry_count,
            res["err_pos"], res["err_quat"],
        )
        sub = []
        for s in range(1, self.INTERP_STEPS_PER_WAYPOINT + 1):
            alpha = s / float(self.INTERP_STEPS_PER_WAYPOINT)
            q_interp = (1.0 - alpha) * current_q11 + alpha * self._last_ik_q
            sub.append(self._assemble_action(q_interp, self._held_state23, +1.0))
        self._sub_traj = np.stack(sub, axis=0).astype(np.float32)
        self._sub_traj_idx = 0

    # ───────────────────────── lifecycle ─────────────────────────
    def on_enter(self, obs) -> None:
        # 1. radio in base + grasp pose
        radio_in_base = self._radio_in_base()
        variant = (self.shared_state or {}).get("grasp_variant", "A")
        grasp_pose = self._top_down_grasp_pose(radio_in_base, variant=variant)
        if variant == "B":
            grasp_pose = apply_roll_180_deg(grasp_pose)

        logger.info(
            "PickUpGrasp: variant=%s, radio=(%.3f, %.3f, %.3f), grasp=(%.3f, %.3f, %.3f) quat=(%.3f,%.3f,%.3f,%.3f)",
            variant, *radio_in_base[:3], *grasp_pose[:3], *grasp_pose[3:],
        )

        # 2. waypoint POSES (positions only; IK solved per-step in forward())
        self._waypoints = vertical_descent_waypoints(grasp_pose)
        logger.info("PickUpGrasp: %d waypoints (CLOSED-LOOP, INTERP=%d, retry≤%d, err_thresh=%.3f rad)",
                    len(self._waypoints), self.INTERP_STEPS_PER_WAYPOINT,
                    self.MAX_RETRIES_PER_WAYPOINT, self.JOINT_ERROR_THRESHOLD_RAD)

        # 3. capture starting state (used to hold L_arm + L_grip throughout)
        raw_proprio = obs.get("robot_r1::proprio")
        if raw_proprio is None:
            raise KeyError("obs missing 'robot_r1::proprio'")
        self._held_state23 = extract_state_23d(raw_proprio).astype(np.float32)

        # 4. (optional) pre-compute approach→grasp bridge so the COMMAND
        #    stream is smooth across the phase boundary. We don't know the
        #    first sub-traj frame yet (planned lazily), so we bridge to the
        #    held state23 — close enough for a smooth first action.
        last_cmd = self.shared_state.get("_last_emitted_action") if self.shared_state else None
        if last_cmd is not None:
            last_action = (last_cmd.detach().cpu().numpy().astype(np.float32).reshape(-1)
                           if hasattr(last_cmd, "detach")
                           else np.asarray(last_cmd, dtype=np.float32).reshape(-1))
            # Bridge target = held state23 with R_grip=+1 (open) and base=0
            target = self._held_state23.copy()
            target[0:3] = 0.0
            target[22] = 1.0
            self._bridge_traj = linear_bridge_to(last_action, target).astype(np.float32)
        else:
            self._bridge_traj = np.zeros((0, 23), dtype=np.float32)
        self._bridge_idx = 0
        logger.info("PickUpGrasp.on_enter: bridge_len=%d (approach→grasp smoothing)",
                    len(self._bridge_traj))

        # 5. reset closed-loop state machine
        self._wp_idx = 0
        self._retry_count = 0
        self._sub_traj = None
        self._sub_traj_idx = 0
        self._last_ik_q = None
        self._closing = False
        self._close_frames_emitted = 0
        self._done_internal = False
        self._last_emitted_action = None

    def forward(self, obs) -> th.Tensor:
        # ── Branch A: emitting pre-grasp bridge frames ──
        if self._bridge_traj is not None and self._bridge_idx < len(self._bridge_traj):
            a = self._bridge_traj[self._bridge_idx]
            self._bridge_idx += 1
            self._last_emitted_action = a
            return th.from_numpy(a).to(th.float32)

        # ── Branch B: emitting close-gripper hold frames ──
        if self._closing:
            if self._close_frames_emitted >= self.CLOSE_GRIPPER_FRAMES:
                self._done_internal = True
            base = (self._last_emitted_action if self._last_emitted_action is not None
                    else np.zeros(23, dtype=np.float32))
            a = base.copy()
            a[22] = -1.0
            self._close_frames_emitted += 1
            self._last_emitted_action = a
            return th.from_numpy(a).to(th.float32)

        # ── Branch C: emitting current sub-trajectory ──
        if self._sub_traj is not None and self._sub_traj_idx < len(self._sub_traj):
            a = self._sub_traj[self._sub_traj_idx]
            self._sub_traj_idx += 1
            self._last_emitted_action = a
            return th.from_numpy(a).to(th.float32)

        # ── Branch D: sub-traj exhausted (or first call). Check & plan. ──
        current_q11 = self._current_q11(obs)

        # If we just finished a sub-traj, check convergence
        if self._sub_traj is not None and self._last_ik_q is not None:
            err = float(np.abs(current_q11 - self._last_ik_q).max())
            if err < self.JOINT_ERROR_THRESHOLD_RAD:
                logger.info("PickUpGrasp wp %d/%d: CONVERGED (err=%.4f rad, %d retries)",
                            self._wp_idx + 1, len(self._waypoints), err, self._retry_count)
                self._wp_idx += 1
                self._retry_count = 0
                self._last_ik_q = None
            elif self._retry_count >= self.MAX_RETRIES_PER_WAYPOINT:
                logger.warning("PickUpGrasp wp %d/%d: GIVING UP (err=%.4f rad, %d retries exhausted)",
                               self._wp_idx + 1, len(self._waypoints), err, self._retry_count)
                self._wp_idx += 1
                self._retry_count = 0
                self._last_ik_q = None
            else:
                self._retry_count += 1
                logger.info("PickUpGrasp wp %d/%d: NOT converged (err=%.4f rad), RETRY %d/%d",
                            self._wp_idx + 1, len(self._waypoints), err,
                            self._retry_count, self.MAX_RETRIES_PER_WAYPOINT)
                # Re-plan from current state (now closer to target)
                self._plan_sub_traj_for_current_wp(current_q11)
                a = self._sub_traj[0]
                self._sub_traj_idx = 1
                self._last_emitted_action = a
                return th.from_numpy(a).to(th.float32)

        # All waypoints done → start close-gripper tail
        if self._wp_idx >= len(self._waypoints):
            self._closing = True
            base = (self._last_emitted_action if self._last_emitted_action is not None
                    else np.zeros(23, dtype=np.float32))
            a = base.copy()
            a[22] = -1.0
            self._close_frames_emitted = 1
            self._last_emitted_action = a
            return th.from_numpy(a).to(th.float32)

        # Plan next waypoint
        self._plan_sub_traj_for_current_wp(current_q11)
        a = self._sub_traj[0]
        self._sub_traj_idx = 1
        self._last_emitted_action = a
        return th.from_numpy(a).to(th.float32)

    def is_done(self, obs, phase_detector) -> bool:
        return self._done_internal

    def reset(self) -> None:
        super().reset()
        self._waypoints = None
        self._held_state23 = None
        self._wp_idx = 0
        self._retry_count = 0
        self._sub_traj = None
        self._sub_traj_idx = 0
        self._last_ik_q = None
        self._bridge_traj = None
        self._bridge_idx = 0
        self._closing = False
        self._close_frames_emitted = 0
        self._done_internal = False
        self._last_emitted_action = None
