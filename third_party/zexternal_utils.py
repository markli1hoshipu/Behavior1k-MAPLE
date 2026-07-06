
"""
controllers:
  base:
    name: HolonomicBaseJointController
    motor_type: velocity
    vel_kp: 150
    command_input_limits: [[-1, -1, -1], [1, 1, 1]]
    command_output_limits: [[-0.75, -0.75, -1], [1, 1, 1]]
    use_impedances: false
"""

# x forward, y left positive, z up

import numpy as np
from math import atan2, sqrt
import math
from typing import Optional, Dict, Any
import glob
import pandas as pd
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
except ImportError:  # only `load_env_config` needs it; IK does not.
    def load_dotenv(*_args, **_kwargs):
        return False
from scipy.spatial.transform import Rotation as R
from collections import OrderedDict
import base64

# used to get close to target in 6D pose
def get_navigation_action(target_pose):
    """
    input: 1x7 pose ; output: 1x3 vx,vy,xz; 
    """
    # Parameters (tune as needed)
    yaw_threshold = 0.20        # m of lateral offset before turning-in-place
    dist_threshold = 0.80        # m of distance before driving forward
    max_lin = 0.5                # m/s
    max_ang = 0.4                # rad/s
    k_yaw = 1.0                  # proportional gain for yaw alignment
    k_dist = 0.8                 # proportional gain for forward velocity

    # Ensure numpy array and extract position (we ignore orientation in this simple controller)
    pose = np.asarray(target_pose).reshape(-1)
    x, y, _z = pose[:3]

    # 1) Align heading toward the target if lateral offset is significant
    # Using desired heading error as atan2(y, x); positive y -> turn left (positive wz)
    heading_error = atan2(y, x)  # radians
    if abs(y) > yaw_threshold and abs(heading_error) > 0.05:
        wz = float(np.clip(k_yaw * heading_error, -max_ang, max_ang))
        return np.array([0.0, 0.0, wz], dtype=np.float32)

    # 2) If roughly aligned, drive forward if still far from the goal
    dist = sqrt(x * x + y * y)
    if dist > dist_threshold:
        vx = float(np.clip(k_dist * dist, -max_lin, max_lin))
        return np.array([vx, 0.0, 0.0], dtype=np.float32)

    # 3) Close enough and roughly aligned: no motion
    return np.zeros(3, dtype=np.float32)

# used to get first action for model in each entire task
def get_first_action(task_num: str, data_path: Path = Path("/work/mark/lerobot/2025-challenge-demos/data")) -> np.ndarray:
    """
    Navigate to data folder, find first parquet file in specified folder,
    and extract the action array from the first line.
    """
    target_folder = data_path / f"task-{task_num.zfill(4)}"
    if not target_folder.exists():
        print(f"Error: Folder {target_folder} does not exist")
        return None
    parquet_files = glob.glob(str(target_folder / "*.parquet"))
    if not parquet_files:
        print(f"Error: No parquet files found in {target_folder}")
        return None
    # parquet_files.sort() # don't think we really need that
    first_parquet = parquet_files[0]
    print(f"Reading from: {first_parquet}")
    df = pd.read_parquet(first_parquet)
    if df.empty:
        print("Error: Parquet file is empty")
        return None
    first_row = df.iloc[0]
    action_array = None
    if 'action' in first_row:
        action_array = first_row['action']
    writable_action_array = action_array.copy()
    writable_action_array[:3] = 0.0 
    return writable_action_array

# used to load GPT configuration from .env file
def load_env_config():
    """Load GPT configuration from .env file."""
    env_path = "/work/mark/behavior-1k/b1k_baselines/baselines/openpi/zexternal/.env"
    load_dotenv(env_path)

    config = {
        'api_key': os.getenv('OPENAI_API_KEY'),
        'model': os.getenv('OPENAI_MODEL', 'gpt-5'),
        'provider': 'gpt'
    }

    if not config['api_key']:
        print(f"Warning: OPENAI_API_KEY not found in {env_path}")
    else:
        print(f"✓ Loaded GPT config - Model: {config['model']}")

    return config

# ik solver
def mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    return np.array([x, y, z, w], dtype=np.float64)

def solve_ik_r1pro(
    mode: str,
    target_pos_m: np.ndarray,
    target_quat_xyzw: np.ndarray,
    urdf_path: str = "./r1pro.urdf",
    extra_link_xyzquat: Optional[np.ndarray] = None,
    initial_guess: Optional[np.ndarray] = None,
    max_iters: int = 800,
    lr: float = 5e-2,
    pos_weight: float = 5.0,
    ori_weight: float = 1.0,
    pos_tol: Optional[float] = None,
    ori_tol: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    try:
        import torch
        import pytorch_kinematics as pk
    except Exception as e:
        raise RuntimeError(
            "pytorch_kinematics (and torch) are required. Install via: pip install pytorch-kinematics torch"
        ) from e

    if mode == "left_torso":
        desired_joint_names = [
            "torso_joint1",
            "torso_joint2",
            "torso_joint3",
            "torso_joint4",
            "left_arm_joint1",
            "left_arm_joint2",
            "left_arm_joint3",
            "left_arm_joint4",
            "left_arm_joint5",
            "left_arm_joint6",
            "left_arm_joint7",
        ]
        eef_link_name = "left_gripper_link"
    elif mode == "right_torso":
        desired_joint_names = [
            "torso_joint1",
            "torso_joint2",
            "torso_joint3",
            "torso_joint4",
            "right_arm_joint1",
            "right_arm_joint2",
            "right_arm_joint3",
            "right_arm_joint4",
            "right_arm_joint5",
            "right_arm_joint6",
            "right_arm_joint7",
        ]
        eef_link_name = "right_gripper_link"
    else:
        raise ValueError("mode must be left_torso or right_torso")

    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # Input target tensors
    target_pos = torch.tensor(target_pos_m, dtype=torch.float64)
    tq = torch.tensor(target_quat_xyzw, dtype=torch.float64)
    tq = tq / torch.clamp(tq.norm(), min=1e-12)

    # Quaternion -> rotation matrix (for robust orientation error)
    def quat_xyzw_to_rotmat(q: torch.Tensor) -> torch.Tensor:
        x, y, z, w = q
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return torch.stack(
            [
                torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)]),
                torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)]),
                torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]),
            ]
        )

    R_target = quat_xyzw_to_rotmat(tq)

    with open(urdf_path, "rb") as f:
        urdf_bytes = f.read()

    chain = pk.build_serial_chain_from_urdf(urdf_bytes, end_link_name=eef_link_name)
    try:
        chain = chain.to(dtype=torch.float64)
    except Exception:
        pass

    # Initial guess
    if initial_guess is None:
        q0_np = np.zeros(len(desired_joint_names), dtype=np.float64)
    else:
        q0_np = np.asarray(initial_guess, dtype=np.float64)
        if q0_np.shape[0] != len(desired_joint_names):
            raise ValueError("initial_guess must have 11 elements (4 torso + 7 arm)")

    q = torch.tensor(q0_np, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([q], lr=lr)

    # Build optional tool transform as Transform3d-like matrix multiplication (xyz + quaternion only)
    def compose_tool(T: torch.Tensor) -> torch.Tensor:
        if extra_link_xyzquat is not None:
            vals = torch.tensor(extra_link_xyzquat, dtype=torch.float64)
            if vals.numel() != 7:
                raise ValueError("extra_link_xyzquat must have 7 elements: tx ty tz qx qy qz qw")
            xyz = vals[:3]
            quat = vals[3:] / torch.clamp(vals[3:].norm(), min=1e-12)
            x, y, z, w = quat
            # quat -> R
            xx, yy, zz = x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z
            wx, wy, wz = w * x, w * y, w * z
            R = torch.stack(
                [
                    torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)]),
                    torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)]),
                    torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]),
                ]
            )
            Ttool = torch.eye(4, dtype=torch.float64)
            Ttool[:3, :3] = R
            Ttool[:3, 3] = xyz
            return T @ Ttool
        return T

    # Orientation error helper using rotation matrices (geodesic distance)
    def orientation_angle_from_rotmats(R_current: torch.Tensor, R_target_: torch.Tensor) -> torch.Tensor:
        R_rel = R_target_.T @ R_current
        trace = torch.clamp(R_rel.trace(), min=-1e6, max=1e6)
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-9, 1.0 - 1e-9)
        return torch.acos(cos_angle)

    # Optimization loop
    best_q = None
    best_loss = float("inf")
    for it in range(max_iters):
        opt.zero_grad()
        # Map q into chain parameter order
        try:
            joint_names_in_chain = list(chain.get_joint_parameter_names())  # type: ignore[attr-defined]
            # Bring q into the order expected by the chain
            name_to_idx = {n: i for i, n in enumerate(desired_joint_names)}
            q_ordered = torch.stack([q[name_to_idx[n]] for n in joint_names_in_chain])
        except Exception:
            q_ordered = q

        fk_all = chain.forward_kinematics(q_ordered, end_only=False)
        T = fk_all[eef_link_name].get_matrix().squeeze(0)
        T = compose_tool(T)

        pos = T[:3, 3]
        R = T[:3, :3]
        pos_err = torch.norm(pos - target_pos)
        ori_err = orientation_angle_from_rotmats(R, R_target)
        loss = pos_weight * pos_err + ori_weight * ori_err

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_q = q.detach().clone()

        # Early stopping if within tolerances (both satisfied)
        if (pos_tol is not None) and (ori_tol is not None):
            if (pos_err.item() <= pos_tol) and (ori_err.item() <= ori_tol):
                if verbose:
                    print(
                        f"Early stop at iter {it}: pos_err={pos_err.item():.6f} <= {pos_tol}, "
                        f"ori_err={ori_err.item():.6f} <= {ori_tol}"
                    )
                break

        loss.backward()
        # Gradient clipping to avoid NaNs
        torch.nn.utils.clip_grad_norm_([q], max_norm=1.0)
        opt.step()
        # Clamp joints to a safe range
        with torch.no_grad():
            q.clamp_(min=-3.14159265, max=3.14159265)

        # if verbose and (it % 100 == 0 or it == max_iters - 1):
        #     print(f"iter {it}: loss={loss.item():.6f} pos_err={pos_err.item():.6f} ori_err={ori_err.item():.6f}")

    if best_q is None:
        best_q = q.detach().clone()

    # Compute final FK for reporting
    try:
        joint_names_in_chain = list(chain.get_joint_parameter_names())  # type: ignore[attr-defined]
        name_to_idx = {n: i for i, n in enumerate(desired_joint_names)}
        q_ordered_best = torch.stack([best_q[name_to_idx[n]] for n in joint_names_in_chain])
    except Exception:
        q_ordered_best = best_q

    fk_all_best = chain.forward_kinematics(q_ordered_best, end_only=False)
    T_best = fk_all_best[eef_link_name].get_matrix().squeeze(0)
    T_best = compose_tool(T_best)

    T_best_np = T_best.detach().cpu().numpy()
    solved_pos = T_best_np[:3, 3]
    solved_quat = mat_to_quat_xyzw(T_best_np[:3, :3])

    def quat_angle_error_rad_np(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> float:
        a = np.asarray(q1_xyzw, dtype=np.float64)
        b = np.asarray(q2_xyzw, dtype=np.float64)
        a = a / max(1e-12, np.linalg.norm(a))
        b = b / max(1e-12, np.linalg.norm(b))
        dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
        return 2.0 * math.acos(dot)

    err_pos = float(np.linalg.norm(np.asarray(target_pos_m, dtype=np.float64) - solved_pos))
    err_quat = float(quat_angle_error_rad_np(solved_quat, target_quat_xyzw))

    result: Dict[str, Any] = {
        "q_sol": best_q.detach().cpu().numpy(),
        "best_loss": float(best_loss),
        "lr": float(lr),
        "target_pos": np.asarray(target_pos_m, dtype=np.float64),
        "target_quat": np.asarray(target_quat_xyzw, dtype=np.float64),
        "solved_pos": solved_pos.astype(np.float64),
        "solved_quat": solved_quat.astype(np.float64),
        "err_pos": err_pos,
        "err_quat": err_quat,
    }

    return result




PROPRIOCEPTION_INDICES = {
    "R1Pro": OrderedDict(
        {
            "joint_qpos": np.s_[
                0:28
            ],  # Full robot joint positions, the first 6 are base joints, which is NOT allowed in standard track
            "joint_qpos_sin": np.s_[
                28:56
            ],  # Full robot joint positions, the first 6 are base joints, which is NOT allowed in standard track
            "joint_qpos_cos": np.s_[
                56:84
            ],  # Full robot joint positions, the first 6 are base joints, which is NOT allowed in standard track
            "joint_qvel": np.s_[84:112],
            "joint_qeffort": np.s_[112:140],
            "robot_pos": np.s_[140:143],  # Global pos, this is NOT allowed in standard track
            "robot_ori_cos": np.s_[143:146],  # Global ori, this is NOT allowed in standard track
            "robot_ori_sin": np.s_[146:149],  # Global ori, this is NOT allowed in standard track
            "robot_2d_ori": np.s_[149:150],  # 2D global ori, this is NOT allowed in standard track
            "robot_2d_ori_cos": np.s_[150:151],  # 2D global ori, this is NOT allowed in standard track
            "robot_2d_ori_sin": np.s_[151:152],  # 2D global ori, this is NOT allowed in standard track
            "robot_lin_vel": np.s_[152:155],
            "robot_ang_vel": np.s_[155:158],
            "arm_left_qpos": np.s_[158:165],
            "arm_left_qpos_sin": np.s_[165:172],
            "arm_left_qpos_cos": np.s_[172:179],
            "arm_left_qvel": np.s_[179:186],
            "eef_left_pos": np.s_[186:189],
            "eef_left_quat": np.s_[189:193],
            "gripper_left_qpos": np.s_[193:195],
            "gripper_left_qvel": np.s_[195:197],
            "arm_right_qpos": np.s_[197:204],
            "arm_right_qpos_sin": np.s_[204:211],
            "arm_right_qpos_cos": np.s_[211:218],
            "arm_right_qvel": np.s_[218:225],
            "eef_right_pos": np.s_[225:228],
            "eef_right_quat": np.s_[228:232],
            "gripper_right_qpos": np.s_[232:234],
            "gripper_right_qvel": np.s_[234:236],
            "trunk_qpos": np.s_[236:240],
            "trunk_qvel": np.s_[240:244],
            "base_qpos": np.s_[244:247],  # Base joint position, this is NOT allowed in standard track
            "base_qpos_sin": np.s_[247:250],  # Base joint position, this is NOT allowed in standard track
            "base_qpos_cos": np.s_[250:253],  # Base joint position, this is NOT allowed in standard track
            "base_qvel": np.s_[253:256],
        }
    ),
}

def extract_state_from_proprio(proprio_data):
    """
    We assume perfect correlation for the two gripper fingers.
    """
    # extract joint position
    base_qvel = proprio_data[PROPRIOCEPTION_INDICES["R1Pro"]["base_qvel"]]  # 3
    trunk_qpos = proprio_data[PROPRIOCEPTION_INDICES["R1Pro"]["trunk_qpos"]]  # 4
    arm_left_qpos = proprio_data[PROPRIOCEPTION_INDICES["R1Pro"]["arm_left_qpos"]]  #  7
    arm_right_qpos = proprio_data[PROPRIOCEPTION_INDICES["R1Pro"]["arm_right_qpos"]]  #  7
    left_gripper_width = proprio_data[PROPRIOCEPTION_INDICES["R1Pro"]["gripper_left_qpos"]].sum(axis=-1, keepdims=True)  # 1
    right_gripper_width = proprio_data[PROPRIOCEPTION_INDICES["R1Pro"]["gripper_right_qpos"]].sum(axis=-1, keepdims=True)  # 1
    
    # fix gripper state to map
    return np.concatenate([
        base_qvel,
        trunk_qpos,
        arm_left_qpos,
        left_gripper_width,
        arm_right_qpos,
        right_gripper_width,
    ], axis=-1)

def extract_action_from_proprio(proprio_data):
    """
    We assume perfect correlation for the two gripper fingers.
    """
    # extract joint position
    base_qvel = np.zeros_like(proprio_data[PROPRIOCEPTION_INDICES["R1Pro"]["base_qvel"]])  # 3
    trunk_qpos = proprio_data[PROPRIOCEPTION_INDICES["R1Pro"]["trunk_qpos"]]  # 4
    arm_left_qpos = proprio_data[PROPRIOCEPTION_INDICES["R1Pro"]["arm_left_qpos"]]  #  7
    arm_right_qpos = proprio_data[PROPRIOCEPTION_INDICES["R1Pro"]["arm_right_qpos"]]  #  7
    left_gripper_width = proprio_data[PROPRIOCEPTION_INDICES["R1Pro"]["gripper_left_qpos"]].sum(axis=-1, keepdims=True)  # 1
    if left_gripper_width > 0.099:
        left_gripper_action = [1]
    else:
        left_gripper_action = [-1]
    right_gripper_width = proprio_data[PROPRIOCEPTION_INDICES["R1Pro"]["gripper_right_qpos"]].sum(axis=-1, keepdims=True)  # 1
    if right_gripper_width > 0.099:
        right_gripper_action = [1]
    else:
        right_gripper_action = [-1]
    
    # fix gripper state to map
    return np.concatenate([
        base_qvel,
        trunk_qpos,
        arm_left_qpos,
        left_gripper_action,
        arm_right_qpos,
        right_gripper_action,
    ], axis=-1)



def ndarray_to_base64(arr): # this is a helper function to convert numpy array to base64 string
    return base64.b64encode(arr.tobytes()).decode('ascii')



def apply_roll_180_deg(pose):
    """
    Applies a 180 degree rotation around the X axis (roll) to the orientation quaternion in the 1x7 pose.
    Args:
        pose: 1x7 array [x, y, z, x, y, z, w]
    Returns:
        1x7 array with updated orientation
    """

    pos = np.asarray(pose[:3])
    quat = np.asarray(pose[3:])

    # 180-degree roll is a quaternion [-1, 0, 0, 0] in (x, y, z, w) format
    roll_180 = R.from_euler('x', 180, degrees=True)
    q = R.from_quat(quat)
    q_new = q * roll_180 
    quat_new = q_new.as_quat()
    return np.concatenate([pos, quat_new])


def transform_pose(obj_pose_camera, camera_pose_base):
    """
    Args:
        obj_pose_camera: 1x7 array [x, y, z, rx, ry, rz, rw] of object in camera coordinates
        camera_pose_base: 1x7 array [x, y, z, rx, ry, rz, rw] of camera in base coordinates
    Returns:
        obj_pose_base: 1x7 array [x, y, z, rx, ry, rz, rw] in base coordinates
    """
    # Extract position and quaternion components
    obj_pos_cam = obj_pose_camera[:3]
    obj_quat_cam = obj_pose_camera[3:]
    
    cam_pos = camera_pose_base[:3]
    cam_quat = camera_pose_base[3:]
    
    # Convert quaternions to rotation matrices
    R_obj_cam = R.from_quat(obj_quat_cam).as_matrix()  # object in camera frame
    R_cam_world = R.from_quat(cam_quat).as_matrix()    # camera in world frame
    
    # Transform position: p_world = R_cam_world * p_cam + t_cam_world
    obj_pos_world = R_cam_world @ obj_pos_cam + cam_pos
    
    # Transform orientation: R_obj_world = R_cam_world * R_obj_cam
    R_obj_world = R_cam_world @ R_obj_cam 
    obj_quat_world = R.from_matrix(R_obj_world).as_quat()
    
    # Combine position and quaternion
    obj_pose_base = np.concatenate([obj_pos_world, obj_quat_world])
    
    return obj_pose_base


def densify_points(points, max_step=0.1):
    """
    Add intermediate points between consecutive points so that 
    the L1 distance between any two consecutive points ≤ max_step.
    
    Args:
        points (list or np.ndarray): List/array of N points (each 3D or nD).
        max_step (float): Maximum allowed L1 step size between consecutive points.
    
    Returns:
        np.ndarray: Smoothed array of points with added intermediate points.
    """
    points = np.array(points)
    dense_points = [points[0]]

    for i in range(1, len(points)):
        p0, p1 = points[i-1], points[i]
        # Use L1 (Manhattan) distance
        dist = np.linalg.norm(p1 - p0, ord=1)
        if dist > max_step:
            num_interp = int(np.ceil(dist / max_step))
            interp = np.linspace(p0, p1, num=num_interp+1)[1:]  # skip the first (already added)
            dense_points.extend(interp)
        else:
            dense_points.append(p1)

    return np.array(dense_points)















def quat_mult(q1, q2):
    """Multiply two quaternions (xyzw convention)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    quat = np.array([x, y, z, w], dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return quat / norm

def quat_rotate_vec_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ], dtype=float)
    return R @ v