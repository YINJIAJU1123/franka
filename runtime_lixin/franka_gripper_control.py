"""Shared Franka FK, Cartesian servo, and task reset for ACT and replay."""

from __future__ import annotations

import time

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from franka_gripper_transport import FrankaGripperTransport


class CartesianServo:
    def __init__(
        self,
        urdf_path: str,
        *,
        eef_frame: str = "fr3_link8",
        position_kp: float = 1.5,
        orientation_kp: float = 1.5,
        max_linear_velocity: float = 0.12,
        max_angular_velocity: float = 0.35,
    ) -> None:
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        if not self.model.existFrame(eef_frame):
            raise RuntimeError(f"EEF frame {eef_frame!r} not found in {urdf_path}")
        self.eef_id = self.model.getFrameId(eef_frame)
        self.position_kp = max(0.0, float(position_kp))
        self.orientation_kp = max(0.0, float(orientation_kp))
        self.max_linear_velocity = max(0.0, float(max_linear_velocity))
        self.max_angular_velocity = max(0.0, float(max_angular_velocity))

    def clip_joints(self, joints: np.ndarray) -> np.ndarray:
        q = np.asarray(joints, dtype=np.float64).reshape(-1)[:7]
        lower = np.asarray(self.model.lowerPositionLimit[:7], dtype=np.float64)
        upper = np.asarray(self.model.upperPositionLimit[:7], dtype=np.float64)
        return np.clip(q, lower, upper)

    def fk(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(joints, dtype=np.float64).reshape(-1)[:7]
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        transform = self.data.oMf[self.eef_id]
        position = np.asarray(transform.translation, dtype=np.float64).copy()
        quaternion = np.asarray(pin.Quaternion(transform.rotation).coeffs(), dtype=np.float64)
        return position, quaternion

    def pose_error(
        self,
        current_pose: np.ndarray,
        target_joints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        current_pose = np.asarray(current_pose, dtype=np.float64).reshape(7)
        current_position = current_pose[:3]
        current_rotation = Rotation.from_quat(current_pose[3:]).as_matrix()
        target_position, target_quaternion = self.fk(target_joints)
        target_rotation = Rotation.from_quat(target_quaternion).as_matrix()
        position_error = target_position - current_position
        # R_target R_current^T gives the shortest orientation error expressed in
        # Yin's base frame (the controller's expected frame).
        rotation_error = Rotation.from_matrix(target_rotation @ current_rotation.T).as_rotvec()
        return position_error, rotation_error, target_position, target_quaternion

    def delta(
        self,
        current_pose: np.ndarray,
        target_joints: np.ndarray,
        dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        position_error, rotation_error, target_position, target_quaternion = self.pose_error(
            current_pose, target_joints
        )
        linear_velocity = self.position_kp * position_error
        angular_velocity = self.orientation_kp * rotation_error
        if self.max_linear_velocity > 0.0:
            linear_velocity = np.clip(
                linear_velocity, -self.max_linear_velocity, self.max_linear_velocity
            )
        if self.max_angular_velocity > 0.0:
            angular_velocity = np.clip(
                angular_velocity, -self.max_angular_velocity, self.max_angular_velocity
            )
        dt_s = max(1e-3, float(dt_s))
        return (
            linear_velocity * dt_s,
            angular_velocity * dt_s,
            position_error,
            rotation_error,
        )


TASK_RESET_TARGETS = {
    "drawer": np.asarray(
        [0.08900524, -0.27886856, -0.08568542, -2.05098510,
         -0.02978628, 1.87084568, 0.86323535], dtype=np.float64
    ),
    "desk": np.asarray(
        [0.09002313, -0.27330105, -0.07433387, -2.05646825,
         -0.02605186, 1.88057631, 0.87309942], dtype=np.float64
    ),
}


def _pose_increment(previous_pose: np.ndarray, target_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    previous_pose = np.asarray(previous_pose, dtype=np.float64).reshape(7)
    target_pose = np.asarray(target_pose, dtype=np.float64).reshape(7)
    previous_rotation = Rotation.from_quat(previous_pose[3:]).as_matrix()
    target_rotation = Rotation.from_quat(target_pose[3:]).as_matrix()
    return (
        target_pose[:3] - previous_pose[:3],
        Rotation.from_matrix(target_rotation @ previous_rotation.T).as_rotvec(),
    )


def stream_task_reset(
    transport: FrankaGripperTransport,
    servo: CartesianServo,
    observation,
    target_joints: np.ndarray,
    duration_s: float,
    hz: float,
    allow_motion: bool,
    state_timeout_s: float = 0.35,
) -> None:
    current_joints = servo.clip_joints(observation.arm_joints)
    target_joints = servo.clip_joints(target_joints)
    steps = max(1, int(np.ceil(max(0.0, float(duration_s)) * float(hz))))
    previous_pose = np.concatenate(servo.fk(current_joints))
    interval = 1.0 / float(hz)
    started = time.monotonic()
    next_send = started
    first_delta = None
    for index in range(1, steps + 1):
        if allow_motion:
            current = transport.snapshot()
            if current is None or current.state_age_s > state_timeout_s:
                raise RuntimeError("robot state stale during task reset")
        fraction = index / float(steps)
        interpolated = current_joints + fraction * (target_joints - current_joints)
        target_pose = np.concatenate(servo.fk(interpolated))
        linear_delta, angular_delta = _pose_increment(previous_pose, target_pose)
        if first_delta is None:
            first_delta = (linear_delta.copy(), angular_delta.copy())
        if allow_motion:
            transport.send_delta(linear_delta, angular_delta, 0.0)
            next_send += interval
            wait_s = next_send - time.monotonic()
            if wait_s > 0.0:
                time.sleep(wait_s)
        previous_pose = target_pose
    elapsed = time.monotonic() - started
    print(
        f">>> task reset {'sent' if allow_motion else 'simulated'}: "
        f"{steps} packets/{elapsed:.2f}s, target_q="
        f"{np.array2string(target_joints, precision=4)}",
        flush=True,
    )
    if first_delta is not None:
        print(
            f">>> reset first delta: dp={np.array2string(first_delta[0], precision=5)} "
            f"dr={np.array2string(first_delta[1], precision=5)}",
            flush=True,
        )

