"""ROS-camera and UDP transport for the Franka + parallel gripper ACT runner."""

from __future__ import annotations

import os
import socket
import struct
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState


@dataclass(frozen=True)
class RobotObservation:
    arm_joints: np.ndarray
    ee_pose: np.ndarray
    gripper_width: float
    gripper_binary: float
    images: dict[str, np.ndarray]
    state_age_s: float
    image_age_s: dict[str, float]
    sequence: int


class FrankaGripperTransport(Node):
    """Receive relayed robot state/cameras and send Yin's 7-float ACT packets.

    Yin's ``udp_to_twist_bridge.py`` accepts ``<7f>`` in delta mode:
    ``dx dy dz droll dpitch dyaw gripper``.  The first six values are pose
    increments in SI units; the bridge divides them by packet inter-arrival
    time and feeds a base-frame Cartesian velocity controller.  The final
    value is a binary gripper request handled by Franka's Move/Grasp actions.
    """

    def __init__(
        self,
        *,
        yin_host: str = "10.10.10.2",
        command_port: int = 8888,
        head_topic: str = "/camera/cam_head/color/image_raw",
        wrist_topic: str = "/camera/cam_wrist/color/image_raw",
        joint_topic: str = "/fr3/filtered_joint_states",
        pose_topic: str = "/franka_robot_state_broadcaster/current_pose",
        gripper_topic: str = "/franka_gripper/joint_states",
        gripper_close_threshold: float = 0.04,
        gripper_open_threshold: float = 0.055,
        enable_commands: bool = False,
        enable_cameras: bool = True,
    ) -> None:
        super().__init__("act_gripper_transport")
        self._lock = threading.Lock()
        self._bridge = CvBridge() if enable_cameras else None
        cv2.setNumThreads(1)  # Avoid image callback CPU oversubscription during inference.
        self._images: dict[str, np.ndarray] = {}
        self._image_times: dict[str, float] = {}
        self._arm_joints = np.full(7, np.nan, dtype=np.float32)
        self._ee_pose = np.full(7, np.nan, dtype=np.float64)
        self._gripper_width = float("nan")
        self._joint_time = 0.0
        self._pose_time = 0.0
        self._gripper_time = 0.0
        self._sequence = 0
        self._gripper_close_threshold = float(gripper_close_threshold)
        self._gripper_open_threshold = max(
            float(gripper_open_threshold), self._gripper_close_threshold
        )
        self._gripper_binary: float | None = None
        self._command_enabled = bool(enable_commands)
        self._callback_counts = {
            "cam_head": 0,
            "cam_wrist": 0,
            "joints": 0,
            "pose": 0,
            "gripper": 0,
        }

        if enable_cameras:
            # The native wrist rectified stream uses sensor-data QoS
            # (best-effort/volatile). A reliable subscription can be listed by
            # ROS but still receive zero samples.
            self.create_subscription(
                Image, head_topic, lambda msg: self._image_cb(msg, "cam_head"), 10
            )
            wrist_reliability = os.environ.get("ACT_WRIST_RELIABILITY", "best_effort")
            if wrist_reliability not in {"best_effort", "reliable"}:
                raise ValueError("ACT_WRIST_RELIABILITY must be best_effort or reliable")
            wrist_qos = QoSProfile(
                depth=1,
                reliability=(ReliabilityPolicy.RELIABLE if wrist_reliability == "reliable" else ReliabilityPolicy.BEST_EFFORT),
                durability=DurabilityPolicy.VOLATILE,
            )
            self.get_logger().info(f"wrist subscription QoS: {wrist_reliability}, depth=1")
            self.create_subscription(
                Image, wrist_topic, lambda msg: self._image_cb(msg, "cam_wrist"), wrist_qos
            )
        self.create_subscription(JointState, joint_topic, self._joint_cb, 10)
        self.create_subscription(PoseStamped, pose_topic, self._pose_cb, 10)
        self.create_subscription(JointState, gripper_topic, self._gripper_cb, 10)
        if enable_cameras:
            self.get_logger().info(f"head camera: {head_topic}")
            self.get_logger().info(f"wrist camera: {wrist_topic}")
        self.get_logger().info(
            f"state topics: joints={joint_topic} pose={pose_topic} gripper={gripper_topic}"
        )

        self._command_sock = (
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if self._command_enabled
            else None
        )
        self._command_target = (yin_host, int(command_port))
        if self._command_enabled:
            self.get_logger().info(
                f"command UDP enabled -> {yin_host}:{command_port} (<7f delta+gripper)"
            )
        else:
            self.get_logger().info("command UDP disabled (dry-run; no arm/gripper packets)")

        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self)
        self._executor_thread = threading.Thread(
            target=self._executor.spin, name="camera-ros", daemon=True
        )
        self._executor_thread.start()

    def _image_cb(self, msg: Image, name: str) -> None:
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv2.cvtColor(np.asarray(bgr), cv2.COLOR_BGR2RGB)
            with self._lock:
                self._images[name] = np.ascontiguousarray(rgb)
                self._image_times[name] = time.monotonic()
                self._callback_counts[name] += 1
        except Exception as exc:
            self.get_logger().warning(f"image conversion failed for {name}: {exc}")

    @staticmethod
    def _joint_map(msg: JointState) -> dict[str, float]:
        return {
            str(name): float(msg.position[index])
            for index, name in enumerate(msg.name)
            if index < len(msg.position) and np.isfinite(float(msg.position[index]))
        }

    def _joint_cb(self, msg: JointState) -> None:
        values = self._joint_map(msg)
        names = [f"fr3_joint{index}" for index in range(1, 8)]
        if not all(name in values for name in names):
            return
        with self._lock:
            self._arm_joints = np.asarray([values[name] for name in names], dtype=np.float32)
            self._joint_time = time.monotonic()
            self._sequence += 1
            self._callback_counts["joints"] += 1

    def _pose_cb(self, msg: PoseStamped) -> None:
        pose = np.asarray(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(pose).all():
            return
        with self._lock:
            self._ee_pose = pose
            self._pose_time = time.monotonic()
            self._callback_counts["pose"] += 1

    def _gripper_cb(self, msg: JointState) -> None:
        values = self._joint_map(msg)
        if "fr3_finger_joint1" in values and "fr3_finger_joint2" in values:
            width = values["fr3_finger_joint1"] + values["fr3_finger_joint2"]
        else:
            width = float("nan")
        if "gripper" in values:
            binary = 1.0 if values["gripper"] >= 0.5 else 0.0
        elif np.isfinite(width):
            previous = self._gripper_binary
            if previous is None:
                binary = 1.0 if width <= self._gripper_close_threshold else 0.0
            elif previous >= 0.5:
                binary = 0.0 if width >= self._gripper_open_threshold else 1.0
            else:
                binary = 1.0 if width <= self._gripper_close_threshold else 0.0
        else:
            return
        with self._lock:
            self._gripper_width = width
            self._gripper_binary = binary
            self._gripper_time = time.monotonic()
            self._callback_counts["gripper"] += 1

    def snapshot(self) -> RobotObservation | None:
        now = time.monotonic()
        with self._lock:
            if min(self._joint_time, self._pose_time, self._gripper_time) <= 0.0:
                return None
            images = {key: value.copy() for key, value in self._images.items()}
            image_age = {
                key: now - stamp for key, stamp in self._image_times.items()
            }
            return RobotObservation(
                arm_joints=self._arm_joints.copy(),
                ee_pose=self._ee_pose.copy(),
                gripper_width=float(self._gripper_width),
                gripper_binary=float(self._gripper_binary),
                images=images,
                state_age_s=max(
                    now - self._joint_time,
                    now - self._pose_time,
                    now - self._gripper_time,
                ),
                image_age_s=image_age,
                sequence=self._sequence,
            )

    def wait_ready(self, timeout_s: float, image_timeout_s: float = 1.0) -> RobotObservation:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            obs = self.snapshot()
            if obs is not None:
                has_state = (
                    obs.state_age_s <= 0.5
                    and np.isfinite(obs.arm_joints).all()
                    and np.isfinite(obs.ee_pose).all()
                    and np.isfinite(obs.gripper_binary)
                )
                has_images = all(
                    name in obs.images and obs.image_age_s.get(name, float("inf")) <= image_timeout_s
                    for name in ("cam_head", "cam_wrist")
                )
                if has_state and has_images:
                    return obs
            time.sleep(0.02)
        with self._lock:
            counts = dict(self._callback_counts)
            missing = [name for name, count in counts.items() if count == 0]
        detail = ", ".join(missing) if missing else "all callbacks arrived but became stale"
        raise TimeoutError(
            "robot state/cameras not ready after "
            f"{float(timeout_s):g}s; missing callbacks: {detail}. "
            "Start the FRST relay and both cameras with ROS_DOMAIN_ID=0, "
            "then check the configured topics. Callback counts: "
            f"{counts}"
        )

    def wait_state_ready(self, timeout_s: float = 10.0) -> RobotObservation:
        """Wait for the arm/pose/gripper relay without requiring camera topics."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            obs = self.snapshot()
            if obs is not None and (
                obs.state_age_s <= 0.5
                and np.isfinite(obs.arm_joints).all()
                and np.isfinite(obs.ee_pose).all()
                and np.isfinite(obs.gripper_binary)
            ):
                return obs
            time.sleep(0.02)
        with self._lock:
            counts = dict(self._callback_counts)
        raise TimeoutError(
            f"robot state not ready after {float(timeout_s):g}s; "
            f"callback counts: {counts}"
        )

    def send_delta(self, linear_delta: np.ndarray, angular_delta: np.ndarray, gripper: float) -> None:
        if not self._command_enabled:
            raise RuntimeError("command send requested while transport is in dry-run mode")
        assert self._command_sock is not None
        linear = np.asarray(linear_delta, dtype=np.float32).reshape(3)
        angular = np.asarray(angular_delta, dtype=np.float32).reshape(3)
        if not np.isfinite(linear).all() or not np.isfinite(angular).all():
            raise ValueError("non-finite Cartesian delta")
        grip = 1.0 if float(gripper) >= 0.5 else 0.0
        payload = struct.pack(
            "<7f", float(linear[0]), float(linear[1]), float(linear[2]),
            float(angular[0]), float(angular[1]), float(angular[2]), grip,
        )
        self._command_sock.sendto(payload, self._command_target)

    def probe_command_receiver(self, timeout_s: float = 1.0) -> bool:
        """Use Yin's transport-only YIN?/YIN! probe without publishing motion."""
        if not self._command_enabled:
            return False
        assert self._command_sock is not None
        token = os.urandom(4)
        previous_timeout = self._command_sock.gettimeout()
        try:
            self._command_sock.settimeout(max(0.05, float(timeout_s)))
            self._command_sock.sendto(b"YIN?" + token, self._command_target)
            reply, source = self._command_sock.recvfrom(64)
            return reply == b"YIN!" + token and source[0] == self._command_target[0]
        except (OSError, socket.timeout):
            return False
        finally:
            self._command_sock.settimeout(previous_timeout)

    def send_zero(self, gripper: float, repeats: int = 5, interval_s: float = 0.02) -> None:
        for _ in range(max(1, int(repeats))):
            self.send_delta(np.zeros(3), np.zeros(3), gripper)
            if interval_s > 0:
                time.sleep(float(interval_s))

    def destroy_node(self) -> bool:
        if self._command_sock is not None:
            try:
                self._command_sock.close()
            except OSError:
                pass
        self._executor.shutdown(timeout_sec=1.0)
        super().destroy_node()
        return True
