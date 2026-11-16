"""Lixin-side receiver for Yin's FRST v2 robot-state relay."""

from __future__ import annotations

import argparse
import socket
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from .frst_protocol import GRIPPER_VALID, JOINTS_VALID, POSE_VALID, FRSTState, decode_frst

STATE_RELAY_PORT = 8896
STATE_SOURCE_HOST = "10.10.10.2"
PUBLISH_HZ = 60.0
STALE_TIMEOUT_S = 0.5
FRANKA_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
GRIPPER_CLOSED_WIDTH_THRESHOLD = 0.04
FILTERED_FR3_TOPIC = "/fr3/filtered_joint_states"
FRANKA_EEF_POSE_TOPIC = "/franka_robot_state_broadcaster/current_pose"
FRANKA_GRIPPER_JOINT_STATE_TOPIC = "/franka_gripper/joint_states"
FRANKA_GRIPPER_BINARY_COMMAND_TOPIC = "/franka_gripper/binary_command"


def _stamp_message(stamp_ns: int, message) -> None:
    message.header.stamp.sec = int(stamp_ns // 1_000_000_000)
    message.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)


class StateRelayReceiver(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("tswam_state_relay_receiver")
        self.args = args
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.filtered_pub = self.create_publisher(JointState, FILTERED_FR3_TOPIC, 10)
        self.pose_pub = self.create_publisher(PoseStamped, FRANKA_EEF_POSE_TOPIC, 10)
        self.gripper_state_pub = self.create_publisher(JointState, FRANKA_GRIPPER_JOINT_STATE_TOPIC, 10)
        self.gripper_command_pub = self.create_publisher(Float32, FRANKA_GRIPPER_BINARY_COMMAND_TOPIC, 10)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Keep this UDP state port exclusive; duplicate receivers would divert packets.
        self.sock.bind((args.bind_host, args.port))
        self.sock.setblocking(False)
        self.latest: FRSTState | None = None
        self.latest_recv_time = 0.0
        self.invalid_packets = 0
        self.packet_count = 0
        self.timer = self.create_timer(1.0 / args.publish_hz, self._tick)
        source = args.source_host or "any host"
        self.get_logger().info(
            f"Listening for FRST v2 on {args.bind_host}:{args.port} from {source}; "
            f"publishing state topics at {args.publish_hz:.1f} Hz"
        )

    def _tick(self) -> None:
        self._poll_socket()
        packet = self.latest
        if packet is None or time.monotonic() - self.latest_recv_time > self.args.stale_timeout:
            return
        self._publish(packet)

    def _poll_socket(self) -> None:
        while True:
            try:
                data, sender = self.sock.recvfrom(2048)
            except BlockingIOError:
                return
            except OSError:
                return
            if self.args.source_host and sender[0] != self.args.source_host:
                continue
            try:
                packet = decode_frst(data)
            except ValueError as exc:
                self.invalid_packets += 1
                if self.invalid_packets == 1 or self.invalid_packets % 100 == 0:
                    self.get_logger().warning(f"Ignoring invalid FRST packet from {sender[0]}: {exc}")
                continue
            self.latest = packet
            self.latest_recv_time = time.monotonic()
            self.packet_count += 1

    def _publish(self, packet: FRSTState) -> None:
        values = packet.values
        if packet.flags & JOINTS_VALID:
            message = JointState()
            _stamp_message(packet.joint_ns, message)
            message.name = list(FRANKA_JOINTS)
            message.position = list(values[0:7])
            message.velocity = list(values[7:14])
            message.effort = list(values[14:21])
            self.pub.publish(message)
            self.filtered_pub.publish(message)
        if packet.flags & POSE_VALID:
            message = PoseStamped()
            _stamp_message(packet.pose_ns, message)
            message.header.frame_id = "fr3_link0"
            message.pose.position.x, message.pose.position.y, message.pose.position.z = values[21:24]
            message.pose.orientation.x, message.pose.orientation.y = values[24:26]
            message.pose.orientation.z, message.pose.orientation.w = values[26:28]
            self.pose_pub.publish(message)
        if packet.flags & GRIPPER_VALID:
            width = float(values[28])
            binary = 1.0 if width <= self.args.gripper_closed_width_threshold else 0.0
            message = JointState()
            _stamp_message(packet.gripper_ns, message)
            message.name = ["fr3_finger_joint1", "fr3_finger_joint2", "gripper"]
            message.position = [width / 2.0, width / 2.0, binary]
            message.velocity = [0.0, 0.0, 0.0]
            message.effort = [0.0, 0.0, 0.0]
            self.gripper_state_pub.publish(message)
            command = Float32()
            command.data = binary
            self.gripper_command_pub.publish(command)

    def close(self) -> None:
        self.sock.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=STATE_RELAY_PORT)
    parser.add_argument("--source-host", default=STATE_SOURCE_HOST)
    parser.add_argument("--publish-hz", type=float, default=PUBLISH_HZ)
    parser.add_argument("--stale-timeout", type=float, default=STALE_TIMEOUT_S)
    parser.add_argument("--gripper-closed-width-threshold", type=float, default=GRIPPER_CLOSED_WIDTH_THRESHOLD)
    args = parser.parse_args(argv)
    if args.port <= 0 or args.publish_hz <= 0 or args.stale_timeout <= 0:
        parser.error("port, publish-hz and stale-timeout must be positive")
    if args.gripper_closed_width_threshold < 0:
        parser.error("gripper-closed-width-threshold must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = StateRelayReceiver(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
