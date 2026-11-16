#!/usr/bin/env python3
"""Sample official state topics and send FRST v2 at 60 Hz for every mode."""
import math
import socket
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from franka_msgs.msg import FrankaRobotState

from yin_franka_bringup.state_protocol import StatePacket


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class StateStreamer(Node):
    def __init__(self):
        super().__init__("fr3_state_streamer")
        self.declare_parameter("target_host", "10.10.10.1")
        self.declare_parameter("target_port", 8896)
        self.target = (self.get_parameter("target_host").value,
                       self.get_parameter("target_port").value)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.samples = {}
        self.sequence = 0
        self.create_subscription(JointState,
            "/franka_robot_state_broadcaster/measured_joint_states", self.joints,
            qos_profile_sensor_data)
        self.create_subscription(FrankaRobotState,
            "/franka_robot_state_broadcaster/robot_state", self.motor,
            qos_profile_system_default)
        self.create_subscription(PoseStamped,
            "/franka_robot_state_broadcaster/current_pose", self.pose,
            qos_profile_sensor_data)
        self.create_subscription(JointState, "/franka_gripper/joint_states",
                                 self.gripper, qos_profile_sensor_data)
        self.create_timer(1.0 / 60.0, self.send)
        self.get_logger().info(f"FRST v2 state -> {self.target}, 60 Hz")

    def save(self, group, stamp, values):
        if all(math.isfinite(value) for value in values):
            self.samples[group] = (time.monotonic(), stamp_ns(stamp), tuple(values))
        else:
            self.samples.pop(group, None)

    def joints(self, msg):
        try:
            indices = [list(msg.name).index(f"fr3_joint{i}") for i in range(1, 8)]
            values = [field[i] for field in (msg.position, msg.velocity, msg.effort)
                      for i in indices]
        except (ValueError, IndexError):
            self.samples.pop(0, None)
            return
        self.save(0, msg.header.stamp, values)

    def pose(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if abs(sum(v * v for v in values[3:]) - 1.0) > 1e-3:
            self.samples.pop(1, None)
            return
        self.save(1, msg.header.stamp, values)

    def motor(self, msg):
        state = msg.measured_joint_motor_state
        if len(state.position) != 7 or len(state.velocity) != 7:
            self.samples.pop(3, None)
            return
        values = list(state.position) + list(state.velocity)
        self.save(3, state.header.stamp, values)

    def gripper(self, msg):
        if len(msg.position) != 2:
            self.samples.pop(2, None)
            return
        self.save(2, msg.header.stamp, (sum(msg.position),))

    def send(self):
        now = time.monotonic()
        flags, values, stamps = 0, [], []
        for group, count in enumerate((21, 7, 1, 14)):
            sample = self.samples.get(group)
            valid = sample is not None and now - sample[0] <= 0.25
            stamps.append(sample[1] if sample else 0)
            if valid:
                flags |= 1 << group
            values.extend(sample[2] if valid else (float("nan"),) * count)
        packet = StatePacket(self.sequence, self.get_clock().now().nanoseconds,
                             *stamps, flags, tuple(values))
        self.sequence = (self.sequence + 1) % 2**64
        try:
            self.sock.sendto(packet.encode(), self.target)
        except OSError as exc:
            self.get_logger().warning(f"State UDP send failed: {exc}", throttle_duration_sec=5.0)

    def destroy_node(self):
        self.sock.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = StateStreamer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
