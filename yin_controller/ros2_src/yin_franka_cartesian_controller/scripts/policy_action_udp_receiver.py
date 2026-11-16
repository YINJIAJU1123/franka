#!/usr/bin/env python3
import argparse
import math
import socket
import struct
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class PolicyActionUdpReceiver(Node):
    def __init__(self, args):
        super().__init__("policy_action_udp_receiver")
        self.source_host = args.source_host
        self.publisher = self.create_publisher(Float64MultiArray, args.topic, 10)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", args.port))
        self.sock.setblocking(False)
        self.last_log_time = 0.0
        self.received = 0
        self.timer = self.create_timer(0.001, self.poll_socket)
        self.get_logger().info(
            f"Listening UDP 0.0.0.0:{args.port} for policy action -> {args.topic}"
        )
        self.get_logger().info("Accepted payloads: <20f> or <Id20f> (seq, stamp, action)")

    def poll_socket(self):
        while rclpy.ok():
            try:
                data, addr = self.sock.recvfrom(2048)
            except BlockingIOError:
                return

            if self.source_host and addr[0] != self.source_host:
                continue
            if len(data) == 8 and data[:4] == b"YIN?":
                self.sock.sendto(b"YIN!" + data[4:], addr)
                continue  # Transport-only probe: never publish an action.
            action = None
            if len(data) == struct.calcsize("<20f"):
                action = struct.unpack("<20f", data)
            elif len(data) == struct.calcsize("<Id20f"):
                _seq, _stamp, *action = struct.unpack("<Id20f", data)
            else:
                self.get_logger().warning(
                    f"Ignoring UDP packet from {addr} with invalid size {len(data)}"
                )
                continue

            if not all(math.isfinite(v) for v in action):
                self.get_logger().warning("Ignoring non-finite policy action")
                continue
            msg = Float64MultiArray()
            msg.data = [float(v) for v in action]
            self.publisher.publish(msg)
            self.received += 1

            now = time.monotonic()
            if now - self.last_log_time > 5.0:
                self.get_logger().info(f"Received {self.received} policy action packets")
                self.last_log_time = now


def parse_args():
    parser = argparse.ArgumentParser(description="Forward UDP policy actions to ROS2.")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--source-host", default="", help="Optional sender IP filter")
    parser.add_argument(
        "--topic",
        type=str,
        default="/franka_policy_torque_controller/policy_action",
    )
    return parser.parse_args(rclpy.utilities.remove_ros_args(args=sys.argv)[1:])


def main():
    args = parse_args()
    rclpy.init()
    node = PolicyActionUdpReceiver(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
