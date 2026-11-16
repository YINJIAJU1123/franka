#!/usr/bin/env python3
"""Lixin-side FRST decoder CLI. No ROS installation or robot access required."""
import argparse
import json
import socket
from yin_franka_bringup.state_protocol import decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8896)
    parser.add_argument("--source-host", default="10.10.10.2")
    args = parser.parse_args()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((args.bind_host, args.port))
        while True:
            payload, sender = sock.recvfrom(2048)
            if sender[0] != args.source_host:
                continue
            try:
                packet = decode(payload)
            except ValueError:
                continue
            print(json.dumps({
                "sequence": packet.sequence, "sent_ns": packet.sent_ns,
                "joint_ns": packet.joint_ns, "pose_ns": packet.pose_ns,
                "gripper_ns": packet.gripper_ns, "flags": packet.flags,
                "motor_ns": packet.motor_ns,
                "theta": packet.values[29:36] if packet.flags & 8 else None,
                "dtheta": packet.values[36:43] if packet.flags & 8 else None,
                "q": packet.values[:7] if packet.flags & 1 else None,
                "dq": packet.values[7:14] if packet.flags & 1 else None,
                "tau_J": packet.values[14:21] if packet.flags & 1 else None,
                "eef_xyz_xyzw": packet.values[21:28] if packet.flags & 2 else None,
                "gripper_width": packet.values[28] if packet.flags & 4 else None,
            }, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
