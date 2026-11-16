#!/usr/bin/env python3
import argparse
from collections import deque
from datetime import datetime
import json
import math
import os
from pathlib import Path
import socket
import struct
import sys
import time
import traceback

import rclpy
from rclpy.action import ActionClient
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

try:
    from franka_msgs.action import Grasp, Move
except ImportError:  # Keep arm-only operation usable without franka_msgs.
    Grasp = None
    Move = None


def decode_eef_packet(data):
    """Decode legacy ``<6f>`` or TS-WAM ``<7f>`` UDP payloads."""

    if len(data) == 24:
        return list(struct.unpack("<6f", data)), None
    if len(data) == 28:
        values = list(struct.unpack("<7f", data))
        return values[:6], float(values[6])
    raise ValueError(f"invalid EEF packet size {len(data)}; expected 24 or 28")


class UdpToTwistBridge(Node):
    def __init__(self, args):
        super().__init__("udp_to_twist_bridge")
        self.packet_mode = args.packet_mode
        self.source_host = args.source_host
        self.expected_rate = max(args.expected_rate, 1.0)
        self.nominal_dt = 1.0 / self.expected_rate
        if args.command_timeout > 0.0:
            self.command_timeout = float(args.command_timeout)
        else:
            self.command_timeout = max(5.0 * self.nominal_dt, 0.30)
        self.linear_ramp_rate = max(float(args.linear_ramp_rate), 0.0)
        self.angular_ramp_rate = max(float(args.angular_ramp_rate), 0.0)
        self.publisher = self.create_publisher(TwistStamped, args.topic, 20)
        self.frame_id = args.frame_id
        self.gripper_state_topic = args.gripper_state_topic
        self.gripper_move_action = args.gripper_move_action
        self.gripper_grasp_action = args.gripper_grasp_action
        self.gripper_open_width = float(args.gripper_open_width)
        self.gripper_closed_width = float(args.gripper_closed_width)
        self.gripper_closed_width_threshold = float(args.gripper_closed_width_threshold)
        self.get_logger().info(f"Gripper feedback threshold: {self.gripper_closed_width_threshold:.4f} m")
        self.gripper_speed = float(args.gripper_speed)
        self.gripper_force = float(args.gripper_force)
        self.gripper_epsilon_inner = float(args.gripper_epsilon_inner)
        self.gripper_epsilon_outer = float(args.gripper_epsilon_outer)
        self.gripper_action_timeout = max(float(args.gripper_action_timeout), 0.0)
        self.gripper_measured_closed = None
        self.gripper_requested_closed = None
        self.gripper_active_target = None
        self.gripper_pending_target = None
        self.gripper_busy = False
        self.gripper_action_start_time = 0.0
        self.gripper_last_attempt_time = 0.0
        self.gripper_move_client = None
        self.gripper_grasp_client = None
        if Move is not None and Grasp is not None:
            self.gripper_move_client = ActionClient(self, Move, self.gripper_move_action)
            self.gripper_grasp_client = ActionClient(self, Grasp, self.gripper_grasp_action)
            self.create_subscription(JointState, self.gripper_state_topic, self._gripper_state_cb, 10)
            if self.gripper_action_timeout > 0.0:
                self.gripper_watchdog_timer = self.create_timer(0.2, self._gripper_watchdog)
        else:
            self.get_logger().warning("franka_msgs actions unavailable; 28-byte gripper fields will be ignored")
        self.last_packet_time = None
        self.last_packet_interval_s = None
        self.max_packet_interval_s = 0.0
        self.latest_twist = [0.0] * 6
        self.delta_queue = deque()
        self.max_pending_packets = 4
        self._timed_out = False
        self.timeout_count = 0
        self._last_backlog_warn_time = 0.0
        self._last_stats_time = 0.0
        self._last_publish_time = None
        self.log_every = max(float(args.log_every), 0.0)
        self.packet_count = 0
        self.invalid_packet_count = 0
        self.backlog_drop_count = 0
        self.ramp_limited_count = 0
        self.published_count = 0
        self.zero_publish_count = 0
        self.last_packet_addr = None
        self.last_packet_values = None
        self.last_target_twist = [0.0] * 6
        self.last_published_twist = [0.0] * 6
        self.log_file_path = self._resolve_log_file(args)
        self.log_file = None
        if self.log_file_path:
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_file_path.open("a", buffering=1)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", args.port))
        self.sock.setblocking(False)

        self.poll_timer = self.create_timer(min(self.nominal_dt / 2.0, 0.001), self.poll_socket)
        self.publish_timer = self.create_timer(self.nominal_dt, self.publish_command)
        self.get_logger().info(
            f"UDP bridge listening on 0.0.0.0:{args.port} mode={self.packet_mode} -> {args.topic}"
        )
        self.get_logger().info(
            f"Publishing at {self.expected_rate:.1f} Hz with command timeout {self.command_timeout:.3f}s"
        )
        self.get_logger().info(
            "Output ramp limits: "
            f"linear={self.linear_ramp_rate:.3f} m/s^2 angular={self.angular_ramp_rate:.3f} rad/s^2"
        )
        self.get_logger().info(
            "TS-WAM packet extension: 24-byte <6f> arm packets remain valid; 28-byte <7f> packets add gripper"
        )
        if self.packet_mode == "delta":
            self.get_logger().info(
                "Delta mode: convert each packet using its inter-arrival period "
                "and hold the resulting twist until the next packet"
            )
        if self.log_file_path:
            self.get_logger().info(f"Writing UDP bridge run log to {self.log_file_path}")
        self._log_event(
            "start",
            port=args.port,
            packet_mode=self.packet_mode,
            expected_rate=self.expected_rate,
            nominal_dt=self.nominal_dt,
            command_timeout=self.command_timeout,
            linear_ramp_rate=self.linear_ramp_rate,
            angular_ramp_rate=self.angular_ramp_rate,
            topic=args.topic,
            frame_id=self.frame_id,
            argv=sys.argv,
            pid=os.getpid(),
        )

    def _gripper_state_cb(self, msg):
        values = {str(name): float(msg.position[index]) for index, name in enumerate(msg.name) if index < len(msg.position)}
        if "gripper" in values:
            self.gripper_measured_closed = values["gripper"] >= 0.5
        elif "fr3_finger_joint1" in values and "fr3_finger_joint2" in values:
            self.gripper_measured_closed = (
                values["fr3_finger_joint1"] + values["fr3_finger_joint2"] <= self.gripper_closed_width_threshold
            )
        elif len(msg.position) >= 2:
            # Upstream franka_gripper versions may publish unnamed finger
            # joints. Their first two positions are still the finger widths.
            self.gripper_measured_closed = float(msg.position[0]) + float(msg.position[1]) <= self.gripper_closed_width_threshold

    def _gripper_watchdog(self):
        if not self.gripper_busy or self.gripper_action_start_time <= 0.0:
            return
        elapsed = time.monotonic() - self.gripper_action_start_time
        if elapsed <= self.gripper_action_timeout:
            return
        self.get_logger().warning(f"gripper action timed out after {elapsed:.2f}s; clearing action state")
        self._log_event("gripper_timeout", elapsed=elapsed, target=self.gripper_active_target)
        self.gripper_busy = False
        self.gripper_active_target = None
        self.gripper_action_start_time = 0.0
        pending = self.gripper_pending_target
        self.gripper_pending_target = None
        if pending is not None and pending != self.gripper_measured_closed:
            self._start_gripper_action(pending)

    def _handle_gripper_command(self, value):
        if value is None:
            return
        try:
            requested = bool(float(value) >= 0.5)
        except (TypeError, ValueError):
            return
        if self.gripper_move_client is None or self.gripper_grasp_client is None:
            return
        if self.gripper_requested_closed is requested and self.gripper_busy:
            return
        self.gripper_requested_closed = requested
        if self.gripper_measured_closed is None:
            self.get_logger().warning(f"gripper command received before gripper state; holding request={requested}")
            return
        if requested == self.gripper_measured_closed or requested == self.gripper_active_target:
            return
        if self.gripper_busy:
            self.gripper_pending_target = requested
            return
        if time.monotonic() - self.gripper_last_attempt_time < 0.5:
            return
        self._start_gripper_action(requested)

    def _start_gripper_action(self, target_closed):
        client = self.gripper_grasp_client if target_closed else self.gripper_move_client
        action_name = self.gripper_grasp_action if target_closed else self.gripper_move_action
        if not client.server_is_ready():
            self.get_logger().warning(f"gripper action server is not ready: {action_name}")
            self.gripper_last_attempt_time = time.monotonic()
            return
        if target_closed:
            goal = Grasp.Goal()
            goal.width = self.gripper_closed_width
            goal.epsilon.inner = self.gripper_epsilon_inner
            goal.epsilon.outer = self.gripper_epsilon_outer
            goal.speed = self.gripper_speed
            goal.force = self.gripper_force
        else:
            goal = Move.Goal()
            goal.width = self.gripper_open_width
            goal.speed = self.gripper_speed
        self.gripper_busy = True
        self.gripper_active_target = bool(target_closed)
        self.gripper_action_start_time = time.monotonic()
        self.gripper_last_attempt_time = self.gripper_action_start_time
        self.get_logger().info(f"starting gripper {action_name} action target_closed={target_closed}")
        future = client.send_goal_async(goal)
        future.add_done_callback(lambda result_future: self._gripper_goal_response(result_future, bool(target_closed)))

    def _gripper_goal_response(self, future, target_closed):
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"gripper goal request failed: {exc}")
            self.gripper_busy = False
            self.gripper_active_target = None
            return
        if not handle.accepted:
            self.get_logger().warning(f"gripper goal rejected target_closed={target_closed}")
            self.gripper_busy = False
            self.gripper_active_target = None
            return
        result_future = handle.get_result_async()
        result_future.add_done_callback(lambda result: self._gripper_result(result, target_closed))

    def _gripper_result(self, future, target_closed):
        try:
            wrapped = future.result()
            status = int(getattr(wrapped, "status", 0))
        except Exception as exc:
            self.get_logger().error(f"gripper action result failed: {exc}")
            status = 0
        self.get_logger().info(f"gripper action finished target_closed={target_closed} status={status}")
        self.gripper_busy = False
        self.gripper_active_target = None
        self.gripper_action_start_time = 0.0
        if status == 4:  # GoalStatus.STATUS_SUCCEEDED
            self.gripper_measured_closed = target_closed
        pending = self.gripper_pending_target
        self.gripper_pending_target = None
        if pending is not None and pending != self.gripper_measured_closed:
            self._start_gripper_action(pending)

    def poll_socket(self):
        try:
            self._poll_socket()
        except Exception as exc:
            self._log_event("exception", where="poll_socket", error=repr(exc), traceback=traceback.format_exc())
            self.get_logger().error(f"Exception in UDP poll_socket: {exc}")
            raise

    def _poll_socket(self):
        while rclpy.ok():
            try:
                data, addr = self.sock.recvfrom(1024)
            except BlockingIOError:
                return

            if self.source_host and addr[0] != self.source_host:
                continue
            if len(data) == 8 and data[:4] == b"YIN?":
                self.sock.sendto(b"YIN!" + data[4:], addr)
                continue  # Transport-only probe: never publish an action.
            if len(data) not in (24, 28):
                self.invalid_packet_count += 1
                self.get_logger().warning(f"Ignoring UDP packet with invalid size: {len(data)}")
                self._log_event(
                    "invalid_packet",
                    addr=addr,
                    size=len(data),
                    expected_size=[24, 28],
                    total_invalid=self.invalid_packet_count,
                )
                continue

            values, gripper_value = decode_eef_packet(data)
            if not all(math.isfinite(v) for v in values):
                self.invalid_packet_count += 1
                continue
            if gripper_value is not None and not (-float("inf") < gripper_value < float("inf")):
                self.invalid_packet_count += 1
                self.get_logger().warning("Ignoring TS-WAM packet with non-finite gripper command")
                continue
            now = time.monotonic()
            packet_interval_s = (
                None if self.last_packet_time is None else now - self.last_packet_time
            )
            self.packet_count += 1
            self.last_packet_addr = addr
            self.last_packet_values = values if gripper_value is None else [*values, gripper_value]
            self._handle_gripper_command(gripper_value)
            if self.packet_mode == "delta":
                if len(self.delta_queue) >= self.max_pending_packets:
                    self.delta_queue.popleft()
                    self.backlog_drop_count += 1
                    now = time.monotonic()
                    if now - self._last_backlog_warn_time > 2.0:
                        self.get_logger().warning(
                            f"UDP delta backlog exceeded {self.max_pending_packets}; dropping oldest packet"
                        )
                        self._last_backlog_warn_time = now
                    self._log_event(
                        "backlog_drop",
                        queue_len=len(self.delta_queue),
                        max_pending_packets=self.max_pending_packets,
                        total_backlog_drops=self.backlog_drop_count,
                    )
                # Keep the arrival interval with the delta.  ACT runs at about
                # 15 Hz while the bridge may publish at 40 Hz; dividing every
                # delta by the bridge period would overestimate its velocity.
                self.delta_queue.append((values, packet_interval_s))
            else:
                self.latest_twist = values

            if self.last_packet_time is not None:
                self.last_packet_interval_s = now - self.last_packet_time
                self.max_packet_interval_s = max(self.max_packet_interval_s, self.last_packet_interval_s)
            self.last_packet_time = now
            self._timed_out = False

    def publish_command(self):
        try:
            self._publish_command()
        except Exception as exc:
            self._log_event("exception", where="publish_command", error=repr(exc), traceback=traceback.format_exc())
            self.get_logger().error(f"Exception in UDP publish_command: {exc}")
            raise

    def _publish_command(self):
        now = time.monotonic()
        target_twist = self._target_twist(now)
        twist_values = self._ramp_twist(target_twist, now)

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.twist.linear.x = float(twist_values[0])
        msg.twist.linear.y = float(twist_values[1])
        msg.twist.linear.z = float(twist_values[2])
        msg.twist.angular.x = float(twist_values[3])
        msg.twist.angular.y = float(twist_values[4])
        msg.twist.angular.z = float(twist_values[5])
        self.publisher.publish(msg)
        self.published_count += 1
        self.last_target_twist = list(target_twist)
        self.last_published_twist = list(twist_values)
        if all(abs(value) <= 1e-12 for value in twist_values):
            self.zero_publish_count += 1
        self._maybe_log_stats()

    def _target_twist(self, now):
        if self.last_packet_time is None:
            return [0.0] * 6

        age = now - self.last_packet_time
        if age > self.command_timeout:
            if self.delta_queue:
                self.delta_queue.clear()
            if not self._timed_out:
                self.timeout_count += 1
                self.get_logger().warning("UDP command stream timed out; ramping to zero twist")
                self._log_event(
                    "timeout",
                    age=age,
                    command_timeout=self.command_timeout,
                    packet_count=self.packet_count,
                    published_count=self.published_count,
                    timeout_count=self.timeout_count,
                    last_published_twist=self.last_published_twist,
                )
                self._timed_out = True
            return [0.0] * 6

        if self.packet_mode == "delta":
            if self.delta_queue:
                delta_values, packet_interval_s = self.delta_queue.popleft()
                if packet_interval_s is None or packet_interval_s <= 0.0:
                    delta_dt = self.nominal_dt
                else:
                    # Avoid extreme velocities from a burst while allowing
                    # lower-rate ACT streams to retain their intended delta.
                    delta_dt = min(
                        max(packet_interval_s, 0.5 * self.nominal_dt),
                        self.command_timeout,
                    )
                self.latest_twist = [value / delta_dt for value in delta_values]
            # A delta is an increment over the interval until the next packet;
            # hold its velocity between bridge publish ticks. The timeout
            # branch above still ramps to zero when the stream really stops.
            return list(self.latest_twist)

        return list(self.latest_twist)

    def _ramp_twist(self, target_twist, now):
        if self._last_publish_time is None:
            self._last_publish_time = now
            return list(target_twist)

        dt = now - self._last_publish_time
        self._last_publish_time = now
        dt = min(max(dt, 0.5 * self.nominal_dt), 2.0 * self.nominal_dt)

        if self.linear_ramp_rate <= 0.0 and self.angular_ramp_rate <= 0.0:
            return list(target_twist)

        output = list(self.last_published_twist)
        ramp_limited = False
        for index in range(6):
            rate = self.linear_ramp_rate if index < 3 else self.angular_ramp_rate
            if rate <= 0.0:
                output[index] = float(target_twist[index])
                continue

            max_step = rate * dt
            error = float(target_twist[index]) - output[index]
            if error > max_step:
                output[index] += max_step
                ramp_limited = True
            elif error < -max_step:
                output[index] -= max_step
                ramp_limited = True
            else:
                output[index] = float(target_twist[index])

        if ramp_limited:
            self.ramp_limited_count += 1
        return output

    def _resolve_log_file(self, args):
        if args.run_log_file:
            return Path(args.run_log_file).expanduser()
        if args.run_log_dir:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return Path(args.run_log_dir).expanduser() / f"udp_to_twist_bridge_{stamp}_pid{os.getpid()}.jsonl"
        return None

    def _json_safe(self, value):
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        return value

    def _log_event(self, event, **fields):
        if not self.log_file:
            return
        record = {
            "t_wall": time.time(),
            "iso": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
        }
        record.update({key: self._json_safe(value) for key, value in fields.items()})
        self.log_file.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _maybe_log_stats(self):
        if self.log_every <= 0.0:
            return
        now = time.monotonic()
        if now - self._last_stats_time < self.log_every:
            return
        self._last_stats_time = now
        age = None if self.last_packet_time is None else now - self.last_packet_time
        self._log_event(
            "stats",
            packet_count=self.packet_count,
            invalid_packet_count=self.invalid_packet_count,
            backlog_drop_count=self.backlog_drop_count,
            ramp_limited_count=self.ramp_limited_count,
            published_count=self.published_count,
            zero_publish_count=self.zero_publish_count,
            queue_len=len(self.delta_queue),
            timed_out=self._timed_out,
            timeout_count=self.timeout_count,
            last_packet_age_s=age,
            last_packet_interval_s=self.last_packet_interval_s,
            max_packet_interval_s=self.max_packet_interval_s,
            last_packet_addr=self.last_packet_addr,
            last_packet_values=self.last_packet_values,
            last_target_twist=self.last_target_twist,
            last_published_twist=self.last_published_twist,
        )

    def destroy_node(self):
        self._log_event(
            "shutdown",
            packet_count=self.packet_count,
            invalid_packet_count=self.invalid_packet_count,
            backlog_drop_count=self.backlog_drop_count,
            ramp_limited_count=self.ramp_limited_count,
            published_count=self.published_count,
            zero_publish_count=self.zero_publish_count,
            timeout_count=self.timeout_count,
            max_packet_interval_s=self.max_packet_interval_s,
        )
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        super().destroy_node()


def parse_args():
    parser = argparse.ArgumentParser(description="Convert UDP EEF delta packets to TwistStamped and optional gripper actions.")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--source-host", default="", help="Optional sender IP filter")
    parser.add_argument("--packet-mode", choices=["delta", "twist"], default="delta")
    parser.add_argument(
        "--topic", type=str, default="/franka_cartesian_twist_controller/twist_cmd"
    )
    parser.add_argument("--frame-id", type=str, default="fr3_link0")
    parser.add_argument("--gripper-state-topic", type=str, default="/franka_gripper/joint_states")
    parser.add_argument("--gripper-move-action", type=str, default="/franka_gripper/move")
    parser.add_argument("--gripper-grasp-action", type=str, default="/franka_gripper/grasp")
    parser.add_argument("--gripper-open-width", type=float, default=0.08)
    parser.add_argument("--gripper-closed-width", type=float, default=0.0)
    parser.add_argument("--gripper-speed", type=float, default=0.5)
    parser.add_argument("--gripper-force", type=float, default=20.0)
    parser.add_argument("--gripper-closed-width-threshold", type=float, default=float(os.environ.get("FRANKA_GRIPPER_CLOSED_WIDTH_THRESHOLD", "0.04")))
    parser.add_argument("--gripper-epsilon-inner", type=float, default=0.005)
    parser.add_argument("--gripper-epsilon-outer", type=float, default=0.08)
    parser.add_argument("--gripper-action-timeout", type=float, default=8.0)
    parser.add_argument("--expected-rate", type=float, default=100.0)
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=0.0,
        help="Seconds before the UDP stream is treated as stale. Default: max(5/rate, 0.30).",
    )
    parser.add_argument(
        "--linear-ramp-rate",
        type=float,
        default=0.45,
        help="Max linear command change rate in m/s^2 before publishing.",
    )
    parser.add_argument(
        "--angular-ramp-rate",
        type=float,
        default=1.40,
        help="Max angular command change rate in rad/s^2 before publishing.",
    )
    parser.add_argument("--run-log-dir", type=str, default="")
    parser.add_argument("--run-log-file", type=str, default="")
    parser.add_argument("--log-every", type=float, default=1.0)
    args = parser.parse_args(rclpy.utilities.remove_ros_args(args=sys.argv)[1:])
    if not math.isfinite(args.gripper_closed_width_threshold) or not 0 <= args.gripper_closed_width_threshold <= args.gripper_open_width:
        parser.error("gripper closed width threshold must be finite and within [0, open width]")
    return args


def main():
    args = parse_args()
    rclpy.init()
    node = UdpToTwistBridge(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            # Launch may already have shutdown the context when the system is torn down
            # (e.g. if ros2_control_node crashes). Don't spam traceback.
            pass


if __name__ == "__main__":
    main()
