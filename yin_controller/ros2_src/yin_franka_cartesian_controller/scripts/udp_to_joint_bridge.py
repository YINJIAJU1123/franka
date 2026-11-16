#!/usr/bin/env python3
"""JTRJ UDP -> serial FollowJointTrajectory without final arrival checks."""
from __future__ import annotations

import argparse
import socket
import sys
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from yin_trajectory_core import JOINT_NAMES, decode, encode, plan_trajectory, validate_points


def duration(seconds):
    sec, nanosec = divmod(round(float(seconds)*1e9), 1_000_000_000)
    return Duration(sec=sec, nanosec=nanosec)


class Receiver(Node):
    def __init__(self, args):
        super().__init__('tswam_joint_trajectory_udp_receiver')
        self.args = args
        self.arm = ActionClient(self, FollowJointTrajectory, args.action)
        self.state = None
        self.gripper_state = None
        self.gripper_handle = None
        self.gripper_pending = None
        self.gripper_command = None
        self.gripper_busy = False
        self.gripper_types = None
        if args.gripper_enabled:
            from franka_msgs.action import Move, Grasp
            self.gripper_types = (Move, Grasp)
            self.move = ActionClient(self, Move, '/franka_gripper/move')
            self.grasp = ActionClient(self, Grasp, '/franka_gripper/grasp')
            self.create_subscription(JointState, '/franka_gripper/joint_states', self.on_gripper, 10)
        self.create_subscription(JointState, args.state_topic, self.on_state, 20)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((args.bind_host, args.udp_port))
        self.sock.setblocking(False)
        self.records = {}  # one final/current reply per session; bounded, no action replays
        self.active = None
        self.create_timer(0.005, self.poll)
        self.create_timer(0.02, self.monitor)
        self.get_logger().info(f'JTRJ v1 UDP {args.bind_host}:{args.udp_port} -> {args.action}; allow_motion={args.allow_motion}')

    def on_state(self, msg):
        index = {name: i for i, name in enumerate(msg.name)}
        if any(n not in index for n in JOINT_NAMES):
            return
        idx = [index[n] for n in JOINT_NAMES]
        if min(len(msg.position), len(msg.velocity)) <= max(idx):
            return
        q, dq = np.asarray(msg.position)[idx], np.asarray(msg.velocity)[idx]
        if np.isfinite([q, dq]).all():
            self.state = (q, dq, time.monotonic())

    def on_gripper(self, msg):
        values = dict(zip(msg.name, msg.position))
        if all(n in values for n in ('fr3_finger_joint1', 'fr3_finger_joint2')):
            width = values['fr3_finger_joint1'] + values['fr3_finger_joint2']
            if np.isfinite(width):
                self.gripper_state = (int(width <= 0.04), time.monotonic())

    def readiness(self):
        fresh = self.state is not None and time.monotonic()-self.state[2] <= self.args.max_state_age
        gripper = not self.args.gripper_enabled or (
            self.move.server_is_ready() and self.grasp.server_is_ready() and
            self.gripper_state is not None and time.monotonic()-self.gripper_state[1] <= self.args.max_state_age)
        return dict(state_fresh=bool(fresh), controller_ready=self.arm.server_is_ready(),
                    gripper_ready=bool(gripper), gripper_enabled=self.args.gripper_enabled,
                    allow_motion=self.args.allow_motion, busy=self.active is not None)

    def reply(self, packet, sender, status, **fields):
        result = dict(type='JTRJ_ACK', version=1, session=packet['session'], seq=packet['seq'], status=status, **fields)
        try:
            self.sock.sendto(encode(result), sender)
        except OSError as exc:
            self.get_logger().warning(f'UDP reply failed: {exc}')
        return result

    def update(self, status, **fields):
        a = self.active
        result = self.reply(a['packet'], a['sender'], status, **a['summary'], **fields)
        self.records[a['packet']['session']] = result

    def poll(self):
        # Bound work so an incoming burst cannot starve ROS feedback/watchdogs.
        for _ in range(16):
            try:
                payload, sender = self.sock.recvfrom(65535)
            except BlockingIOError:
                return
            if self.args.source_host and sender[0] != self.args.source_host:
                continue
            packet = None
            try:
                packet = decode(payload)
                kind = packet.get('type')
                if kind == 'JTRJ_PING':
                    self.reply(packet, sender, 'READY', **self.readiness())
                    continue
                record = self.records.get(packet['session'])
                if kind == 'JTRJ_STATUS':
                    if record is not None and record['seq'] == packet['seq']:
                        self.sock.sendto(encode(record), sender)
                    else:
                        self.reply(packet, sender, 'UNKNOWN')
                    continue
                if kind == 'JTRJ_CANCEL':
                    if self.active is not None and self.active['key'] == (packet['session'], packet['seq']):
                        self.fail('sender requested cancel', status='CANCELED')
                    else:
                        self.reply(packet, sender, 'UNKNOWN')
                    continue
                if kind != 'JTRJ':
                    raise ValueError('unknown JTRJ command')
                if record is not None and packet['seq'] <= record['seq']:
                    if packet['seq'] == record['seq']:
                        self.sock.sendto(encode(record), sender)
                    else:
                        self.reply(packet, sender, 'REJECTED', reason='stale sequence')
                    continue
                if self.active is not None:
                    raise ValueError('another trajectory is active; preemption is refused')
                if record is None and len(self.records) >= 32:
                    raise ValueError('session capacity reached; restart receiver while idle')
                self.submit(packet, sender)
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                self.get_logger().warning(f'rejected UDP from {sender}: {exc}')
                if packet is not None:
                    result = self.reply(packet, sender, 'REJECTED', reason=str(exc))
                    # Do not overwrite the active request's status on a bad new request.
                    if self.active is None and len(self.records) < 32:
                        self.records[packet['session']] = result

    def submit(self, packet, sender):
        if type(packet.get('execute')) is not bool:
            raise ValueError('execute must be explicitly true or false')
        state = self.readiness()
        if not state['state_fresh']:
            raise ValueError('no fresh position/velocity state')
        points = validate_points(packet['points'])
        step = float(packet['step_seconds'])
        if not np.isfinite(step) or not 0.001 <= step <= 30:
            raise ValueError('step_seconds must be in [0.001,30]')
        plan = plan_trajectory(points, step, *self.state[:2],
                               max_velocity=self.args.max_velocity,
                               max_acceleration=self.args.max_acceleration,
                               max_jerk=self.args.max_jerk)
        summary = plan.summary()
        summary['gripper_enabled'] = self.args.gripper_enabled
        summary['completion_basis'] = 'trajectory_time'
        if not packet['execute']:
            self.records[packet['session']] = self.reply(packet, sender, 'VALIDATED', **summary)
            return  # No action/trajectory/gripper publication in validation mode.
        if not self.args.allow_motion or not state['controller_ready'] or not state['gripper_ready']:
            raise ValueError(f'controller/gripper/motion not ready: {state}')
        if not self.args.gripper_enabled:
            raise ValueError('TS-WAM has gripper actions; launch with load_gripper:=true')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(JOINT_NAMES)
        for t, q, dq, ddq in zip(plan.times, plan.positions, plan.velocities, plan.accelerations):
            goal.trajectory.points.append(JointTrajectoryPoint(
                positions=q.tolist(), velocities=dq.tolist(), accelerations=ddq.tolist(), time_from_start=duration(t)))
        goal.path_tolerance = [JointTolerance(name=n, position=0.10) for n in JOINT_NAMES]
        # -1 erases a tolerance; 0 would inherit the controller's defaults.
        # JTC finishes at the final trajectory time without waiting for the
        # measured position/velocity/acceleration to converge.  Keep action
        # results for timeline sequencing and execution faults, not arrival.
        goal.goal_tolerance = [JointTolerance(name=n, position=-1.0, velocity=-1.0,
                                             acceleration=-1.0) for n in JOINT_NAMES]
        self.active = dict(key=(packet['session'], packet['seq']), packet=packet, sender=sender,
                           plan=plan, summary=summary, handle=None, failed=False, arm_done=False,
                           gripper_index=-1, deadline=time.monotonic()+plan.times[-1]+8.)
        self.gripper_command = self.gripper_state[0]
        self.gripper_pending = self.gripper_command
        self.update('SUBMITTED')
        future = self.arm.send_goal_async(goal, feedback_callback=self.feedback)
        future.add_done_callback(self.accepted)
        self.get_logger().info(f"JTRJ seq={packet['seq']} {summary}")

    def accepted(self, future):
        try:
            handle = future.result()
            if not handle.accepted:
                self.update('REJECTED', reason='JTC rejected goal')
                self.active = None
                return
            self.active['handle'] = handle
            handle.get_result_async().add_done_callback(self.completed)
            if self.active['failed']:
                handle.cancel_goal_async()
            else:
                self.update('ACCEPTED')
        except Exception as exc:
            self.fail(f'goal submission failed: {exc}')

    def feedback(self, message):
        a = self.active
        if a is None or a['failed']:
            return
        t = message.feedback.desired.time_from_start
        elapsed = t.sec + t.nanosec*1e-9
        index = min(len(a['plan'].gripper)-1, int(np.searchsorted(a['plan'].times[1:], elapsed+1e-6, side='right')) - 1)
        if index > a['gripper_index']:
            a['gripper_index'] = index
            self.gripper_pending = int(a['plan'].gripper[index] >= 0.5)
            self.send_gripper()

    def completed(self, future):
        if self.active is None:
            return
        try:
            result = future.result()
            if self.active['failed']:
                if not self.gripper_busy:
                    self.active = None
                else:
                    self.active['arm_done'] = True
                return
            if result.status != GoalStatus.STATUS_SUCCEEDED or result.result.error_code != 0:
                self.update('FAILED', reason=f'JTC status={result.status} code={result.result.error_code}: {result.result.error_string}')
                self.active['failed'] = True
                self.active['arm_done'] = True
                if self.gripper_handle is not None:
                    self.gripper_handle.cancel_goal_async()
                if not self.gripper_busy:
                    self.active = None
                return
            self.active['arm_done'] = True
            self.gripper_pending = int(self.active['plan'].gripper[-1] >= 0.5)
            self.send_gripper()
            self.finish_if_ready()
        except Exception as exc:
            self.fail(f'JTC result failed: {exc}')

    def send_gripper(self):
        if self.active is None or self.active['failed'] or self.gripper_busy or self.gripper_pending == self.gripper_command:
            return
        command = self.gripper_pending
        Move, Grasp = self.gripper_types
        goal = Grasp.Goal() if command else Move.Goal()
        goal.width = 0.0 if command else 0.08
        goal.speed = 0.1
        if command:
            goal.force = 20.0
            goal.epsilon.inner = goal.epsilon.outer = 0.005
        self.gripper_busy = True
        client = self.grasp if command else self.move
        def accepted(future):
            try:
                handle = future.result()
                if not handle.accepted:
                    raise RuntimeError('gripper goal rejected')
                self.gripper_handle = handle
                handle.get_result_async().add_done_callback(done)
                if self.active is None or self.active['failed']:
                    handle.cancel_goal_async()
            except Exception as exc:
                self.gripper_busy = False
                self.fail(str(exc))
        def done(future):
            self.gripper_busy = False
            self.gripper_handle = None
            if self.active is None:
                return
            if self.active['failed']:
                if self.active['arm_done']:
                    self.active = None
                return
            try:
                result = future.result()
                if result.status != GoalStatus.STATUS_SUCCEEDED or not result.result.success:
                    raise RuntimeError('gripper operation failed')
                self.gripper_command = command
                self.send_gripper()
                self.finish_if_ready()
            except Exception as exc:
                self.fail(str(exc))
        client.send_goal_async(goal).add_done_callback(accepted)

    def finish_if_ready(self):
        a = self.active
        if a and not a['failed'] and a['arm_done'] and not self.gripper_busy and self.gripper_pending == self.gripper_command:
            self.update('SUCCEEDED')
            self.active = None

    def fail(self, reason, status='FAILED'):
        a = self.active
        if a is None or a['failed']:
            return
        a['failed'] = True
        self.update(status, reason=reason)
        self.get_logger().error(reason)
        if a['handle'] is not None:
            a['handle'].cancel_goal_async()
        if self.gripper_handle is not None:
            self.gripper_handle.cancel_goal_async()
        if a['arm_done'] and not self.gripper_busy:
            self.active = None

    def monitor(self):
        if self.active is None:
            return
        if time.monotonic() > self.active['deadline']:
            self.fail('trajectory/gripper timed out')
        elif not self.readiness()['state_fresh']:
            self.fail('joint feedback became stale')
        elif self.args.gripper_enabled and not self.readiness()['gripper_ready']:
            self.fail('gripper feedback/server became unavailable')

    def close(self):
        self.fail('receiver shutting down')
        self.sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind-host', default='0.0.0.0')
    parser.add_argument('--udp-port', type=int, default=8894)
    parser.add_argument('--source-host', default='10.10.10.1')
    parser.add_argument('--action', default='/fr3_arm_controller/follow_joint_trajectory')
    parser.add_argument('--state-topic', default='/joint_states')
    parser.add_argument('--gripper-enabled', choices=('true', 'false'), default='false')
    parser.add_argument('--allow-motion', action='store_true')
    parser.add_argument('--max-state-age', type=float, default=0.5)
    parser.add_argument('--max-velocity', type=float, default=0.5)
    parser.add_argument('--max-acceleration', type=float, default=2.0)
    parser.add_argument('--max-jerk', type=float, default=50.0)
    args = parser.parse_args(rclpy.utilities.remove_ros_args(args=sys.argv)[1:])
    args.gripper_enabled = args.gripper_enabled == 'true'
    if not 0 < args.udp_port < 65536:
        parser.error('invalid UDP port')
    if not all(np.isfinite(v) and v>0 for v in (args.max_state_age,args.max_velocity,args.max_acceleration,args.max_jerk)):
        parser.error('limits must be finite and positive')
    rclpy.init()
    node = Receiver(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
