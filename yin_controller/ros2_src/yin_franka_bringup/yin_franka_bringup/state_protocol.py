"""FRST v2: one mode-independent little-endian state datagram.

Header: magic, version, validity bits, sequence, send/joints/pose/gripper
timestamps (Unix/ROS nanoseconds). Payload: q[7], dq[7], measured tau_J[7],
O_T_EE xyz+xyzw[7], gripper opening width[1]. SI units throughout.
Payload additionally contains motor-side theta[7]/dtheta[7]. Absent/stale groups have clear validity bits and NaN payloads. Timestamps
refer to individual source samples; the packet is not an atomic sample.
"""
from dataclasses import dataclass
import math
import struct

PACKET = struct.Struct("<4sHHQqqqqq43d")
JOINTS_VALID = 1
POSE_VALID = 2
GRIPPER_VALID = 4
MOTOR_VALID = 8
FIELDS = ((JOINTS_VALID, 0, 21), (POSE_VALID, 21, 28), (GRIPPER_VALID, 28, 29), (MOTOR_VALID, 29, 43))


@dataclass(frozen=True)
class StatePacket:
    sequence: int
    sent_ns: int
    joint_ns: int
    pose_ns: int
    gripper_ns: int
    motor_ns: int
    flags: int
    values: tuple

    def encode(self):
        self.validate()
        return PACKET.pack(b"FRST", 2, self.flags, self.sequence, self.sent_ns,
                           self.joint_ns, self.pose_ns, self.gripper_ns, self.motor_ns, *self.values)

    def validate(self):
        if self.flags & ~15 or len(self.values) != 43:
            raise ValueError("Invalid state flags or field count")
        if not 0 <= self.sequence < 2**64:
            raise ValueError("Invalid sequence")
        for stamp in (self.sent_ns, self.joint_ns, self.pose_ns, self.gripper_ns, self.motor_ns):
            if not 0 <= stamp < 2**63:
                raise ValueError("Invalid timestamp")
        for bit, start, stop in FIELDS:
            group = self.values[start:stop]
            if self.flags & bit:
                if not all(math.isfinite(v) for v in group):
                    raise ValueError("Non-finite valid state")
            elif not all(math.isnan(v) for v in group):
                raise ValueError("Invalid groups must contain NaN")
        if self.flags & POSE_VALID:
            norm = math.sqrt(sum(v * v for v in self.values[24:28]))
            if abs(norm - 1.0) > 1e-3:
                raise ValueError("Invalid quaternion")


def decode(payload):
    if len(payload) != PACKET.size:
        raise ValueError("Invalid state packet size")
    magic, version, flags, sequence, sent, joint, pose, gripper, motor, *values = PACKET.unpack(payload)
    if magic != b"FRST" or version != 2:
        raise ValueError("Unknown state protocol")
    result = StatePacket(sequence, sent, joint, pose, gripper, motor, flags, tuple(values))
    result.validate()
    return result
