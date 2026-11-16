"""Decoder for the Yin FRST v2 state datagram."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

FRST_PACKET = struct.Struct("<4sHHQqqqqq43d")
JOINTS_VALID = 1
POSE_VALID = 2
GRIPPER_VALID = 4
MOTOR_VALID = 8
_FIELDS = ((JOINTS_VALID, 0, 21), (POSE_VALID, 21, 28), (GRIPPER_VALID, 28, 29), (MOTOR_VALID, 29, 43))


@dataclass(frozen=True)
class FRSTState:
    sequence: int
    sent_ns: int
    joint_ns: int
    pose_ns: int
    gripper_ns: int
    motor_ns: int
    flags: int
    values: tuple[float, ...]


def decode_frst(payload: bytes) -> FRSTState:
    """Decode and validate one 400-byte FRST v2 datagram."""
    if len(payload) != FRST_PACKET.size:
        raise ValueError(f"invalid FRST packet size {len(payload)}; expected {FRST_PACKET.size}")
    magic, version, flags, sequence, sent, joint, pose, gripper, motor, *raw_values = FRST_PACKET.unpack(payload)
    if magic != b"FRST" or version != 2:
        raise ValueError("unknown FRST packet header")
    if flags & ~15:
        raise ValueError(f"invalid FRST validity flags {flags}")
    if min(sent, joint, pose, gripper, motor) < 0:
        raise ValueError("FRST timestamps must be non-negative")
    values = tuple(float(value) for value in raw_values)
    for bit, start, end in _FIELDS:
        group = values[start:end]
        if flags & bit:
            if not all(math.isfinite(value) for value in group):
                raise ValueError("valid FRST group contains non-finite values")
        elif not all(math.isnan(value) for value in group):
            raise ValueError("invalid FRST group must contain NaN values")
    if flags & POSE_VALID and abs(sum(value * value for value in values[24:28]) - 1.0) > 1e-3:
        raise ValueError("invalid FRST pose quaternion")
    return FRSTState(sequence, sent, joint, pose, gripper, motor, flags, values)
