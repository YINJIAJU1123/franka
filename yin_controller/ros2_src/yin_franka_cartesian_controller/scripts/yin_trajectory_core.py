"""Shared JTRJ wire validation and FR3 time parameterization (no ROS)."""
from __future__ import annotations

from dataclasses import dataclass
import json
import uuid

import numpy as np
from scipy.interpolate import make_interp_spline, PPoly

JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))
# From the FR3 URDF used by this workspace, not the Panda limits.
LOWER = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
UPPER = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
MAX_DATAGRAM = 60000
MAX_POINTS = 256


def encode(message: dict) -> bytes:
    payload = json.dumps(message, separators=(",", ":"), allow_nan=False).encode()
    if len(payload) > MAX_DATAGRAM:
        raise ValueError("JTRJ exceeds one UDP datagram; reduce chunk size")
    return payload


def decode(payload: bytes) -> dict:
    if len(payload) > MAX_DATAGRAM:
        raise ValueError("oversized JTRJ datagram")
    try:
        message = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid JTRJ JSON") from exc
    if not isinstance(message, dict) or message.get("version") != 1:
        raise ValueError("expected JTRJ protocol version 1")
    session, seq = message.get("session"), message.get("seq")
    if not isinstance(session, str) or len(session) != 32 or uuid.UUID(session).hex != session:
        raise ValueError("session must be a UUID hex string")
    if type(seq) is not int or not 0 <= seq < 2**53:
        raise ValueError("seq must be a nonnegative integer below 2**53")
    return message


def validate_points(points) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8 or not 1 <= len(values) <= MAX_POINTS:
        raise ValueError(f"points must have shape [1..{MAX_POINTS},8]")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite trajectory value")
    if np.any(values[:, :7] < LOWER) or np.any(values[:, :7] > UPPER):
        raise ValueError("trajectory exceeds FR3 joint limits")
    if np.any(values[:, 7] < 0) or np.any(values[:, 7] > 1):
        raise ValueError("gripper commands must be in [0,1]")
    return values


def request(session: str, seq: int, points, step_seconds: float, *, execute: bool) -> bytes:
    values = validate_points(points)
    step = float(step_seconds)
    if not np.isfinite(step) or not 1e-3 <= step <= 30:
        raise ValueError("step_seconds must be in [0.001,30]")
    if type(execute) is not bool:
        raise ValueError("execute must be a boolean")
    payload = encode(dict(type="JTRJ", version=1, session=session, seq=seq,
                          points=values.tolist(), step_seconds=step, execute=execute))
    decode(payload)
    return payload


@dataclass
class Plan:
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    gripper: np.ndarray
    time_scale: float
    peaks: dict

    def summary(self) -> dict:
        return dict(points=len(self.gripper), duration_seconds=float(self.times[-1]),
                    entry_seconds=float(self.times[1]), time_scale=float(self.time_scale),
                    step_seconds=float(self.times[2]-self.times[1]) if len(self.times)>2 else None,
                    peaks=self.peaks)


def extrema(spline: PPoly, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate each polynomial at endpoints and its derivative's exact roots."""
    lows, highs = [], []
    for joint in range(7):
        poly = PPoly(spline.c[:, :, joint], spline.x, extrapolate=False).derivative(order)
        roots = poly.derivative().roots(extrapolate=False)
        roots = roots[np.isfinite(roots)]
        values = poly(np.concatenate((spline.x, roots)))
        lows.append(np.min(values)); highs.append(np.max(values))
    return np.asarray(lows), np.asarray(highs)


def plan_trajectory(points, step_seconds, q, dq, *, max_velocity=0.5,
                    max_acceleration=2.0, max_jerk=50.0,
                    max_duration=30.0) -> Plan:
    """Preserve waypoints; lengthen entry and, if necessary, report time scaling.

    SciPy supplies a quintic spline with nonzero interior derivatives. JTC
    reconstructs its segments from positions+velocities+accelerations. Initial
    acceleration and terminal velocity/acceleration are zero. This checks reference derivatives, not collisions.
    """
    points = validate_points(points)
    q, dq = np.asarray(q, float), np.asarray(dq, float)
    if q.shape != (7,) or dq.shape != (7,) or not np.isfinite([q, dq]).all():
        raise ValueError("fresh seven-joint position and velocity are required")
    if np.any(q < LOWER) or np.any(q > UPPER):
        raise ValueError("measured state exceeds FR3 joint limits")
    dt = float(step_seconds)
    limits = np.asarray([max_velocity, max_acceleration, max_jerk, max_duration, dt])
    if not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("trajectory time and limits must be finite and positive")
    # Use measured velocity as the boundary condition; no settled-start test.
    # Time scaling cannot reduce a fixed initial velocity that exceeds the limit.
    if np.max(np.abs(dq)) > max_velocity:
        raise ValueError("measured initial velocity exceeds trajectory velocity limit")
    distance = float(np.max(np.abs(points[0, :7]-q)))
    entry = max(dt, 2*distance/max_velocity, (6*distance/max_acceleration)**0.5,
                (60*distance/max_jerk)**(1/3))
    positions = np.vstack((q, points[:, :7]))
    scale = 1.
    for _ in range(8):
        times = np.r_[0., max(entry, dt*scale) + np.arange(len(points))*dt*scale]
        if times[-1] > max_duration:
            raise ValueError(f"trajectory requires excessive slowdown: scale={scale:.2f}, duration={times[-1]:.2f}s")
        bs = make_interp_spline(times, positions, k=5,
                                bc_type=([(1, dq), (2, np.zeros(7))],
                                         [(1, np.zeros(7)), (2, np.zeros(7))]))
        polynomials = [PPoly.from_spline((bs.t, bs.c[:,j], bs.k)) for j in range(7)]
        spline = PPoly(np.stack([p.c for p in polynomials], axis=-1), polynomials[0].x,
                       extrapolate=False)
        peaks = {}
        for order, name in enumerate(("velocity", "acceleration", "jerk"), 1):
            lo, hi = extrema(spline, order)
            peaks[name] = float(np.max(np.maximum(np.abs(lo), np.abs(hi))))
        factor = max(peaks["velocity"]/max_velocity,
                     (peaks["acceleration"]/max_acceleration)**0.5,
                     (peaks["jerk"]/max_jerk)**(1/3))
        if factor <= 1 + 1e-8:
            lo, hi = extrema(spline, 0)
            if np.any(lo < LOWER-1e-9) or np.any(hi > UPPER+1e-9):
                raise ValueError("interpolated trajectory overshoots FR3 joint limits")
            return Plan(times, positions, spline(times, 1), spline(times, 2), points[:, 7].copy(), scale, peaks)
        scale *= factor * 1.02
        if len(points) == 1:
            entry *= factor * 1.02
    raise ValueError("trajectory time parameterization did not converge")
