#!/usr/bin/env python3
"""真机 Franka + 平行夹爪 ACT 推理。

This runner is deliberately separate from ``run_act_bc_real.py``.  The
checkpoint predicts absolute 7-joint Franka positions plus a binary gripper
command.  Yin's current Cartesian controller does not accept joint targets,
so each predicted joint target is converted to an EEF pose with Pinocchio and
tracked with a bounded Cartesian velocity delta.  Commands are sent as Yin's
TS-WAM ``<7f>`` UDP packet::

    dx dy dz droll dpitch dyaw gripper

Robot state is received from the local FRST ROS relay; images arrive on
the two camera topics. The relay owns FRST UDP state port 8896.

Controls: B starts/ends an episode, C exits.  Before waiting for B, the runner
automatically streams the task reset pose through Yin's Cartesian controller.
The reset can be disabled with ``--auto-reset 0``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import time
from pathlib import Path

import numpy as np
import rclpy
import torch

from act_gripper_policy import load_act_gripper_policy, preprocess_image
from franka_gripper_transport import FrankaGripperTransport
from franka_gripper_control import TASK_RESET_TARGETS, CartesianServo, stream_task_reset


class KeyboardState:
    def __init__(self) -> None:
        self.start_requested = False
        self.end_requested = False
        self.exit_requested = False
        self.waiting_for_start = False
        self._listener = None

    def start(self) -> None:
        try:
            from pynput import keyboard
        except Exception as exc:
            raise RuntimeError(
                "pynput is required for B/C controls; run from the same tools environment "
                "as the existing ACT real runner"
            ) from exc

        def on_press(key):
            char = getattr(key, "char", None)
            if char is not None:
                char = char.lower()
            if char == "b":
                if self.waiting_for_start:
                    self.start_requested = True
                    print("\n>>> [B] episode start", flush=True)
                else:
                    self.end_requested = True
                    print("\n>>> [B] episode end", flush=True)
            elif char == "c":
                self.exit_requested = True
                print("\n>>> [C] exit", flush=True)

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()


def _load_stats(checkpoint_dir: Path) -> dict:
    path = checkpoint_dir / "dataset_stats.pth"
    if not path.exists():
        return {}
    return torch.load(path, map_location="cpu", weights_only=False)


def _vector_bound(stats: dict, key: str, name: str, dim: int = 8) -> np.ndarray | None:
    value = stats.get(key, {}).get(name) if isinstance(stats.get(key), dict) else None
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return array[:dim] if array.size >= dim else None


def _dataset_start(config: dict) -> tuple[np.ndarray | None, float | None]:
    """Read the first arm/gripper state of each episode for reset diagnostics."""
    root = Path(config.get("data_dir", "")).expanduser()
    if not root.is_absolute():
        root = (Path(__file__).resolve().parent / root).resolve()
    if not root.exists():
        return None, None
    try:
        import pyarrow.parquet as pq

        paths = sorted(root.glob("data/chunk-*/episode_*.parquet"))
        starts = []
        for path in paths:
            table = pq.read_table(path, columns=["observation.state"])
            if table.num_rows <= 0:
                continue
            values = np.asarray(table.column("observation.state")[0].as_py(), dtype=np.float64)
            if values.size >= 8:
                starts.append(values[:8])
        if not starts:
            return None, None
        starts = np.asarray(starts, dtype=np.float64)
        start_q = starts[:, :7].mean(axis=0).astype(np.float32)
        start_gripper = float(np.mean(starts[:, 7]) >= 0.5)
        return start_q, start_gripper
    except Exception as exc:
        print(f"[ACT-Gripper] reset metadata unavailable: {exc}", flush=True)
        return None, None


def _parse_bool_int(value: str | int) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ACT real inference for Franka + parallel gripper")
    parser.add_argument("--act_ckpt_dir", required=True)
    parser.add_argument("--act_ckpt_file", default="")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--allow-motion", type=int, default=0)
    parser.add_argument(
        "--dry-run-steps", type=int, default=0,
        help="0: continuous inference with B/C controls; positive: finite automatic dry-run",
    )
    parser.add_argument("--urdf", default=str(Path(__file__).resolve().parent / "fr3.urdf"))
    parser.add_argument("--eef-frame", default="fr3_link8")
    parser.add_argument("--command-port", type=int, default=8888)
    parser.add_argument("--yin-host", default="10.10.10.2")
    parser.add_argument("--head-topic", default="/camera/cam_head/color/image_raw")
    parser.add_argument("--wrist-topic", default="/camera/cam_wrist/color/image_raw")
    parser.add_argument("--joint-topic", default="/fr3/filtered_joint_states")
    parser.add_argument("--pose-topic", default="/franka_robot_state_broadcaster/current_pose")
    parser.add_argument("--gripper-topic", default="/franka_gripper/joint_states")
    parser.add_argument("--position-kp", type=float, default=1.5)
    parser.add_argument("--orientation-kp", type=float, default=1.5)
    parser.add_argument("--max-linear-velocity", type=float, default=0.12)
    parser.add_argument("--max-angular-velocity", type=float, default=0.35)
    parser.add_argument("--state-timeout", type=float, default=0.35)
    parser.add_argument("--image-timeout", type=float, default=1.0)
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.04)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.055)
    parser.add_argument("--reset-task", choices=tuple(TASK_RESET_TARGETS), default="drawer")
    parser.add_argument("--reset-duration", type=float, default=5.0)
    parser.add_argument("--auto-reset", type=int, default=1)
    parser.add_argument("--gripper-reset", choices=("hold", "open", "dataset"), default="open")
    parser.add_argument("--gripper-settle-s", type=float, default=0.8)
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-hysteresis", type=float, default=0.05)
    parser.add_argument("--clamp-state-to-stats", type=int, default=1)
    parser.add_argument("--clamp-action-to-stats", type=int, default=1)
    parser.add_argument("--strict-reset", type=int, default=0)
    parser.add_argument("--reset-joint-tolerance", type=float, default=0.25)
    parser.add_argument("--log-dir", default="")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.hz <= 0.0:
        raise ValueError("--hz must be positive")
    if args.reset_duration < 0.0:
        raise ValueError("--reset-duration must be non-negative")
    allow_motion = _parse_bool_int(args.allow_motion)
    auto_reset = _parse_bool_int(args.auto_reset)
    dry_run_steps = max(0, int(args.dry_run_steps))
    if allow_motion and dry_run_steps:
        raise ValueError("--dry-run-steps must be 0 when --allow-motion=1")
    automated_dry_run = not allow_motion and dry_run_steps > 0
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is unavailable; select --device cpu explicitly for a dry run")
    device = torch.device(args.device)

    checkpoint_dir = Path(args.act_ckpt_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_file = args.act_ckpt_file or None
    policy, raw_config, image_size = load_act_gripper_policy(
        config_path, checkpoint_dir, device, checkpoint_file
    )
    stats = _load_stats(checkpoint_dir)
    state_min = _vector_bound(stats, "observation.state", "min")
    state_max = _vector_bound(stats, "observation.state", "max")
    action_min = _vector_bound(stats, "action", "min")
    action_max = _vector_bound(stats, "action", "max")
    start_q, start_gripper = _dataset_start(raw_config)

    dataset_cfg = raw_config.get("dataset", {})
    crop_mode = str(dataset_cfg.get("crop_mode", "right"))
    cam_modes = dict(dataset_cfg.get("cam_modes", {}))
    transport: FrankaGripperTransport | None = None
    keyboard = KeyboardState()
    servo = CartesianServo(
        args.urdf,
        eef_frame=args.eef_frame,
        position_kp=args.position_kp,
        orientation_kp=args.orientation_kp,
        max_linear_velocity=args.max_linear_velocity,
        max_angular_velocity=args.max_angular_velocity,
    )
    log_file = None
    writer = None
    final_gripper_cmd = 0.0
    try:
        rclpy.init()
        transport = FrankaGripperTransport(
            yin_host=args.yin_host,
            command_port=args.command_port,
            head_topic=args.head_topic,
            wrist_topic=args.wrist_topic,
            joint_topic=args.joint_topic,
            pose_topic=args.pose_topic,
            gripper_topic=args.gripper_topic,
            gripper_close_threshold=args.gripper_close_threshold,
            gripper_open_threshold=args.gripper_open_threshold,
            enable_commands=allow_motion,
        )
        if allow_motion and not transport.probe_command_receiver(timeout_s=1.0):
            raise RuntimeError(
                "Yin UDP command bridge did not answer YIN?/YIN! on port "
                f"{args.command_port}; start run_franka_cartesian_teleop_safe.sh load_gripper:=true"
            )
        if allow_motion:
            print(">>> Yin UDP command bridge probe succeeded", flush=True)
        else:
            print(">>> dry-run: skipping Yin command bridge probe and all action sends", flush=True)
        if not automated_dry_run:
            keyboard.start()
            print(">>> Keyboard: [B]=start/end episode [C]=exit", flush=True)
        print(
            f">>> ACT-Gripper: state=ROS relay, command=<7f>/UDP:{args.command_port}, "
            f"hz={args.hz:g}, image={image_size}x{image_size}, "
            f"episode_limit={dry_run_steps if automated_dry_run else 'unlimited'}",
            flush=True,
        )
        print(
            ">>> mode=" + ("REAL MOTION" if allow_motion else "DRY RUN (no arm/gripper action packets)"),
            flush=True,
        )
        print(
            f">>> auto reset: task={args.reset_task} duration={args.reset_duration:g}s "
            f"before B={'enabled' if auto_reset else 'disabled'}",
            flush=True,
        )
        if start_q is not None:
            print(f">>> dataset start q mean={np.array2string(start_q, precision=4)}", flush=True)

        if args.log_dir:
            log_dir = Path(args.log_dir).expanduser()
        else:
            log_dir = checkpoint_dir / "act_gripper_real_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"real_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        log_file = log_path.open("w", newline="")
        writer = csv.writer(log_file)
        writer.writerow([
            "wall_time", "episode", "step", "state_seq", "q_target", "q_feedback",
            "gripper_action_raw", "gripper_cmd", "gripper_width", "gripper_feedback",
            "position_error", "rotation_error", "linear_delta", "angular_delta", "command_sent",
        ])
        print(f">>> log: {log_path}", flush=True)

        episode = 0
        previous_gripper_cmd = 0.0
        while not keyboard.exit_requested:
            policy.reset()
            keyboard.start_requested = False
            keyboard.end_requested = False
            keyboard.waiting_for_start = True
            print(f"\n=== Episode {episode} ===", flush=True)
            obs = transport.wait_ready(
                timeout_s=args.ready_timeout, image_timeout_s=args.image_timeout
            )
            if auto_reset:
                stream_task_reset(
                    transport, servo, obs, TASK_RESET_TARGETS[args.reset_task],
                    args.reset_duration, args.hz, allow_motion, args.state_timeout,
                )
                obs = transport.wait_ready(
                    timeout_s=args.ready_timeout, image_timeout_s=args.image_timeout
                )
            if start_q is not None:
                reset_error = np.abs(obs.arm_joints.astype(np.float64) - start_q.astype(np.float64))
                print(
                    f"[RESET] q error norm={np.linalg.norm(reset_error):.4f} "
                    f"max_abs={reset_error.max():.4f} tol={args.reset_joint_tolerance:.4f}",
                    flush=True,
                )
                if args.strict_reset and float(reset_error.max()) > args.reset_joint_tolerance:
                    print("[RESET] strict check failed; reposition arm manually before pressing B", flush=True)
                    while not keyboard.exit_requested:
                        time.sleep(0.2)
                        current = transport.snapshot()
                        if current is not None and start_q is not None:
                            err = np.abs(current.arm_joints.astype(np.float64) - start_q)
                            if float(err.max()) <= args.reset_joint_tolerance:
                                print("[RESET] arm is within strict tolerance", flush=True)
                                break
                        if keyboard.start_requested:
                            keyboard.start_requested = False
                            print("[RESET] B ignored until strict reset passes", flush=True)
                    if keyboard.exit_requested:
                        break

            if args.gripper_reset == "hold":
                previous_gripper_cmd = float(obs.gripper_binary >= 0.5)
            elif args.gripper_reset == "dataset" and start_gripper is not None:
                previous_gripper_cmd = float(start_gripper >= 0.5)
            else:
                previous_gripper_cmd = 0.0
            final_gripper_cmd = previous_gripper_cmd
            if allow_motion:
                transport.send_zero(previous_gripper_cmd, repeats=5)
            if allow_motion and args.gripper_settle_s > 0.0:
                time.sleep(float(args.gripper_settle_s))

            if automated_dry_run:
                keyboard.waiting_for_start = False
                print(f">>> automatic dry-run: evaluating {dry_run_steps} policy steps", flush=True)
            else:
                print("Press B to start; C exits.", flush=True)
                while not keyboard.start_requested and not keyboard.exit_requested:
                    time.sleep(0.05)
                keyboard.waiting_for_start = False
                keyboard.start_requested = False
                if keyboard.exit_requested:
                    break

            step = 0
            last_send = time.monotonic()
            while not keyboard.end_requested and not keyboard.exit_requested:
                tick = time.monotonic()
                obs = transport.snapshot()
                if obs is None or obs.state_age_s > args.state_timeout:
                    raise RuntimeError(
                        f"FRST state stream stale at episode={episode} step={step}; "
                        "stopping policy execution"
                    )
                if not np.isfinite(obs.arm_joints).all() or not np.isfinite(obs.ee_pose).all():
                    raise RuntimeError("FRST state contains invalid arm pose/joints")
                if any(
                    name not in obs.images or obs.image_age_s.get(name, float("inf")) > args.image_timeout
                    for name in ("cam_head", "cam_wrist")
                ):
                    raise RuntimeError("camera stream stale; stopping ACT for safety")

                state = np.concatenate(
                    [obs.arm_joints.astype(np.float32), np.array([obs.gripper_binary], dtype=np.float32)]
                )
                if args.clamp_state_to_stats and state_min is not None and state_max is not None:
                    state = np.clip(state, state_min, state_max)
                policy_obs = {
                    "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
                }
                for camera in ("cam_head", "cam_wrist"):
                    mode = str(cam_modes.get(camera, crop_mode))
                    policy_obs[f"observation.images.{camera}"] = preprocess_image(
                        obs.images[camera], image_size, mode
                    ).to(device)

                with torch.inference_mode():
                    action_tensor = policy.select_action(policy_obs)
                action = action_tensor.squeeze().detach().cpu().numpy().astype(np.float32).reshape(-1)
                if action.size != 8:
                    raise RuntimeError(f"Gripper ACT must output 8 values, got {action.shape}")
                if args.clamp_action_to_stats and action_min is not None and action_max is not None:
                    action = np.clip(action, action_min, action_max)

                raw_gripper = float(action[7])
                threshold = float(args.gripper_threshold)
                hysteresis = max(0.0, float(args.gripper_hysteresis))
                if previous_gripper_cmd >= 0.5:
                    if raw_gripper <= threshold - hysteresis:
                        previous_gripper_cmd = 0.0
                elif raw_gripper >= threshold + hysteresis:
                    previous_gripper_cmd = 1.0
                final_gripper_cmd = previous_gripper_cmd

                now = time.monotonic()
                dt_s = min(max(now - last_send, 0.5 / args.hz), 2.0 / args.hz)
                # Yin's relay current_pose may be a tool TCP pose, while the
                # ACT target is converted through the local FR3 URDF flange
                # frame (fr3_link8). Subtracting those two frames directly
                # creates a large false error even when q_target ~= q_feedback.
                # Use FK(q_feedback) as the control anchor so both poses share
                # exactly the same frame.
                current_fk_position, current_fk_quaternion = servo.fk(obs.arm_joints)
                current_fk_pose = np.concatenate(
                    [current_fk_position, current_fk_quaternion]
                )
                linear_delta, angular_delta, position_error, rotation_error = servo.delta(
                    current_fk_pose, action[:7], dt_s
                )
                if allow_motion:
                    transport.send_delta(linear_delta, angular_delta, previous_gripper_cmd)
                last_send = now

                writer.writerow([
                    dt.datetime.now().isoformat(timespec="milliseconds"), episode, step,
                    obs.sequence, np.array2string(action[:7], precision=6),
                    np.array2string(obs.arm_joints, precision=6), raw_gripper,
                    previous_gripper_cmd, obs.gripper_width, obs.gripper_binary,
                    np.array2string(position_error, precision=6),
                    np.array2string(rotation_error, precision=6),
                    np.array2string(linear_delta, precision=6),
                    np.array2string(angular_delta, precision=6),
                    int(allow_motion),
                ])
                log_file.flush()
                if step == 0 or step % max(1, int(args.hz)) == 0:
                    print(
                        f"[{'AUTO' if allow_motion else 'DRY'}] ep={episode} step={step} "
                        f"pos_err={np.linalg.norm(position_error):.4f}m "
                        f"rot_err={np.linalg.norm(rotation_error):.3f}rad "
                        f"grip={raw_gripper:.3f}->{int(previous_gripper_cmd)}",
                        flush=True,
                    )
                step += 1
                if automated_dry_run and step >= dry_run_steps:
                    break
                time.sleep(max(0.0, 1.0 / args.hz - (time.monotonic() - tick)))

            if allow_motion:
                transport.send_zero(previous_gripper_cmd, repeats=8)
            print(f"Episode {episode} done ({step} steps)", flush=True)
            keyboard.end_requested = False
            episode += 1
            if automated_dry_run:
                print(">>> dry-run completed without sending action packets", flush=True)
                break
    except KeyboardInterrupt:
        print("\nExiting...", flush=True)
    finally:
        keyboard.stop()
        if transport is not None:
            try:
                if allow_motion:
                    transport.send_zero(final_gripper_cmd, repeats=5)
            except Exception:
                pass
            transport.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    main()
