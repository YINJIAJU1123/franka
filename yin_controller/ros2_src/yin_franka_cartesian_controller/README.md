# yin_franka_cartesian_controller

Minimal ROS 2 overlay for Franka FR3 6D Cartesian teleoperation.

This workspace is the single user-facing entrypoint.
It reuses the already working Franka underlay in `/home/yin/franka_ros2_ws`,
so you do not need to manually source or launch from the old workspace anymore.

## What it provides

- `franka_cartesian_twist_controller`
  - ROS 2 controller plugin
  - consumes `geometry_msgs/msg/TwistStamped`
  - commands Franka `cartesian_velocity` interface directly
- `udp_to_twist_bridge.py`
  - listens for UDP packets on NUC
  - converts packets into `TwistStamped`
- `franka_cartesian_teleop.launch.py`
  - brings up Franka ros2_control
  - spawns the custom controller
  - optionally starts the UDP bridge

## Recommended startup

```bash
/home/yin/Yin_ros2_ws/scripts/run_franka_cartesian_teleop.sh
```

Equivalent manual entrypoint:

```bash
source /home/yin/Yin_ros2_ws/setup_full.bash
ros2 launch yin_franka_cartesian_controller franka_cartesian_teleop.launch.py robot_ip:=172.16.0.2
```

## Fake preview mode

Use this when you want to verify the new Cartesian teleop pipeline before touching the real robot:

```bash
/home/yin/Yin_ros2_ws/scripts/run_franka_cartesian_fake_preview.sh
```

This mode:

- starts Franka with `use_fake_hardware:=true`
- starts RViz
- starts the same UDP bridge
- starts the same Cartesian velocity controller
- publishes a preview TCP frame `fr3_commanded_ee` in RViz

Important limitation:

- this is not a full dynamics simulator
- fake hardware can validate launch, topics, scaling, watchdog and command flow
- the moving preview frame shows the commanded Cartesian target
- it does not prove real force response or real robot tracking quality

## Gazebo sim-only mode

Use this when you want the same `TwistStamped` teleop input, but with a simulated robot that actually moves:

```bash
/home/yin/Yin_ros2_ws/scripts/run_franka_cartesian_gz_sim.sh
```

This mode is intentionally separate from the real robot path:

- real robot:
  - `FrankaCartesianTwistController`
  - writes Franka native `cartesian_velocity`
- Gazebo sim:
  - `FrankaCartesianSimController`
  - reads the same `TwistStamped`
  - solves differential IK in-process
  - writes 7 joint velocity commands, because current `franka_ign_ros2_control` does not expose `cartesian_velocity`

By default the Gazebo launch does not start the UDP bridge.
That keeps sim startup cleaner when you only want to test with `ros2 topic pub`.

If you want the NUC to listen for your PC UDP packets during Gazebo sim:

```bash
/home/yin/Yin_ros2_ws/scripts/run_franka_cartesian_gz_sim.sh start_udp_bridge:=true
```

## What this controller is actually doing

- It does **not** receive 6D pose targets.
- It does **not** call MoveIt IK.
- It does **not** publish `JointTrajectory`.
- It commands Franka's ROS 2 `cartesian_velocity` interface directly.
- The command type is `TwistStamped`: `[vx, vy, vz, wx, wy, wz]`.

That means:

- `joint_states` are only feedback/state topics.
- no teleop command is sent as `joint_state`
- no teleop command is sent as `joint_trajectory`

This is intentional because continuous teleop should use Cartesian velocity, not trajectory points.

## PC-side command contract

Best option:
- send `TwistStamped` in base frame
- units must be SI
- linear: `m/s`
- angular: `rad/s`

Compatible UDP option:
- packet format: little-endian `<6f`
- payload order: `vx vy vz wx wy wz` if `udp_packet_mode:=twist`
- payload order: `dx dy dz droll dpitch dyaw` per packet if `udp_packet_mode:=delta`
- default bridge rate assumption for `delta` mode: `100 Hz`

For minimum PC-side change, keep your current UDP sender and run `udp_packet_mode:=delta`.
For best teleop feel, change the PC side to send `twist` directly.

## Do not run these at the same time

- old MoveIt launch from `franka_ros2_ws`
- old `RLinf/franka_teleop/udp_eef_receiver.py`
- old `libfranka` direct-control programs

Only one Franka control stack should hold FCI at a time.

## Important notes

- The controller expects base-frame Cartesian commands.
- The watchdog zeroes the command if no fresh command arrives within `0.15 s`.
- Velocity and acceleration limits are in `config/controllers.yaml`.
- Firmware 5.9 may still limit what the underlying Franka stack accepts, but this package itself does not add a new firmware dependency.
