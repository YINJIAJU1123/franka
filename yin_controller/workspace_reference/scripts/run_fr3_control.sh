#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ROOT_DIR}/setup_full.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-66}"
MODE="${1:?mode required: cartesian|joint_trajectory|policy_torque|impedance_replay|joint_position_direct}"
shift
child_pid=""

cleanup() {
  trap - INT TERM HUP EXIT
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    # Keep ros2 launch attached to this shell so terminal SIGINT reaches it.
    kill -INT "${child_pid}" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "${child_pid}" 2>/dev/null || return
      sleep 0.1
    done
    kill -TERM "${child_pid}" 2>/dev/null || true
  fi
}

trap cleanup INT TERM HUP EXIT
ros2 launch yin_franka_bringup fr3_control.launch.py mode:="${MODE}" "$@" &
child_pid=$!
wait "${child_pid}"
status=$?
child_pid=""
trap - INT TERM EXIT
exit "${status}"
