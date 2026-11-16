#!/usr/bin/env bash

# Source from a fresh shell; local_setup avoids recorded old underlays.
YIN_WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ROS setup scripts reference optional variables while nounset may be enabled
# by a caller's safety wrapper.
_YIN_HAD_NOUNSET=0
case "$-" in *u*) _YIN_HAD_NOUNSET=1;; esac
set +u
source /opt/ros/humble/setup.bash
source "${YIN_WS_ROOT}/upstream_description_install/local_setup.bash"
source "${YIN_WS_ROOT}/upstream_ros2_install/local_setup.bash"
export CMAKE_PREFIX_PATH="${YIN_WS_ROOT}/upstream_libfranka_0.20.4_install${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export LD_LIBRARY_PATH="${YIN_WS_ROOT}/upstream_libfranka_0.20.4_install/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -f "${YIN_WS_ROOT}/install/local_setup.bash" ]]; then
  source "${YIN_WS_ROOT}/install/local_setup.bash"
fi
if (( _YIN_HAD_NOUNSET )); then set -u; fi
if [[ "${AMENT_PREFIX_PATH:-}:${CMAKE_PREFIX_PATH:-}:${LD_LIBRARY_PATH:-}" == *"/franka_ros2_ws/"* ]]; then
  echo "ERROR: old franka_ros2_ws is sourced; open a fresh shell before using this workspace." >&2
  return 1
fi
