#!/bin/bash
# 查看多相机画面
# 用法:
#   bash view_cameras.sh --color
#   bash view_cameras.sh --depth
#   bash view_cameras.sh --color --depth
# 需要先在另一个终端启动相机。
# by Jiaju 3.19

set -eo pipefail

ENABLE_COLOR=false
ENABLE_DEPTH=false

usage() {
    cat <<'EOF'
Usage: bash view_cameras.sh [OPTIONS]

Options:
  --color        Show color streams.
  --rgb, --RGB   Alias for --color.
  --depth        Show depth streams.
  -h, --help     Show this help.

Examples:
  bash view_cameras.sh --color
  bash view_cameras.sh --depth
  bash view_cameras.sh --color --depth
EOF
}

while (($#)); do
    case "$1" in
        --color|--rgb|--RGB)
            ENABLE_COLOR=true
            ;;
        --depth)
            ENABLE_DEPTH=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ "${ENABLE_COLOR}" != true && "${ENABLE_DEPTH}" != true ]]; then
    ENABLE_COLOR=true
fi

set +u
source /opt/ros/humble/setup.bash
set -u

MULTI_CAMERA_DISPLAY=/data/Yin_ros2_ws/src/camera/multi_camera_display.py

echo "等待相机 topic 出现 (最多 20 秒)..."
for i in $(seq 1 20); do
    TOPICS="$(ros2 topic list 2>/dev/null || true)"
    if [[ "${ENABLE_COLOR}" == true ]] && echo "${TOPICS}" | grep -q "/camera/cam_wrist/color/image_raw"; then
        echo "相机 topic 已就绪。"
        break
    fi
    if [[ "${ENABLE_DEPTH}" == true ]] && echo "${TOPICS}" | grep -q "/camera/cam_wrist/depth/image_rect_raw"; then
        echo "相机 topic 已就绪。"
        break
    fi
    sleep 1
done

TOPICS="$(ros2 topic list 2>/dev/null || true)"
if [[ "${ENABLE_COLOR}" == true ]] && ! echo "${TOPICS}" | grep -q "/camera/cam_wrist/color/image_raw"; then
    echo "错误: color topic 未出现，请先运行 bash start_cameras.sh --color"
    exit 1
fi
if [[ "${ENABLE_DEPTH}" == true ]] && ! echo "${TOPICS}" | grep -q "/camera/cam_wrist/depth/image_rect_raw"; then
    echo "错误: depth topic 未出现，请先运行 bash start_cameras.sh --depth"
    exit 1
fi

echo ""
echo "打开画面查看器..."
echo "  color: ${ENABLE_COLOR}"
echo "  depth: ${ENABLE_DEPTH}"
echo ""

PIDS=()
launch_display() {
    local mode="$1"
    echo "  display: ${mode}"
    /usr/bin/python3 "${MULTI_CAMERA_DISPLAY}" --mode "${mode}" &
    PIDS+=("$!")
    sleep 0.5
}

if [[ "${ENABLE_COLOR}" == true ]]; then
    launch_display color
fi

if [[ "${ENABLE_DEPTH}" == true ]]; then
    launch_display depth
fi

wait
