#!/bin/bash
# 启动指定的 RealSense 相机 (D435I 腕部 + 3x D435I 外部)
# 用法:
#   bash start_cameras.sh --color --head --wrist
#   bash start_cameras.sh --color --depth --head --left
#   bash start_cameras.sh --color --all
# by Jiaju 3.19

set -eo pipefail
set -u

# Keep every camera on the default ROS 2 DDS implementation/transport, while
# restricting discovery and traffic to this host. Clear historical UDP-only
# overrides that may still be present in the interactive shell.
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=1
unset RMW_IMPLEMENTATION
unset FASTDDS_BUILTIN_TRANSPORTS
unset RMW_FASTRTPS_USE_SHM

ENABLE_RGB=false
ENABLE_DEPTH=false
ENABLE_DISPLAY=false
ENABLE_CLOUD=false
HAS_MODE_ARG=false
ENABLE_WRIST=false
ENABLE_HEAD=false
ENABLE_LEFT=false
ENABLE_RIGHT=false
HAS_CAMERA_ARG=false

usage() {
    cat <<'EOF'
Usage: bash start_cameras.sh [OPTIONS]

Options:
  --color        Enable RGB/color streams for selected cameras.
  --rgb, --RGB   Alias for --color.
  --depth        Enable depth streams for selected cameras.
  --wrist        Select D435I wrist camera (cam_wrist).
  --head         Select head/external camera (cam_head).
  --left         Select left camera (cam_left).
  --right        Select right camera (cam_right).
  --all          Explicitly select all four cameras.
  --display      Open one 2x2 live display window with titles and FPS.
  --cloud        Enable PointCloud2 topics for selected cameras; also enables depth.
  -h, --help     Show this help.

Examples:
  bash start_cameras.sh --color --head --wrist
  bash start_cameras.sh --color --depth --head --left
  bash start_cameras.sh --color --head --left --right --display
  bash start_cameras.sh --color --all
  bash start_cameras.sh --color --depth --cloud --wrist
EOF
}

while (($#)); do
    case "$1" in
        --color|--rgb|--RGB)
            ENABLE_RGB=true
            HAS_MODE_ARG=true
            ;;
        --depth)
            ENABLE_DEPTH=true
            HAS_MODE_ARG=true
            ;;
        --display)
            ENABLE_DISPLAY=true
            ;;
        --cloud)
            ENABLE_CLOUD=true
            HAS_MODE_ARG=true
            ;;
        --wrist|--wirst)
            ENABLE_WRIST=true
            HAS_CAMERA_ARG=true
            ;;
        --head)
            ENABLE_HEAD=true
            HAS_CAMERA_ARG=true
            ;;
        --left)
            ENABLE_LEFT=true
            HAS_CAMERA_ARG=true
            ;;
        --right)
            ENABLE_RIGHT=true
            HAS_CAMERA_ARG=true
            ;;
        --all)
            ENABLE_WRIST=true
            ENABLE_HEAD=true
            ENABLE_LEFT=true
            ENABLE_RIGHT=true
            HAS_CAMERA_ARG=true
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

if [[ "${HAS_MODE_ARG}" != true ]]; then
    echo "Error: must specify at least one mode: --color/--rgb/--RGB, --depth, or --cloud" >&2
    usage >&2
    exit 2
fi

if [[ "${HAS_CAMERA_ARG}" != true ]]; then
    echo "Error: must select at least one camera: --wrist, --head, --left, --right, or --all" >&2
    usage >&2
    exit 2
fi

if [[ "${ENABLE_CLOUD}" == true && "${ENABLE_DEPTH}" != true ]]; then
    echo "Info: --cloud requires depth; enabling --depth automatically."
    ENABLE_DEPTH=true
fi

DISPLAY_COLOR=false
DISPLAY_DEPTH=false
if [[ "${ENABLE_DISPLAY}" == true && "${ENABLE_RGB}" == true ]]; then
    DISPLAY_COLOR=true
fi
if [[ "${ENABLE_DISPLAY}" == true && "${ENABLE_DEPTH}" == true ]]; then
    DISPLAY_DEPTH=true
fi

ENABLE_COLOR1=false
ENABLE_COLOR2=false
ENABLE_COLOR3=false
ENABLE_COLOR4=false
ENABLE_DEPTH1=false
ENABLE_DEPTH2=false
ENABLE_DEPTH3=false
ENABLE_DEPTH4=false
ENABLE_CLOUD1=false
ENABLE_CLOUD2=false
ENABLE_CLOUD3=false
ENABLE_CLOUD4=false

if [[ "${ENABLE_WRIST}" == true ]]; then
    ENABLE_COLOR1="${ENABLE_RGB}"
    ENABLE_DEPTH1="${ENABLE_DEPTH}"
    ENABLE_CLOUD1="${ENABLE_CLOUD}"
fi
if [[ "${ENABLE_HEAD}" == true ]]; then
    ENABLE_COLOR2="${ENABLE_RGB}"
    ENABLE_DEPTH2="${ENABLE_DEPTH}"
    ENABLE_CLOUD2="${ENABLE_CLOUD}"
fi
if [[ "${ENABLE_LEFT}" == true ]]; then
    ENABLE_COLOR3="${ENABLE_RGB}"
    ENABLE_DEPTH3="${ENABLE_DEPTH}"
    ENABLE_CLOUD3="${ENABLE_CLOUD}"
fi
if [[ "${ENABLE_RIGHT}" == true ]]; then
    ENABLE_COLOR4="${ENABLE_RGB}"
    ENABLE_DEPTH4="${ENABLE_DEPTH}"
    ENABLE_CLOUD4="${ENABLE_CLOUD}"
fi

set +u
source /opt/ros/humble/setup.bash
set -u

echo "启动选择的 RealSense 相机..."
echo "  RGB/color: ${ENABLE_RGB}"
echo "  depth:     ${ENABLE_DEPTH}"
echo "  display:   ${ENABLE_DISPLAY}"
echo "  cloud:     ${ENABLE_CLOUD}"
echo "  selected:  wrist=${ENABLE_WRIST} head=${ENABLE_HEAD} left=${ENABLE_LEFT} right=${ENABLE_RIGHT}"
echo "  D435I → cam_wrist (腕部): /camera/cam_wrist/color/image_raw  (1280x720@30fps) serial=135122078001"
echo "  D435I → cam_head  (外部): /camera/cam_head/color/image_raw   (640x480@30fps) serial=243722073540"
echo "  D435I → cam_left  (新增): /camera/cam_left/color/image_raw   (640x480@30fps) serial=317422071966"
echo "  D435I → cam_right (新增): /camera/cam_right/color/image_raw  (640x480@30fps) serial=141722078044"
echo "  relay: /wrist/image_raw  /head/image_raw  /left/image_raw  /right/image_raw"
echo ""
echo "提示: 如果相机崩溃 (VIDIOC_QBUF No such device)，等待 3-5 秒让设备自动重连，"
echo "      或重新插拔 USB，然后重启本脚本。"
echo "      如果左右画面反了，交换 cam_left/cam_right 的 serial 即可。"
echo ""

ros2 launch /data/Yin_ros2_ws/src/camera/camera_launch.py \
    enable_color1:="${ENABLE_COLOR1}" \
    enable_color2:="${ENABLE_COLOR2}" \
    enable_color3:="${ENABLE_COLOR3}" \
    enable_color4:="${ENABLE_COLOR4}" \
    enable_depth1:="${ENABLE_DEPTH1}" \
    enable_depth2:="${ENABLE_DEPTH2}" \
    enable_depth3:="${ENABLE_DEPTH3}" \
    enable_depth4:="${ENABLE_DEPTH4}" \
    pointcloud.enable1:="${ENABLE_CLOUD1}" \
    pointcloud.enable2:="${ENABLE_CLOUD2}" \
    pointcloud.enable3:="${ENABLE_CLOUD3}" \
    pointcloud.enable4:="${ENABLE_CLOUD4}" \
    pointcloud.ordered_pc1:="${ENABLE_CLOUD1}" \
    pointcloud.ordered_pc2:="${ENABLE_CLOUD2}" \
    pointcloud.ordered_pc3:="${ENABLE_CLOUD3}" \
    pointcloud.ordered_pc4:="${ENABLE_CLOUD4}" \
    pointcloud.allow_no_texture_points1:="true" \
    pointcloud.allow_no_texture_points2:="true" \
    pointcloud.allow_no_texture_points3:="true" \
    pointcloud.allow_no_texture_points4:="true" \
    display_color:="${DISPLAY_COLOR}" \
    display_depth:="${DISPLAY_DEPTH}"
