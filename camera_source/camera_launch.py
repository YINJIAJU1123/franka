"""
Multi RealSense camera launch for frankaleap + franka compatibility.
# by Jiaju 3.19

Final topic paths (match ACT raw training streams):
  /camera/cam_wrist/color/image_raw   (D435I, native 1280x720@30fps for ACT)
  /camera/cam_head/color/image_raw    (D435I, 外部相机，原生 640x480@30fps)
  /camera/cam_left/color/image_raw    (D435I, 新增左/侧视相机，原生 640x480@30fps)
  /camera/cam_right/color/image_raw   (D435I, 新增右/侧视相机，原生 640x480@30fps)
  /wrist/image_raw                    (relay from cam_wrist, franka compat)
  /head/image_raw                     (relay from cam_head)
  /left/image_raw                     (relay from cam_left)
  /right/image_raw                    (relay from cam_right)

中间 topic (腕部相机原生，一般不需要直接订阅):
  /camera/cam_wrist/color/image_rect_raw

Camera serials (updated 2026-09-02):
  D435I (cam_wrist, 腕部): 135122078001
  D435I (cam_head,  外部): 243722073540

Camera serials (detected 2026-05-28, swapped after visual check):
  D435I (cam_left,  新增): 317422071966
  D435I (cam_right, 新增): 141722078044

D435I already publishes image_raw; the optional wrist resize node is disabled by default.
"""

import copy
import os
import sys

from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

sys.path.append(os.path.join(
    get_package_share_directory('realsense2_camera'), 'launch'
))
import rs_launch

CAMERA_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_CAMERA_DISPLAY = os.path.join(CAMERA_DIR, 'multi_camera_display.py')


def duplicate_params(general_params, suffix):
    local = copy.deepcopy(general_params)
    for p in local:
        p['original_name'] = p['name']
        p['name'] += suffix
    return local


def set_configurable_parameters(params):
    return {p['original_name']: LaunchConfiguration(p['name']) for p in params}


# by Jiaju 3.19: D435I=cam_wrist(腕部), D435I=cam_head(外部)
local_parameters = [
    # Viewer toggles. start_cameras.sh sets these from --display and stream mode.
    {'name': 'display_color', 'default': 'false', 'description': ''},
    {'name': 'display_depth', 'default': 'false', 'description': ''},

    # ── Camera 1: D435I → cam_wrist (腕部相机) ──────────────────────
    # cam_wrist_resize_node.py 负责统一输出 640x480 image_raw
    {'name': 'camera_name1',      'default': 'cam_wrist',      'description': ''},  # by Jiaju 3.19
    {'name': 'camera_namespace1', 'default': 'camera',         'description': ''},
    {'name': 'serial_no1',        'default': "'135122078001'", 'description': ''},
    {'name': 'enable_color1',     'default': 'true',           'description': ''},
    {'name': 'enable_depth1',     'default': 'false',          'description': ''},
    {'name': 'rgb_camera.color_profile1', 'default': '1280,720,30', 'description': ''},
    {'name': 'rgb_camera.color_format1', 'default': 'RGB8', 'description': ''},
    {'name': 'enable_wrist_resize', 'default': 'false', 'description': 'Only enable for a rectified-only source; D435I already publishes image_raw.'},
    {'name': 'config_file1',      'default': "''",             'description': ''},
    # 腕部相机曝光设置（沿用原有配置）。
    {'name': 'depth_module.enable_auto_exposure1',      'default': 'false',  'description': ''},
    {'name': 'depth_module.exposure1',                  'default': '12000',  'description': ''},
    {'name': 'depth_module.gain1',                      'default': '16',     'description': ''},
    {'name': 'depth_module.enable_auto_white_balance1', 'default': 'false',  'description': ''},
    {'name': 'depth_module.white_balance1',             'default': '4600.0', 'description': ''},
    # ── Camera 2: D435I → cam_head (外部相机) ────────────────────────
    # 有独立 RGB 传感器，降低到 15fps 减少 USB 带宽压力 (by Jiaju 8.8)
    {'name': 'camera_name2',                'default': 'cam_head',       'description': ''},
    {'name': 'camera_namespace2',           'default': 'camera',         'description': ''},
    {'name': 'serial_no2',                  'default': "'243722073540'", 'description': ''},
    {'name': 'enable_color2',               'default': 'true',           'description': ''},
    {'name': 'enable_depth2',               'default': 'false',          'description': ''},
    {'name': 'rgb_camera.color_profile2',   'default': '640,480,30',     'description': ''},
    {'name': 'rgb_camera.color_format2',    'default': 'RGB8',           'description': ''},
    {'name': 'depth_module.depth_profile2', 'default': '640,480,30',     'description': ''},
    {'name': 'config_file2',                'default': "''",             'description': ''},

    # ── Camera 3: D435I → cam_left (新增侧视相机，先随便映射，必要时和 cam_right 交换) ──
    {'name': 'camera_name3',                'default': 'cam_left',       'description': ''},
    {'name': 'camera_namespace3',           'default': 'camera',         'description': ''},
    {'name': 'serial_no3',                  'default': "'317422071966'", 'description': ''},
    {'name': 'enable_color3',               'default': 'false',          'description': ''},  # by Jiaju 8.8: 默认禁用，按需启动
    {'name': 'enable_depth3',               'default': 'false',          'description': ''},
    {'name': 'rgb_camera.color_profile3',   'default': '640,480,30',     'description': ''},
    {'name': 'rgb_camera.color_format3',    'default': 'RGB8',           'description': ''},
    {'name': 'depth_module.depth_profile3', 'default': '640,480,30',     'description': ''},
    {'name': 'config_file3',                'default': "''",             'description': ''},

    # ── Camera 4: D435I → cam_right (新增侧视相机，先随便映射，必要时和 cam_left 交换) ──
    {'name': 'camera_name4',                'default': 'cam_right',      'description': ''},
    {'name': 'camera_namespace4',           'default': 'camera',         'description': ''},
    {'name': 'serial_no4',                  'default': "'141722078044'", 'description': ''},
    {'name': 'enable_color4',               'default': 'false',          'description': ''},  # by Jiaju 8.8: 默认禁用，按需启动
    {'name': 'enable_depth4',               'default': 'false',          'description': ''},
    {'name': 'rgb_camera.color_profile4',   'default': '640,480,30',     'description': ''},
    {'name': 'rgb_camera.color_format4',    'default': 'RGB8',           'description': ''},
    {'name': 'depth_module.depth_profile4', 'default': '640,480,30',     'description': ''},
    {'name': 'config_file4',                'default': "''",             'description': ''},
]


def generate_launch_description():
    params1 = duplicate_params(rs_launch.configurable_parameters, '1')
    params2 = duplicate_params(rs_launch.configurable_parameters, '2')
    params3 = duplicate_params(rs_launch.configurable_parameters, '3')
    params4 = duplicate_params(rs_launch.configurable_parameters, '4')

    return LaunchDescription(
        rs_launch.declare_configurable_parameters(local_parameters) +
        rs_launch.declare_configurable_parameters(params1) +
        rs_launch.declare_configurable_parameters(params2) +
        rs_launch.declare_configurable_parameters(params3) +
        rs_launch.declare_configurable_parameters(params4) +
        [
            # ── D435I: cam_wrist ─────────────────────────────────────
            OpaqueFunction(
                function=rs_launch.launch_setup,
                kwargs={
                    'params': set_configurable_parameters(params1),
                    'param_name_suffix': '1',
                }
            ),

            # ── D435I: cam_head (640x480 image_raw) ──────────────────
            # by Jiaju 3.19: delay 3s to let the wrist camera initialize first
            TimerAction(
                period=3.0,
                actions=[
                    OpaqueFunction(
                        function=rs_launch.launch_setup,
                        kwargs={
                            'params': set_configurable_parameters(params2),
                            'param_name_suffix': '2',
                        }
                    ),
                ]
            ),

            # ── D435I: cam_left (640x480 image_raw) ──────────────────
            # by Jiaju 8.8: 仅在 enable_color3=true 时启动，避免占用 USB 带宽
            TimerAction(
                period=6.0,
                actions=[
                    OpaqueFunction(
                        function=rs_launch.launch_setup,
                        kwargs={
                            'params': set_configurable_parameters(params3),
                            'param_name_suffix': '3',
                        }
                    ),
                ],
                condition=IfCondition(LaunchConfiguration('enable_color3')),
            ),

            # ── D435I: cam_right (640x480 image_raw) ─────────────────
            # by Jiaju 8.8: 仅在 enable_color4=true 时启动，避免占用 USB 带宽
            TimerAction(
                period=9.0,
                actions=[
                    OpaqueFunction(
                        function=rs_launch.launch_setup,
                        kwargs={
                            'params': set_configurable_parameters(params4),
                            'param_name_suffix': '4',
                        }
                    ),
                ],
                condition=IfCondition(LaunchConfiguration('enable_color4')),
            ),

            # ── cam_wrist resize: image_rect_raw → image_raw 640x480 ──
            # by Jiaju 3.19
            ExecuteProcess(
                cmd=[
                    '/usr/bin/python3',
                    os.path.join(CAMERA_DIR, 'cam_wrist_resize_node.py'),
                ],
                output='screen',
                name='cam_wrist_resize',
                condition=IfCondition(LaunchConfiguration('enable_wrist_resize')),
            ),

            # ── Relay: franka backward compatibility ──────────────────
            Node(
                package='topic_tools',
                executable='relay',
                name='relay_wrist_compat',
                arguments=[
                    '/camera/cam_wrist/color/image_raw',
                    '/wrist/image_raw',
                ]
            ),
            Node(
                package='topic_tools',
                executable='relay',
                name='relay_head_compat',
                arguments=[
                    '/camera/cam_head/color/image_raw',
                    '/head/image_raw',
                ]
            ),
            Node(
                package='topic_tools',
                executable='relay',
                name='relay_left_compat',
                arguments=[
                    '/camera/cam_left/color/image_raw',
                    '/left/image_raw',
                ]
            ),
            Node(
                package='topic_tools',
                executable='relay',
                name='relay_right_compat',
                arguments=[
                    '/camera/cam_right/color/image_raw',
                    '/right/image_raw',
                ]
            ),

            # ── Optional live viewers (--display): one 2x2 window per stream mode ──
            ExecuteProcess(
                cmd=['/usr/bin/python3', MULTI_CAMERA_DISPLAY, '--mode', 'color'],
                output='screen',
                name='display_all_color',
                condition=IfCondition(LaunchConfiguration('display_color')),
            ),
            ExecuteProcess(
                cmd=['/usr/bin/python3', MULTI_CAMERA_DISPLAY, '--mode', 'depth'],
                output='screen',
                name='display_all_depth',
                condition=IfCondition(LaunchConfiguration('display_depth')),
            ),
        ]
    )
