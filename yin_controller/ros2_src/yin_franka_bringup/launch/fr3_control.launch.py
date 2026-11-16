"""Single entry point for the four supported FR3 control modes."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def setup(context):
    mode = LaunchConfiguration("mode").perform(context)
    namespace = LaunchConfiguration("namespace").perform(context)
    arm_id = LaunchConfiguration("arm_id").perform(context)
    package = get_package_share_directory("yin_franka_bringup")
    custom = get_package_share_directory("yin_franka_cartesian_controller")
    impedance = get_package_share_directory("yin_franka_joint_impedance_replay")
    direct_joint = get_package_share_directory("yin_franka_joint_position_controller")
    official = get_package_share_directory("franka_fr3_moveit_config")
    configs = {
        "cartesian": os.path.join(custom, "config", "controllers_safe.yaml"),
        "joint_trajectory": os.path.join(official, "config", "fr3_ros_controllers.yaml"),
        "policy_torque": os.path.join(custom, "config", "controllers_policy_torque_safe.yaml"),
        "impedance_replay": os.path.join(impedance, "config", "controllers.yaml"),
        "joint_position_direct": os.path.join(direct_joint, "config", "controllers.yaml"),
    }
    if LaunchConfiguration("use_fake_hardware").perform(context).lower() == "true":
        configs["joint_position_direct"] = os.path.join(direct_joint, "config", "controllers_fake.yaml")
    controllers = {
        "cartesian": "franka_cartesian_twist_controller",
        "joint_trajectory": "fr3_arm_controller",
        "policy_torque": "franka_policy_torque_controller",
        "impedance_replay": "franka_joint_impedance_replay_controller",
        "joint_position_direct": "joint_position_controller",
    }
    if mode not in configs:
        raise RuntimeError(f"Unsupported FR3 control mode: {mode}")
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("franka_bringup"), "launch", "franka.launch.py")),
        launch_arguments={
            "robot_type": arm_id,
            "arm_prefix": LaunchConfiguration("arm_prefix"),
            "namespace": LaunchConfiguration("namespace"),
            "robot_ip": LaunchConfiguration("robot_ip"),
            "load_gripper": LaunchConfiguration("load_gripper"),
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "fake_sensor_commands": LaunchConfiguration("fake_sensor_commands"),
            "joint_state_rate": "60",
            # The controller manager's JointStateBroadcaster is the sole
            # source for /joint_states. Starting the legacy joint_state_publisher
            # here produces malformed mixed-length messages on hardware.
            "spawn_joint_state_publisher": "false",
            "controllers_yaml": configs[mode],
        }.items(),
    )
    spawner = TimerAction(period=3.0, actions=[Node(
        package="controller_manager", executable="spawner", namespace=namespace,
        arguments=[controllers[mode], "--controller-manager-timeout", "60"], output="screen")])
    state = Node(package="yin_franka_bringup", executable="state_streamer.py",
                 namespace=namespace, output="screen",
                 parameters=[{"target_host": LaunchConfiguration("state_host"),
                              "target_port": LaunchConfiguration("state_port")}])
    receivers = {
        "cartesian": Node(package="yin_franka_cartesian_controller", executable="udp_to_twist_bridge.py",
                           namespace=namespace, output="screen", arguments=["--port", "8888", "--packet-mode", "delta", "--topic", "/franka_cartesian_twist_controller/twist_cmd"]),
        "joint_trajectory": Node(package="yin_franka_cartesian_controller", executable="udp_to_joint_bridge.py",
                                  namespace=namespace, output="screen", arguments=["--udp-port", "8894", "--action", "/fr3_arm_controller/follow_joint_trajectory", "--allow-motion", "--gripper-enabled", LaunchConfiguration("load_gripper")]),
        "policy_torque": Node(package="yin_franka_cartesian_controller", executable="policy_action_udp_receiver.py",
                               namespace=namespace, output="screen", arguments=["--port", "8891", "--topic", "/franka_policy_torque_controller/policy_action"]),
        "impedance_replay": Node(package="yin_franka_joint_impedance_replay", executable="joint_impedance_trajectory_udp_receiver.py",
                                  namespace=namespace, output="screen", arguments=["--udp-port", "8895"]),
        "joint_position_direct": Node(package="yin_franka_joint_position_controller", executable="joint_position_udp_receiver.py",
                                  namespace=namespace, output="screen", arguments=["--port", "8897", "--topic", "/yin_joint_position_controller/target"]),
    }
    return [bringup, state, spawner, receivers[mode]]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="cartesian",
                              description="cartesian, joint_trajectory, policy_torque, impedance_replay, or joint_position_direct"),
        DeclareLaunchArgument("robot_ip", default_value="172.16.0.2"),
        DeclareLaunchArgument("arm_id", default_value="fr3"),
        DeclareLaunchArgument("arm_prefix", default_value=""),
        DeclareLaunchArgument("namespace", default_value=""),
        DeclareLaunchArgument("load_gripper", default_value="false"),
        DeclareLaunchArgument("use_fake_hardware", default_value="false"),
        DeclareLaunchArgument("fake_sensor_commands", default_value="false"),
        DeclareLaunchArgument("state_host", default_value="10.10.10.1"),
        DeclareLaunchArgument("state_port", default_value="8896"),
        OpaqueFunction(function=setup),
    ])
