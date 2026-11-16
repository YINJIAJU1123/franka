#pragma once

#include <array>
#include <functional>
#include <memory>
#include <string>

#include <controller_interface/controller_interface.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include <realtime_tools/realtime_buffer.hpp>

#include <franka_semantic_components/franka_cartesian_velocity_interface.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace yin_franka_cartesian_controller {

class FrankaCartesianTwistController : public controller_interface::ControllerInterface {
 public:
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  struct TwistCommand {
    std::array<double, 6> values{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
    bool valid{false};
  };

  void command_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
  std::array<double, 6> compute_target_command(const rclcpp::Time& time,
                                               const rclcpp::Duration& period);
  static void load_array_param(const std::shared_ptr<rclcpp_lifecycle::LifecycleNode>& node,
                               const std::string& name,
                               std::array<double, 3>& target);
  static void limit_vector_norm(std::array<double, 6>& command, size_t offset, double max_norm);
  static double apply_deadband(double value, double deadband);
  static double clamp(double value, double min_value, double max_value);

  std::unique_ptr<franka_semantic_components::FrankaCartesianVelocityInterface>
      cartesian_velocity_interface_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr command_sub_;
  realtime_tools::RealtimeBuffer<TwistCommand> latest_command_;

  std::string arm_id_{"fr3"};
  std::string command_topic_{"~/twist_cmd"};
  double command_timeout_{0.15};
  double command_filter_tau_{0.03};
  int fresh_command_required_cycles_{3};
  std::array<double, 3> linear_scale_{{1.0, 1.0, 1.0}};
  std::array<double, 3> angular_scale_{{1.0, 1.0, 1.0}};
  std::array<double, 3> max_linear_velocity_{{0.15, 0.15, 0.15}};
  std::array<double, 3> max_angular_velocity_{{0.80, 0.80, 0.80}};
  double max_linear_speed_norm_{0.0};
  double max_angular_speed_norm_{0.0};
  std::array<double, 3> linear_deadband_{{0.0, 0.0, 0.0}};
  std::array<double, 3> angular_deadband_{{0.0, 0.0, 0.0}};
  std::array<double, 3> linear_accel_limit_{{0.80, 0.80, 0.80}};
  std::array<double, 3> angular_accel_limit_{{3.00, 3.00, 3.00}};
  std::array<double, 3> linear_jerk_limit_{{5.0, 5.0, 5.0}};
  std::array<double, 3> angular_jerk_limit_{{25.0, 25.0, 25.0}};
  std::array<double, 6> filtered_target_{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  std::array<double, 6> filtered_command_{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  std::array<double, 6> filtered_accel_{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  rclcpp::Time last_seen_command_stamp_{0, 0, RCL_ROS_TIME};
  int fresh_command_cycles_{0};
  bool command_armed_{false};
  bool command_active_{false};
};

}  // namespace yin_franka_cartesian_controller
