#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <franka_semantic_components/franka_robot_model.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace yin_franka_cartesian_controller {

class FrankaPolicyTorqueController
    : public controller_interface::ControllerInterface {
public:
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  using Matrix6x7d = Eigen::Matrix<double, 6, 7>;

  [[nodiscard]] controller_interface::InterfaceConfiguration
  command_interface_configuration() const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration
  state_interface_configuration() const override;
  controller_interface::return_type
  update(const rclcpp::Time &time, const rclcpp::Duration &period) override;
  CallbackReturn on_init() override;
  CallbackReturn
  on_configure(const rclcpp_lifecycle::State &previous_state) override;
  CallbackReturn
  on_activate(const rclcpp_lifecycle::State &previous_state) override;
  CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &previous_state) override;

private:
  static constexpr size_t kNumJoints = 7;
  static constexpr size_t kActionDim = 20;

  struct PolicyAction {
    std::array<double, kActionDim> values{};
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
    uint64_t sequence{0};
    bool valid{false};
  };

  void command_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
  bool read_joint_state();
  PolicyAction latest_action_or_timeout(const rclcpp::Time &time);
  Vector6d clamp_delta_ee(const PolicyAction &action) const;
  void refresh_target_q(const PolicyAction &action);
  void refresh_coriolis();
  Matrix6x7d compute_tool_zero_jacobian();
  Vector7d compute_delta_q(const Vector6d &delta_ee);
  Vector7d compute_torque(const PolicyAction &action, bool command_fresh);
  void publish_debug(const rclcpp::Time &time, const PolicyAction &action,
                     bool command_fresh, double action_age,
                     const Vector7d &tau);
  void write_torque_command(const Vector7d &tau, double dt);
  void write_zero_command();

  std::array<double, kNumJoints>
  load_array_param(const std::string &name) const;
  static double clamp(double value, double min_value, double max_value);
  static bool is_finite_array(const std::array<double, kActionDim> &values);
  static bool is_finite_vector(const Vector7d &values);
  static void limit_vector_norm(Vector6d &values, int offset, double max_norm);

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr
      command_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr debug_pub_;
  realtime_tools::RealtimeBuffer<PolicyAction> latest_action_;

  std::unique_ptr<franka_semantic_components::FrankaRobotModel>
      franka_robot_model_;
  const std::string robot_state_interface_name_{"robot_state"};
  const std::string robot_model_interface_name_{"robot_model"};

  std::string arm_id_{"fr3"};
  std::string command_topic_{"~/policy_action"};
  std::array<std::string, kNumJoints> joint_names_{
      {"fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4", "fr3_joint5",
       "fr3_joint6", "fr3_joint7"}};

  Vector7d q_{Vector7d::Zero()};
  Vector7d dq_{Vector7d::Zero()};
  Vector7d dq_filtered_{Vector7d::Zero()};
  Vector7d target_q_{Vector7d::Zero()};
  Vector7d last_tau_commanded_{Vector7d::Zero()};
  Vector6d last_delta_ee_{Vector6d::Zero()};
  Vector7d last_delta_q_{Vector7d::Zero()};
  Vector7d last_kp_{Vector7d::Zero()};
  Vector7d last_kd_{Vector7d::Zero()};
  Vector7d last_coriolis_{Vector7d::Zero()};
  Vector6d last_jacobian_row_norms_{Vector6d::Zero()};
  Eigen::Vector3d tool_offset_from_joint7_{Eigen::Vector3d(0.0, 0.0, 0.2104)};

  std::array<double, kNumJoints> lower_joint_limits_{};
  std::array<double, kNumJoints> upper_joint_limits_{};
  std::array<double, kNumJoints> max_delta_q_{};
  std::array<double, kNumJoints> max_torque_{};
  std::array<double, kNumJoints> max_torque_rate_{};
  std::array<double, kNumJoints> min_kp_{};
  std::array<double, kNumJoints> max_kp_{};
  std::array<double, kNumJoints> timeout_damping_{};

  double action_timeout_{0.20};
  double dls_lambda_{0.01};
  double max_pos_delta_{0.06};
  double max_ori_delta_{0.10};
  double max_pos_delta_norm_{0.08};
  double max_ori_delta_norm_{0.14};
  double min_kd_multiplier_{0.05};
  double max_kd_multiplier_{2.0};
  double dq_filter_alpha_{1.0};
  bool add_coriolis_{true};
  bool command_active_{false};
  bool timed_out_reported_{false};
  uint64_t command_sequence_counter_{0};
  uint64_t target_action_sequence_{0};
  uint64_t update_counter_{0};
  int coriolis_update_decimation_{4};
  double debug_publish_period_{0.25};
  rclcpp::Time last_debug_publish_{0, 0, RCL_ROS_TIME};
};

} // namespace yin_franka_cartesian_controller
