#include "yin_franka_cartesian_controller/franka_policy_torque_controller.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

#include <franka/model.h>
#include <pluginlib/class_list_macros.hpp>

namespace yin_franka_cartesian_controller {

controller_interface::InterfaceConfiguration
FrankaPolicyTorqueController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto &joint_name : joint_names_) {
    config.names.push_back(joint_name + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
FrankaPolicyTorqueController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto &joint_name : joint_names_) {
    config.names.push_back(joint_name + "/position");
    config.names.push_back(joint_name + "/velocity");
  }
  if (franka_robot_model_) {
    for (const auto &name : franka_robot_model_->get_state_interface_names()) {
      config.names.push_back(name);
    }
  }
  return config;
}

controller_interface::return_type
FrankaPolicyTorqueController::update(const rclcpp::Time &time,
                                     const rclcpp::Duration &period) {
  if (!read_joint_state()) {
    return controller_interface::return_type::ERROR;
  }

  const double dt = std::max(1e-4, period.seconds());
  const auto action = latest_action_or_timeout(time);
  const double action_age =
      action.valid ? (time - action.stamp).seconds() : -1.0;
  const bool command_fresh = action.valid && action_age <= action_timeout_;
  if (!command_fresh && !timed_out_reported_) {
    RCLCPP_WARN(get_node()->get_logger(),
                "Policy action stream timed out; damping joints with zero pose "
                "residual.");
    timed_out_reported_ = true;
  }
  if (command_fresh) {
    timed_out_reported_ = false;
  }

  Vector7d tau = Vector7d::Zero();
  try {
    tau = compute_torque(action, command_fresh);
  } catch (const std::exception &e) {
    RCLCPP_ERROR_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 1000,
        "Policy torque update failed: %s; writing zero command.", e.what());
    write_zero_command();
    return controller_interface::return_type::OK;
  }
  if (!is_finite_vector(tau)) {
    RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(),
                          1000,
                          "Computed non-finite torque; writing zero command.");
    write_zero_command();
    return controller_interface::return_type::OK;
  }

  write_torque_command(tau, dt);
  publish_debug(time, action, command_fresh, action_age, tau);
  return controller_interface::return_type::OK;
}

CallbackReturn FrankaPolicyTorqueController::on_init() {
  try {
    auto_declare<std::string>("arm_id", arm_id_);
    auto_declare<std::string>("command_topic", command_topic_);
    auto_declare<double>("action_timeout", action_timeout_);
    auto_declare<double>("dls_lambda", dls_lambda_);
    auto_declare<double>("max_pos_delta", max_pos_delta_);
    auto_declare<double>("max_ori_delta", max_ori_delta_);
    auto_declare<double>("max_pos_delta_norm", max_pos_delta_norm_);
    auto_declare<double>("max_ori_delta_norm", max_ori_delta_norm_);
    auto_declare<double>("min_kd_multiplier", min_kd_multiplier_);
    auto_declare<double>("max_kd_multiplier", max_kd_multiplier_);
    auto_declare<double>("dq_filter_alpha", dq_filter_alpha_);
    auto_declare<bool>("add_coriolis", add_coriolis_);
    auto_declare<int>("coriolis_update_decimation",
                      coriolis_update_decimation_);
    auto_declare<double>("debug_publish_period", debug_publish_period_);
    auto_declare<std::vector<double>>("tool_offset_from_joint7",
                                      {0.0, 0.0, 0.2104});
    auto_declare<std::vector<double>>(
        "lower_joint_limits",
        {-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159});
    auto_declare<std::vector<double>>(
        "upper_joint_limits",
        {2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159});
    auto_declare<std::vector<double>>(
        "max_delta_q", {0.10, 0.10, 0.10, 0.10, 0.12, 0.12, 0.12});
    auto_declare<std::vector<double>>(
        "max_torque", {18.0, 18.0, 18.0, 18.0, 12.0, 12.0, 12.0});
    auto_declare<std::vector<double>>(
        "max_torque_rate", {80.0, 80.0, 80.0, 80.0, 60.0, 60.0, 60.0});
    auto_declare<std::vector<double>>(
        "min_kp", {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0});
    auto_declare<std::vector<double>>(
        "max_kp", {200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0});
    auto_declare<std::vector<double>>("timeout_damping",
                                      {4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0});
  } catch (const std::exception &e) {
    fprintf(stderr, "Exception in FrankaPolicyTorqueController::on_init: %s\n",
            e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn FrankaPolicyTorqueController::on_configure(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  try {
    arm_id_ = get_node()->get_parameter("arm_id").as_string();
    command_topic_ = get_node()->get_parameter("command_topic").as_string();
    action_timeout_ =
        std::max(0.0, get_node()->get_parameter("action_timeout").as_double());
    dls_lambda_ =
        std::max(1e-8, get_node()->get_parameter("dls_lambda").as_double());
    max_pos_delta_ =
        std::max(0.0, get_node()->get_parameter("max_pos_delta").as_double());
    max_ori_delta_ =
        std::max(0.0, get_node()->get_parameter("max_ori_delta").as_double());
    max_pos_delta_norm_ = std::max(
        0.0, get_node()->get_parameter("max_pos_delta_norm").as_double());
    max_ori_delta_norm_ = std::max(
        0.0, get_node()->get_parameter("max_ori_delta_norm").as_double());
    min_kd_multiplier_ = std::max(
        0.0, get_node()->get_parameter("min_kd_multiplier").as_double());
    max_kd_multiplier_ =
        std::max(min_kd_multiplier_,
                 get_node()->get_parameter("max_kd_multiplier").as_double());
    dq_filter_alpha_ = clamp(
        get_node()->get_parameter("dq_filter_alpha").as_double(), 0.0, 1.0);
    add_coriolis_ = get_node()->get_parameter("add_coriolis").as_bool();
    coriolis_update_decimation_ = std::max(
        1,
        static_cast<int>(
            get_node()->get_parameter("coriolis_update_decimation").as_int()));
    debug_publish_period_ = std::max(
        0.0, get_node()->get_parameter("debug_publish_period").as_double());
    const auto tool_offset =
        get_node()->get_parameter("tool_offset_from_joint7").as_double_array();
    if (tool_offset.size() != 3 ||
        !std::all_of(tool_offset.begin(), tool_offset.end(),
                     [](double value) { return std::isfinite(value); })) {
      throw std::invalid_argument(
          "tool_offset_from_joint7 must contain exactly three finite values");
    }
    tool_offset_from_joint7_ =
        Eigen::Vector3d(tool_offset[0], tool_offset[1], tool_offset[2]);

    for (size_t i = 0; i < kNumJoints; ++i) {
      joint_names_[i] = arm_id_ + "_joint" + std::to_string(i + 1);
    }

    lower_joint_limits_ = load_array_param("lower_joint_limits");
    upper_joint_limits_ = load_array_param("upper_joint_limits");
    max_delta_q_ = load_array_param("max_delta_q");
    max_torque_ = load_array_param("max_torque");
    max_torque_rate_ = load_array_param("max_torque_rate");
    min_kp_ = load_array_param("min_kp");
    max_kp_ = load_array_param("max_kp");
    timeout_damping_ = load_array_param("timeout_damping");

    franka_robot_model_ =
        std::make_unique<franka_semantic_components::FrankaRobotModel>(
            arm_id_ + "/" + robot_model_interface_name_,
            arm_id_ + "/" + robot_state_interface_name_);

    command_sub_ =
        get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
            command_topic_, rclcpp::SystemDefaultsQoS(),
            std::bind(&FrankaPolicyTorqueController::command_callback, this,
                      std::placeholders::_1));
    debug_pub_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
        "~/debug", 10);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Failed to configure policy torque controller: %s", e.what());
    return CallbackReturn::ERROR;
  }

  RCLCPP_INFO(
      get_node()->get_logger(),
      "Configured FrankaPolicyTorqueController arm_id=%s command_topic=%s "
      "timeout=%.3f coriolis_decimation=%d debug_period=%.3f "
      "tool_offset_from_joint7=(%.4f %.4f %.4f) dls_lambda=%.6f",
      arm_id_.c_str(), command_topic_.c_str(), action_timeout_,
      coriolis_update_decimation_, debug_publish_period_,
      tool_offset_from_joint7_.x(), tool_offset_from_joint7_.y(),
      tool_offset_from_joint7_.z(), dls_lambda_);
  return CallbackReturn::SUCCESS;
}

CallbackReturn FrankaPolicyTorqueController::on_activate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  if (!read_joint_state()) {
    RCLCPP_ERROR(
        get_node()->get_logger(),
        "Cannot activate policy torque controller: joint state unavailable.");
    return CallbackReturn::ERROR;
  }
  if (franka_robot_model_) {
    franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  }
  dq_filtered_.setZero();
  target_q_ = q_;
  last_tau_commanded_.setZero();
  last_delta_ee_.setZero();
  last_delta_q_.setZero();
  last_kp_.setZero();
  last_kd_.setZero();
  last_coriolis_.setZero();
  last_jacobian_row_norms_.setZero();
  latest_action_.writeFromNonRT(PolicyAction{});
  timed_out_reported_ = false;
  target_action_sequence_ = 0;
  update_counter_ = 0;
  last_debug_publish_ = get_node()->now();
  command_active_ = true;
  return CallbackReturn::SUCCESS;
}

CallbackReturn FrankaPolicyTorqueController::on_deactivate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  write_zero_command();
  command_active_ = false;
  if (franka_robot_model_) {
    franka_robot_model_->release_interfaces();
  }
  return CallbackReturn::SUCCESS;
}

void FrankaPolicyTorqueController::command_callback(
    const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
  if (msg->data.size() != kActionDim) {
    RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 1000,
        "Ignoring policy action with %zu values; expected %zu.",
        msg->data.size(), kActionDim);
    return;
  }

  PolicyAction action;
  std::copy(msg->data.begin(), msg->data.end(), action.values.begin());
  if (!is_finite_array(action.values)) {
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(),
                         1000, "Ignoring non-finite policy action.");
    return;
  }
  action.stamp = get_node()->now();
  action.sequence = ++command_sequence_counter_;
  action.valid = true;
  latest_action_.writeFromNonRT(action);
}

bool FrankaPolicyTorqueController::read_joint_state() {
  if (state_interfaces_.size() < 2 * kNumJoints) {
    return false;
  }
  for (size_t i = 0; i < kNumJoints; ++i) {
    q_(i) = state_interfaces_[2 * i].get_value();
    dq_(i) = state_interfaces_[2 * i + 1].get_value();
  }
  return q_.allFinite() && dq_.allFinite();
}

FrankaPolicyTorqueController::PolicyAction
FrankaPolicyTorqueController::latest_action_or_timeout(
    const rclcpp::Time & /*time*/) {
  const PolicyAction *latest = latest_action_.readFromRT();
  if (latest == nullptr || !latest->valid || !command_active_) {
    return PolicyAction{};
  }
  return *latest;
}

FrankaPolicyTorqueController::Vector6d
FrankaPolicyTorqueController::clamp_delta_ee(const PolicyAction &action) const {
  Vector6d delta_ee;
  for (int i = 0; i < 3; ++i) {
    delta_ee(i) = clamp(action.values[i], -max_pos_delta_, max_pos_delta_);
    delta_ee(i + 3) =
        clamp(action.values[i + 3], -max_ori_delta_, max_ori_delta_);
  }
  limit_vector_norm(delta_ee, 0, max_pos_delta_norm_);
  limit_vector_norm(delta_ee, 3, max_ori_delta_norm_);
  return delta_ee;
}

void FrankaPolicyTorqueController::refresh_target_q(
    const PolicyAction &action) {
  last_delta_ee_ = clamp_delta_ee(action);
  const Vector7d delta_q = compute_delta_q(last_delta_ee_);
  target_q_ = q_ + delta_q;
  for (size_t i = 0; i < kNumJoints; ++i) {
    target_q_(i) =
        clamp(target_q_(i), lower_joint_limits_[i], upper_joint_limits_[i]);
  }
  target_action_sequence_ = action.sequence;
}

void FrankaPolicyTorqueController::refresh_coriolis() {
  if (!add_coriolis_) {
    last_coriolis_.setZero();
    return;
  }

  ++update_counter_;
  if (update_counter_ == 1 ||
      (update_counter_ % static_cast<uint64_t>(coriolis_update_decimation_) ==
       0)) {
    const auto coriolis_array = franka_robot_model_->getCoriolisForceVector();
    for (size_t i = 0; i < kNumJoints; ++i) {
      last_coriolis_(i) = coriolis_array[i];
    }
  }
}

FrankaPolicyTorqueController::Matrix6x7d
FrankaPolicyTorqueController::compute_tool_zero_jacobian() {
  // Training uses the panda_tool frame.  The deployed URDF's fr3_hand_tcp has
  // the identical fixed transform from link7: 0.107 m to the hand followed by
  // 0.1034 m to the TCP, both along link7 +Z.  libfranka's kEndEffector and
  // kFlange Jacobians are zero on this stack, so shift the valid kJoint7
  // zero-Jacobian to the configured TCP point explicitly.
  const std::array<double, 42> jacobian_array =
      franka_robot_model_->getZeroJacobian(franka::Frame::kJoint7);
  Matrix6x7d jacobian = Eigen::Map<const Matrix6x7d>(jacobian_array.data());

  const std::array<double, 16> pose_array =
      franka_robot_model_->getPoseMatrix(franka::Frame::kJoint7);
  const Eigen::Map<const Eigen::Matrix4d> link7_pose(pose_array.data());
  const Eigen::Vector3d offset_base =
      link7_pose.block<3, 3>(0, 0) * tool_offset_from_joint7_;
  Eigen::Matrix3d offset_cross;
  offset_cross << 0.0, -offset_base.z(), offset_base.y(), offset_base.z(), 0.0,
      -offset_base.x(), -offset_base.y(), offset_base.x(), 0.0;

  // v_tcp = v_joint7 + omega x r = v_joint7 - [r]x omega.
  jacobian.topRows<3>() -= offset_cross * jacobian.bottomRows<3>();
  return jacobian;
}

FrankaPolicyTorqueController::Vector7d
FrankaPolicyTorqueController::compute_delta_q(const Vector6d &delta_ee) {
  if (delta_ee.squaredNorm() <= 1e-18) {
    last_jacobian_row_norms_.setZero();
    return Vector7d::Zero();
  }

  const Matrix6x7d jacobian = compute_tool_zero_jacobian();
  for (int i = 0; i < last_jacobian_row_norms_.size(); ++i) {
    last_jacobian_row_norms_(i) = jacobian.row(i).norm();
  }
  // Match cpos_n JointControlError._lmda, which is 0.01 * I in training.
  const Eigen::Matrix<double, 6, 6> lhs =
      jacobian * jacobian.transpose() +
      dls_lambda_ * Eigen::Matrix<double, 6, 6>::Identity();
  Vector7d delta_q = jacobian.transpose() * lhs.ldlt().solve(delta_ee);

  for (size_t i = 0; i < kNumJoints; ++i) {
    delta_q(i) = clamp(delta_q(i), -max_delta_q_[i], max_delta_q_[i]);
    const double q_des = clamp(q_(i) + delta_q(i), lower_joint_limits_[i],
                               upper_joint_limits_[i]);
    delta_q(i) = q_des - q_(i);
  }
  return delta_q;
}

FrankaPolicyTorqueController::Vector7d
FrankaPolicyTorqueController::compute_torque(const PolicyAction &action,
                                             bool command_fresh) {
  dq_filtered_ =
      (1.0 - dq_filter_alpha_) * dq_filtered_ + dq_filter_alpha_ * dq_;

  Vector7d tau = Vector7d::Zero();
  last_delta_ee_.setZero();
  last_delta_q_.setZero();
  last_kp_.setZero();
  last_kd_.setZero();
  if (command_fresh) {
    if (action.sequence != target_action_sequence_) {
      refresh_target_q(action);
    } else {
      last_delta_ee_ = clamp_delta_ee(action);
    }
    last_delta_q_ = target_q_ - q_;
    for (size_t i = 0; i < kNumJoints; ++i) {
      const double delta_q =
          clamp(last_delta_q_(i), -max_delta_q_[i], max_delta_q_[i]);
      const double kp = clamp(action.values[6 + i], min_kp_[i], max_kp_[i]);
      const double kd_mult =
          clamp(action.values[13 + i], min_kd_multiplier_, max_kd_multiplier_);
      const double kd = kd_mult * std::sqrt(std::max(0.0, kp));
      last_kp_(i) = kp;
      last_kd_(i) = kd;
      tau(i) = kp * delta_q - kd * dq_filtered_(i);
    }
  } else {
    target_q_ = q_;
    target_action_sequence_ = 0;
    for (size_t i = 0; i < kNumJoints; ++i) {
      tau(i) = -timeout_damping_[i] * dq_filtered_(i);
    }
  }

  refresh_coriolis();
  for (size_t i = 0; i < kNumJoints; ++i) {
    tau(i) += last_coriolis_(i);
  }

  for (size_t i = 0; i < kNumJoints; ++i) {
    tau(i) = clamp(tau(i), -max_torque_[i], max_torque_[i]);
  }
  return tau;
}

void FrankaPolicyTorqueController::publish_debug(const rclcpp::Time &time,
                                                 const PolicyAction &action,
                                                 bool command_fresh,
                                                 double action_age,
                                                 const Vector7d &tau) {
  if (!debug_pub_) {
    return;
  }
  if (debug_publish_period_ <= 0.0) {
    return;
  }
  if (debug_pub_->get_subscription_count() == 0 &&
      debug_pub_->get_intra_process_subscription_count() == 0) {
    return;
  }
  if ((time - last_debug_publish_).seconds() < debug_publish_period_) {
    return;
  }
  last_debug_publish_ = time;

  std_msgs::msg::Float64MultiArray msg;
  msg.data.reserve(1 + 1 + 1 + 1 + kActionDim + 3 * kNumJoints + 6 + 7 + 7 + 7 +
                   7 + 6);
  msg.data.push_back(action.valid ? 1.0 : 0.0);
  msg.data.push_back(command_fresh ? 1.0 : 0.0);
  msg.data.push_back(action_age);
  msg.data.push_back(static_cast<double>(action.sequence));
  for (double value : action.values) {
    msg.data.push_back(value);
  }
  for (int i = 0; i < q_.size(); ++i) {
    msg.data.push_back(q_(i));
  }
  for (int i = 0; i < dq_filtered_.size(); ++i) {
    msg.data.push_back(dq_filtered_(i));
  }
  for (int i = 0; i < tau.size(); ++i) {
    msg.data.push_back(tau(i));
  }
  for (int i = 0; i < last_tau_commanded_.size(); ++i) {
    msg.data.push_back(last_tau_commanded_(i));
  }
  for (int i = 0; i < last_delta_ee_.size(); ++i) {
    msg.data.push_back(last_delta_ee_(i));
  }
  for (int i = 0; i < last_delta_q_.size(); ++i) {
    msg.data.push_back(last_delta_q_(i));
  }
  for (int i = 0; i < last_kp_.size(); ++i) {
    msg.data.push_back(last_kp_(i));
  }
  for (int i = 0; i < last_kd_.size(); ++i) {
    msg.data.push_back(last_kd_(i));
  }
  for (int i = 0; i < last_coriolis_.size(); ++i) {
    msg.data.push_back(last_coriolis_(i));
  }
  for (int i = 0; i < last_jacobian_row_norms_.size(); ++i) {
    msg.data.push_back(last_jacobian_row_norms_(i));
  }
  debug_pub_->publish(msg);
}

void FrankaPolicyTorqueController::write_torque_command(const Vector7d &tau,
                                                        double dt) {
  Vector7d limited_tau = tau;
  for (size_t i = 0; i < kNumJoints; ++i) {
    const double max_step = std::max(0.0, max_torque_rate_[i]) * dt;
    limited_tau(i) =
        last_tau_commanded_(i) +
        clamp(tau(i) - last_tau_commanded_(i), -max_step, max_step);
    command_interfaces_[i].set_value(limited_tau(i));
  }
  last_tau_commanded_ = limited_tau;
}

void FrankaPolicyTorqueController::write_zero_command() {
  for (auto &command_interface : command_interfaces_) {
    command_interface.set_value(0.0);
  }
  last_tau_commanded_.setZero();
}

std::array<double, FrankaPolicyTorqueController::kNumJoints>
FrankaPolicyTorqueController::load_array_param(const std::string &name) const {
  const auto values = get_node()->get_parameter(name).as_double_array();
  if (values.size() != kNumJoints) {
    throw std::runtime_error("Parameter '" + name +
                             "' must have exactly 7 values.");
  }
  std::array<double, kNumJoints> out{};
  std::copy(values.begin(), values.end(), out.begin());
  return out;
}

double FrankaPolicyTorqueController::clamp(double value, double min_value,
                                           double max_value) {
  return std::min(std::max(value, min_value), max_value);
}

bool FrankaPolicyTorqueController::is_finite_array(
    const std::array<double, kActionDim> &values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

bool FrankaPolicyTorqueController::is_finite_vector(const Vector7d &values) {
  return values.allFinite();
}

void FrankaPolicyTorqueController::limit_vector_norm(Vector6d &values,
                                                     int offset,
                                                     double max_norm) {
  if (max_norm <= 0.0) {
    return;
  }
  const double norm = values.segment<3>(offset).norm();
  if (norm <= max_norm || norm <= 1e-12) {
    return;
  }
  values.segment<3>(offset) *= max_norm / norm;
}

} // namespace yin_franka_cartesian_controller

PLUGINLIB_EXPORT_CLASS(
    yin_franka_cartesian_controller::FrankaPolicyTorqueController,
    controller_interface::ControllerInterface)
