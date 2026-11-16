#include "yin_franka_cartesian_controller/franka_cartesian_twist_controller.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <pluginlib/class_list_macros.hpp>

namespace yin_franka_cartesian_controller {

controller_interface::InterfaceConfiguration
FrankaCartesianTwistController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = cartesian_velocity_interface_->get_command_interface_names();
  return config;
}

controller_interface::InterfaceConfiguration
FrankaCartesianTwistController::state_interface_configuration() const {
  return controller_interface::InterfaceConfiguration{
      controller_interface::interface_configuration_type::NONE};
}

controller_interface::return_type FrankaCartesianTwistController::update(
    const rclcpp::Time& time,
    const rclcpp::Duration& period) {
  const auto target = compute_target_command(time, period);
  const double dt = std::max(1e-4, period.seconds());

  auto step_velocity = [&](size_t idx,
                           double v_target,
                           double accel_limit,
                           double jerk_limit) {
    const double v_prev = filtered_command_[idx];
    const double a_prev = filtered_accel_[idx];
    const double v_err = v_target - v_prev;

    double a_target = clamp(v_err / dt, -accel_limit, accel_limit);

    double a_cmd = a_target;
    if (jerk_limit > 0.0) {
      const double max_da = jerk_limit * dt;
      a_cmd = a_prev + clamp(a_target - a_prev, -max_da, max_da);
    }

    double v_next = v_prev + a_cmd * dt;

    if ((v_err > 0.0 && v_next > v_target) || (v_err < 0.0 && v_next < v_target)) {
      v_next = v_target;
      a_cmd = (v_next - v_prev) / dt;
    }

    filtered_command_[idx] = v_next;
    filtered_accel_[idx] = a_cmd;
  };

  for (size_t i = 0; i < 3; ++i) {
    step_velocity(i, target[i], linear_accel_limit_[i], linear_jerk_limit_[i]);
  }
  for (size_t i = 0; i < 3; ++i) {
    step_velocity(i + 3, target[i + 3], angular_accel_limit_[i], angular_jerk_limit_[i]);
  }

  Eigen::Vector3d linear(filtered_command_[0], filtered_command_[1], filtered_command_[2]);
  Eigen::Vector3d angular(filtered_command_[3], filtered_command_[4], filtered_command_[5]);
  if (cartesian_velocity_interface_->setCommand(linear, angular)) {
    return controller_interface::return_type::OK;
  }

  RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                        "Failed to write cartesian velocity command.");
  return controller_interface::return_type::ERROR;
}

CallbackReturn FrankaCartesianTwistController::on_init() {
  auto_declare<std::string>("arm_id", arm_id_);
  auto_declare<std::string>("command_topic", command_topic_);
  auto_declare<double>("command_timeout", command_timeout_);
  auto_declare<double>("command_filter_tau", command_filter_tau_);
  auto_declare<int>("fresh_command_required_cycles", fresh_command_required_cycles_);
  auto_declare<std::vector<double>>("linear_scale", {1.0, 1.0, 1.0});
  auto_declare<std::vector<double>>("angular_scale", {1.0, 1.0, 1.0});
  auto_declare<std::vector<double>>("max_linear_velocity", {0.15, 0.15, 0.15});
  auto_declare<std::vector<double>>("max_angular_velocity", {0.80, 0.80, 0.80});
  auto_declare<double>("max_linear_speed_norm", 0.0);
  auto_declare<double>("max_angular_speed_norm", 0.0);
  auto_declare<std::vector<double>>("linear_deadband", {0.0, 0.0, 0.0});
  auto_declare<std::vector<double>>("angular_deadband", {0.0, 0.0, 0.0});
  auto_declare<std::vector<double>>("linear_accel_limit", {0.80, 0.80, 0.80});
  auto_declare<std::vector<double>>("angular_accel_limit", {3.00, 3.00, 3.00});
  auto_declare<std::vector<double>>("linear_jerk_limit", {5.0, 5.0, 5.0});
  auto_declare<std::vector<double>>("angular_jerk_limit", {25.0, 25.0, 25.0});
  return CallbackReturn::SUCCESS;
}

CallbackReturn FrankaCartesianTwistController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  command_topic_ = get_node()->get_parameter("command_topic").as_string();
  command_timeout_ = get_node()->get_parameter("command_timeout").as_double();
  command_filter_tau_ = std::max(0.0, get_node()->get_parameter("command_filter_tau").as_double());
  fresh_command_required_cycles_ = static_cast<int>(
      std::max<int64_t>(1, get_node()->get_parameter("fresh_command_required_cycles").as_int()));

  load_array_param(get_node(), "linear_scale", linear_scale_);
  load_array_param(get_node(), "angular_scale", angular_scale_);
  load_array_param(get_node(), "max_linear_velocity", max_linear_velocity_);
  load_array_param(get_node(), "max_angular_velocity", max_angular_velocity_);
  max_linear_speed_norm_ = std::max(0.0, get_node()->get_parameter("max_linear_speed_norm").as_double());
  max_angular_speed_norm_ = std::max(0.0, get_node()->get_parameter("max_angular_speed_norm").as_double());
  load_array_param(get_node(), "linear_deadband", linear_deadband_);
  load_array_param(get_node(), "angular_deadband", angular_deadband_);
  load_array_param(get_node(), "linear_accel_limit", linear_accel_limit_);
  load_array_param(get_node(), "angular_accel_limit", angular_accel_limit_);
  load_array_param(get_node(), "linear_jerk_limit", linear_jerk_limit_);
  load_array_param(get_node(), "angular_jerk_limit", angular_jerk_limit_);

  cartesian_velocity_interface_ =
      std::make_unique<franka_semantic_components::FrankaCartesianVelocityInterface>(false);

  command_sub_ = get_node()->create_subscription<geometry_msgs::msg::TwistStamped>(
      command_topic_, rclcpp::SystemDefaultsQoS(),
      std::bind(&FrankaCartesianTwistController::command_callback, this, std::placeholders::_1));

  RCLCPP_INFO(get_node()->get_logger(),
              "Configured FrankaCartesianTwistController arm_id=%s command_topic=%s "
              "timeout=%.3f filter_tau=%.3f fresh_cycles=%d",
              arm_id_.c_str(), command_topic_.c_str(), command_timeout_, command_filter_tau_,
              fresh_command_required_cycles_);
  return CallbackReturn::SUCCESS;
}

CallbackReturn FrankaCartesianTwistController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  cartesian_velocity_interface_->assign_loaned_command_interfaces(command_interfaces_);
  filtered_target_.fill(0.0);
  filtered_command_.fill(0.0);
  filtered_accel_.fill(0.0);
  latest_command_.writeFromNonRT(TwistCommand{});
  last_seen_command_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  fresh_command_cycles_ = 0;
  command_armed_ = false;
  command_active_ = true;
  return CallbackReturn::SUCCESS;
}

CallbackReturn FrankaCartesianTwistController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  filtered_target_.fill(0.0);
  filtered_command_.fill(0.0);
  filtered_accel_.fill(0.0);
  Eigen::Vector3d linear(0.0, 0.0, 0.0);
  Eigen::Vector3d angular(0.0, 0.0, 0.0);
  cartesian_velocity_interface_->setCommand(linear, angular);
  cartesian_velocity_interface_->release_interfaces();
  command_active_ = false;
  command_armed_ = false;
  fresh_command_cycles_ = 0;
  return CallbackReturn::SUCCESS;
}

void FrankaCartesianTwistController::command_callback(
    const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
  TwistCommand command;
  command.values = {
      msg->twist.linear.x,
      msg->twist.linear.y,
      msg->twist.linear.z,
      msg->twist.angular.x,
      msg->twist.angular.y,
      msg->twist.angular.z,
  };
  command.stamp = (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0)
                      ? get_node()->now()
                      : rclcpp::Time(msg->header.stamp);
  command.valid = true;
  latest_command_.writeFromNonRT(command);
}

std::array<double, 6> FrankaCartesianTwistController::compute_target_command(
    const rclcpp::Time& time,
    const rclcpp::Duration& period) {
  std::array<double, 6> raw_command{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  std::array<double, 6> filtered_command = filtered_target_;
  const TwistCommand* latest = latest_command_.readFromRT();
  if (latest == nullptr || !latest->valid || !command_active_) {
    filtered_target_ = filtered_command;
    return filtered_command;
  }

  const bool command_fresh = (time - latest->stamp).seconds() <= command_timeout_;
  if (!command_fresh) {
    fresh_command_cycles_ = 0;
    command_armed_ = false;
  } else if (latest->stamp != last_seen_command_stamp_) {
    last_seen_command_stamp_ = latest->stamp;
    if (fresh_command_cycles_ < fresh_command_required_cycles_) {
      ++fresh_command_cycles_;
    }
    if (fresh_command_cycles_ >= fresh_command_required_cycles_) {
      command_armed_ = true;
    }
  }

  if (command_fresh && command_armed_) {
    for (size_t i = 0; i < 3; ++i) {
      const double value = apply_deadband(latest->values[i] * linear_scale_[i], linear_deadband_[i]);
      raw_command[i] = clamp(value, -max_linear_velocity_[i], max_linear_velocity_[i]);
    }
    for (size_t i = 0; i < 3; ++i) {
      const double value =
          apply_deadband(latest->values[i + 3] * angular_scale_[i], angular_deadband_[i]);
      raw_command[i + 3] = clamp(value, -max_angular_velocity_[i], max_angular_velocity_[i]);
    }
    limit_vector_norm(raw_command, 0, max_linear_speed_norm_);
    limit_vector_norm(raw_command, 3, max_angular_speed_norm_);
  }

  const double dt = std::max(1e-4, period.seconds());
  const double alpha = command_filter_tau_ <= 1e-6 ? 1.0 : clamp(dt / (command_filter_tau_ + dt), 0.0, 1.0);
  for (size_t i = 0; i < filtered_command.size(); ++i) {
    filtered_command[i] = filtered_target_[i] + alpha * (raw_command[i] - filtered_target_[i]);
  }

  filtered_target_ = filtered_command;
  return filtered_command;
}

void FrankaCartesianTwistController::load_array_param(
    const std::shared_ptr<rclcpp_lifecycle::LifecycleNode>& node,
    const std::string& name,
    std::array<double, 3>& target) {
  const auto values = node->get_parameter(name).as_double_array();
  if (values.size() != 3) {
    throw std::runtime_error("Parameter '" + name + "' must have exactly 3 values.");
  }
  std::copy(values.begin(), values.end(), target.begin());
}

void FrankaCartesianTwistController::limit_vector_norm(
    std::array<double, 6>& command, size_t offset, double max_norm) {
  if (max_norm <= 0.0) {
    return;
  }
  const double norm = std::sqrt(command[offset] * command[offset] +
                                command[offset + 1] * command[offset + 1] +
                                command[offset + 2] * command[offset + 2]);
  if (norm <= max_norm || norm <= 1e-12) {
    return;
  }
  const double scale = max_norm / norm;
  command[offset] *= scale;
  command[offset + 1] *= scale;
  command[offset + 2] *= scale;
}

double FrankaCartesianTwistController::apply_deadband(double value, double deadband) {
  return std::abs(value) < deadband ? 0.0 : value;
}

double FrankaCartesianTwistController::clamp(double value, double min_value, double max_value) {
  return std::min(std::max(value, min_value), max_value);
}

}  // namespace yin_franka_cartesian_controller

PLUGINLIB_EXPORT_CLASS(yin_franka_cartesian_controller::FrankaCartesianTwistController,
                       controller_interface::ControllerInterface)
