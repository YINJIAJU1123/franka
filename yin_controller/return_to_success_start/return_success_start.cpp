// One-shot, bounded return using unmodified libfranka 0.20.4 MotionGenerator.
// No collision, impedance, recovery, ACT, camera, or gripper-command changes.
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/robot.h>
#include "examples_common.h"

using Q = std::array<double, 7>;
const Q kExpected{{0.631602644492, 0.529338777065, -0.401326894760,
                   -1.590189218520, 0.254921674728, 2.248061895370, 0.917087197304}};
const Q kTarget{{0.619065, 0.554681, -0.401264, -1.533722,
                 0.273446, 2.165253, 0.913639}};
constexpr double kSpeed = 0.005;
struct Row { double time; Q q, dq, command; };

void print_q(const Q& q) {
  std::cout << '[';
  for (size_t i = 0; i < 7; ++i) std::cout << (i ? "," : "") << q[i];
  std::cout << ']';
}
void save(const std::string& path, const std::vector<Row>& rows) {
  std::ofstream f(path);
  f << std::setprecision(12) << "time";
  for (const auto* key : {"q", "dq", "command"})
    for (int i = 1; i <= 7; ++i) f << ',' << key << i;
  f << '\n';
  for (const auto& r : rows) {
    f << r.time;
    for (const auto& q : {r.q, r.dq, r.command})
      for (double x : q) f << ',' << x;
    f << '\n';
  }
}

int main(int argc, char** argv) {
  if (argc != 3 || (std::string(argv[1]) != "--plan" && std::string(argv[1]) != "--execute"))
    return 64;
  std::cout << std::setprecision(12);
  std::vector<Row> rows;
  rows.reserve(22000);
  const bool execute = std::string(argv[1]) == "--execute";
  try {
    if (!execute) {
      franka::RobotState s{};
      s.q = kExpected; s.q_d = kExpected;
      MotionGenerator generator(kSpeed, kTarget);
      for (size_t i = 0; i < 20000; ++i) {
        auto command = generator(s, franka::Duration(i == 0 ? 0 : 1));
        rows.push_back({i / 1000.0, s.q, s.dq, command.q});
        s.q = command.q; s.q_d = command.q;
        if (command.motion_finished) break;
      }
      if (rows.back().time >= 19.999) throw std::runtime_error("Plan did not finish");
      save(argv[2], rows);
      std::cout << "{\"mode\":\"offline_plan_no_robot_connection\",\"duration_s\":"
                << rows.back().time << ",\"speed_factor\":" << kSpeed << ",\"target\":";
      print_q(kTarget); std::cout << "}\n";
      return 0;
    }

    franka::Robot robot("172.16.0.2", franka::RealtimeConfig::kEnforce);
    auto initial = robot.readOnce();
    if (initial.robot_mode != franka::RobotMode::kIdle || bool(initial.current_errors))
      throw std::runtime_error("Robot is not idle/error-free");
    for (size_t j = 0; j < 7; ++j) {
      if (std::abs(initial.q[j] - kExpected[j]) > 0.003 ||
          std::abs(initial.dq[j]) > 0.005 ||
          std::abs(initial.q[j] - kTarget[j]) > 0.100)
        throw std::runtime_error("Current joints differ from the approved offline plan");
    }
    franka::Gripper gripper("172.16.0.2");
    if (gripper.readOnce().width < 0.074)
      throw std::runtime_error("Gripper is not open");
    std::cout << "{\"starting_bounded_return\":true,\"q_before\":";
    print_q(initial.q); std::cout << ",\"target\":"; print_q(kTarget);
    std::cout << ",\"speed_factor\":" << kSpeed << "}" << std::endl;
    MotionGenerator generator(kSpeed, kTarget);
    double elapsed = 0.0;
    robot.control([&](const franka::RobotState& s, franka::Duration dt) -> franka::JointPositions {
      elapsed += dt.toSec();
      if (elapsed > 20.0 || bool(s.current_errors))
        throw std::runtime_error("Return time/error monitor stopped execution");
      for (size_t j = 0; j < 7; ++j) {
        if (!std::isfinite(s.q[j]) || !std::isfinite(s.dq[j]) ||
            std::abs(s.dq[j]) > 0.030 ||
            s.q[j] < std::min(initial.q[j], kTarget[j]) - 0.006 ||
            s.q[j] > std::max(initial.q[j], kTarget[j]) + 0.006)
          throw std::runtime_error("Return velocity/position monitor stopped execution");
      }
      auto command = generator(s, dt);
      rows.push_back({elapsed, s.q, s.dq, command.q});
      return command;
    }, franka::ControllerMode::kJointImpedance, true, 100.0);
    auto final = robot.readOnce();
    double error = 0.0;
    for (size_t j = 0; j < 7; ++j) error = std::max(error, std::abs(final.q[j] - kTarget[j]));
    save(argv[2], rows);
    std::cout << "{\"return_completed\":true,\"q_after\":"; print_q(final.q);
    std::cout << ",\"max_joint_error_rad\":" << error << ",\"duration_s\":" << elapsed
              << ",\"current_errors\":\"" << final.current_errors
              << "\",\"gripper_commands_sent\":0,\"controller_parameters_changed\":false}\n";
    return error <= 0.0015 && !bool(final.current_errors) ? 0 : 3;
  } catch (const std::exception& e) {
    save(argv[2], rows);
    std::cerr << "RETURN_STOPPED: " << e.what() << std::endl;
    return 2;
  }
}
