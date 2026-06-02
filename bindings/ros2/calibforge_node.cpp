//
// CalibForge — ROS2 node (rclcpp): publish a calibrated camera's sensor_msgs/CameraInfo.
//
// Built ONLY in a ROS2 workspace (colcon / ament_cmake — see CMakeLists.txt + package.xml in
// this directory), not by the core CalibForge CMake. The calibration→message mapping reuses
// the header-only io layer (calibforge/opencv_yaml.hpp), which is exercised by the host test
// suite (test_opencv_yaml / test_ros_isaac_emit); this node is the thin rclcpp wrapper around
// it. URDF for a rig is emitted offline via calibforge::io::toIsaacUrdf.

#include <chrono>
#include <memory>
#include <optional>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include "calibforge/opencv_yaml.hpp"

namespace calibforge {
namespace ros2 {

using namespace std::chrono_literals;

class CameraInfoPublisher : public rclcpp::Node {
 public:
  CameraInfoPublisher() : rclcpp::Node("calibforge_camera_info") {
    const std::string yaml = declare_parameter<std::string>("calibration_yaml", "");
    const std::string frame = declare_parameter<std::string>("frame_id", "camera_optical");
    const std::string topic = declare_parameter<std::string>("topic", "camera_info");

    pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(topic, 10);
    if (!yaml.empty()) {
      msg_ = toCameraInfoMsg(io::readOpenCvYaml(yaml), frame);
      RCLCPP_INFO(get_logger(), "Loaded calibration from %s (%dx%d)", yaml.c_str(),
                  msg_->width, msg_->height);
    } else {
      RCLCPP_WARN(get_logger(), "No 'calibration_yaml' parameter set; nothing to publish.");
    }
    timer_ = create_wall_timer(1s, [this]() {
      if (msg_) pub_->publish(*msg_);
    });
  }

  // io::Intrinsics (K, distortion, size) -> sensor_msgs/CameraInfo (P = [K|0] for monocular).
  static sensor_msgs::msg::CameraInfo toCameraInfoMsg(const io::Intrinsics& in,
                                                      const std::string& frame_id) {
    sensor_msgs::msg::CameraInfo m;
    m.header.frame_id = frame_id;
    m.width = static_cast<unsigned>(in.image_width);
    m.height = static_cast<unsigned>(in.image_height);
    m.distortion_model = in.distortion_model;
    m.d.assign(in.distortion_coeffs.begin(), in.distortion_coeffs.end());
    for (int i = 0; i < 9; ++i) m.k[i] = in.camera_matrix[i];
    m.r = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double fx = in.camera_matrix[0], fy = in.camera_matrix[4];
    const double cx = in.camera_matrix[2], cy = in.camera_matrix[5];
    m.p = {fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0};
    return m;
  }

 private:
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::optional<sensor_msgs::msg::CameraInfo> msg_;
};

}  // namespace ros2
}  // namespace calibforge

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<calibforge::ros2::CameraInfoPublisher>());
  rclcpp::shutdown();
  return 0;
}
