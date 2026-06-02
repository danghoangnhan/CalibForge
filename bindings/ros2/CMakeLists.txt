# CalibForge ROS2 node — ament_cmake package. Build inside a ROS2 workspace with colcon:
#   ln -s <calibforge>/bindings/ros2 <ws>/src/calibforge_ros2 && colcon build
# NOT built by the core CalibForge CMakeLists (that one stays ROS2-free / host-only).
cmake_minimum_required(VERSION 3.16)
project(calibforge_ros2)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(calibforge_node calibforge_node.cpp)
# Reuse the header-only io layer from the repo (opencv_yaml.hpp is dependency-free).
target_include_directories(calibforge_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../io/include)
ament_target_dependencies(calibforge_node rclcpp sensor_msgs)

install(TARGETS calibforge_node DESTINATION lib/${PROJECT_NAME})

ament_package()
