# `bindings/` — language & framework bindings

## Python (`python/`, pybind11) — built & tested

The `calibforge` module exposes the camera models (pinhole, Brown-Conrady, Kannala-Brandt,
double-sphere, EUCM), the observability gate + `parameter_uncertainty`, and `calibrate_single`.
pybind11 is BSD-3 (see [`../docs/DEPENDENCIES.md`](../docs/DEPENDENCIES.md)).

```bash
cmake -S . -B build -DCALIBFORGE_PYTHON=ON   # fetches pybind11
cmake --build build --target calibforge_python -j
PYTHONPATH=build python3 -m pytest tests/python   # or: ctest --test-dir build -R calibforge_python
```

Off by default, so the core build/CI is unaffected. See `tests/python/test_bindings.py`.

## ROS2 (`ros2/`, rclcpp) — ament package (build in a ROS2 workspace)

`calibforge_node` publishes a calibrated camera's `sensor_msgs/CameraInfo` from a CalibForge
calibration; the message mapping reuses the header-only `io/` layer (covered by the host test
suite). It is an `ament_cmake` package, **not** built by the core CMake (which stays ROS2-free):

```bash
ln -s "$(pwd)/bindings/ros2" <ros2_ws>/src/calibforge_ros2
cd <ros2_ws> && colcon build --packages-select calibforge_ros2
ros2 run calibforge_ros2 calibforge_node --ros-args -p calibration_yaml:=calib.yaml
```

URDF for a multi-camera rig is emitted offline via `calibforge::io::toIsaacUrdf`
(see `io/isaac_urdf.hpp`). Online/targetless `ExtrinsicTracker` integration arrives with v0.5.
