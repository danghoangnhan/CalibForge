#include "calibforge/isaac_urdf.hpp"

#include <string>

#include <Eigen/Dense>

#include "calibforge/brown_conrady_camera.hpp"
#include "calibforge/frames.hpp"
#include "calibforge/ros_camera_info.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::BrownConradyCamera;
using calibforge::io::bodyToOpticalRotation;
using calibforge::io::brownConradyToCameraInfo;
using calibforge::io::CameraExtrinsic;
using calibforge::io::CameraInfo;
using calibforge::io::RigDescription;
using calibforge::io::rotationToRpy;
using calibforge::io::rpyToRotation;
using calibforge::io::toCameraInfoYaml;
using calibforge::io::toIsaacUrdf;

static bool contains(const std::string& hay, const std::string& needle) {
  return hay.find(needle) != std::string::npos;
}

// CameraInfo carries K, the right distortion model for a 5-coeff input, and P = [K | 0].
CF_TEST(camera_info_emits_expected_fields) {
  BrownConradyCamera cam(520.0, 520.0, 320.0, 240.0, -0.15, 0.04, 0.001, -0.001, 0.0);
  CameraInfo ci = brownConradyToCameraInfo(cam, 640, 480, "front_left");
  std::string y = toCameraInfoYaml(ci);

  CF_EXPECT_TRUE(contains(y, "image_width: 640"));
  CF_EXPECT_TRUE(contains(y, "camera_name: front_left"));
  CF_EXPECT_TRUE(contains(y, "distortion_model: plumb_bob"));
  CF_EXPECT_TRUE(ci.D.size() == 5);
  CF_EXPECT_TRUE(contains(y, "data: [520, 0, 320, 0, 520, 240, 0, 0, 1]"));           // K
  CF_EXPECT_TRUE(contains(y, "data: [520, 0, 320, 0, 0, 520, 240, 0, 0, 0, 1, 0]"));  // P=[K|0]
}

// The URDF is rooted at base_link with one fixed joint per camera and an optical sub-frame.
CF_TEST(isaac_urdf_has_base_link_root_and_fixed_joints) {
  RigDescription rig;
  CameraExtrinsic c;
  c.name = "front_left";
  c.T_base_camera = Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.12, 0.0, 0.30));
  rig.cameras.push_back(c);
  std::string urdf = toIsaacUrdf(rig);

  CF_EXPECT_TRUE(contains(urdf, "<link name=\"base_link\"/>"));
  CF_EXPECT_TRUE(contains(urdf, "<joint name=\"base_link_to_front_left\" type=\"fixed\">"));
  CF_EXPECT_TRUE(contains(urdf, "<parent link=\"base_link\"/>"));
  CF_EXPECT_TRUE(contains(urdf, "<link name=\"front_left_optical\"/>"));
  CF_EXPECT_TRUE(contains(urdf, "xyz=\"0.12 0 0.3\""));
}

// The body->optical rotation round-trips through URDF rpy and equals the REP-103 constant.
CF_TEST(isaac_urdf_optical_rotation_is_rep103) {
  Eigen::Matrix3d R_bo = bodyToOpticalRotation();
  Eigen::Vector3d rpy = rotationToRpy(R_bo);
  Eigen::Matrix3d R_back = rpyToRotation(rpy[0], rpy[1], rpy[2]);
  CF_EXPECT_TRUE((R_back - R_bo).norm() < 1e-9);
  // The optical rpy is (-pi/2, 0, -pi/2).
  CF_EXPECT_NEAR(rpy[0], -M_PI / 2.0, 1e-9);
  CF_EXPECT_NEAR(rpy[1], 0.0, 1e-9);
  CF_EXPECT_NEAR(rpy[2], -M_PI / 2.0, 1e-9);
}

// A general extrinsic rotation round-trips through the URDF rpy convention (verify frames).
CF_TEST(extrinsic_rotation_roundtrips_through_rpy) {
  Sophus::SO3d Rso = Sophus::SO3d::exp(Eigen::Vector3d(0.3, -0.5, 0.2));
  Eigen::Matrix3d R = Rso.matrix();
  Eigen::Vector3d rpy = rotationToRpy(R);
  Eigen::Matrix3d R_back = rpyToRotation(rpy[0], rpy[1], rpy[2]);
  CF_EXPECT_TRUE((R_back - R).norm() < 1e-9);
}

// The URDF + CameraInfo emitted for the same camera reference the same frame_id, so a
// downstream consumer (e.g. Isaac Perceptor) can join the kinematic frame to the intrinsics.
// This is the design contract documented in isaac_urdf.hpp: distortion lives in CameraInfo,
// the URDF carries only the frame tree, and the frame_id is what stitches them.
CF_TEST(isaac_urdf_and_camera_info_share_frame_id) {
  const std::string name = "front_left";

  RigDescription rig;
  CameraExtrinsic c;
  c.name = name;
  c.T_base_camera = Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(0.0, 0.0, 0.0)),
                                 Eigen::Vector3d(0.12, 0.0, 0.30));
  BrownConradyCamera cam(520.0, 520.0, 320.0, 240.0, -0.15, 0.04, 0.001, -0.001, 0.0);
  c.info = brownConradyToCameraInfo(cam, 640, 480, name + "_optical");
  rig.cameras.push_back(c);

  const std::string urdf = toIsaacUrdf(rig);
  const std::string ci_yaml = toCameraInfoYaml(c.info);

  // The optical sub-frame on the URDF must match the frame_id stamped on the CameraInfo.
  CF_EXPECT_TRUE(contains(urdf, "<link name=\"" + name + "_optical\"/>"));
  CF_EXPECT_TRUE(contains(ci_yaml, "camera_name: " + name + "_optical"));
}
