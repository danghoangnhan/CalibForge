#include "calibforge/kalibr_camchain.hpp"

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/isaac_urdf.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::io::CameraExtrinsic;
using calibforge::io::consecutiveFromReference;
using calibforge::io::KalibrCamchain;
using calibforge::io::KalibrCamera;
using calibforge::io::parseKalibrYaml;
using calibforge::io::RigDescription;
using calibforge::io::toIsaacUrdf;
using calibforge::io::toKalibrYaml;

static bool contains(const std::string& h, const std::string& n) {
  return h.find(n) != std::string::npos;
}

// A 2-camera camchain (cam0 reference + cam1 with a consecutive extrinsic) round-trips.
CF_TEST(kalibr_camchain_roundtrips) {
  KalibrCamchain chain;
  KalibrCamera c0;
  c0.camera_model = "pinhole";
  c0.intrinsics = {520.0, 521.5, 320.0, 240.0};
  c0.distortion_model = "radtan";
  c0.distortion_coeffs = {-0.15, 0.04, 0.001, -0.001};
  c0.width = 640; c0.height = 480;
  KalibrCamera c1 = c0;
  c1.intrinsics = {510.0, 508.0, 322.0, 238.0};
  c1.has_extrinsic = true;
  c1.T_cn_cnm1 << 0.9998, -0.0100, 0.0150, -0.100,
                  0.0101, 0.9999, -0.0050, 0.0050,
                  -0.0149, 0.0051, 0.9999, 0.0020,
                  0.0, 0.0, 0.0, 1.0;
  chain.cameras = {c0, c1};

  KalibrCamchain back = parseKalibrYaml(toKalibrYaml(chain));
  CF_EXPECT_TRUE(back.cameras.size() == 2);
  CF_EXPECT_TRUE(back.cameras[0].camera_model == "pinhole");
  CF_EXPECT_TRUE(back.cameras[0].distortion_model == "radtan");
  CF_EXPECT_TRUE(!back.cameras[0].has_extrinsic);
  CF_EXPECT_TRUE(back.cameras[1].has_extrinsic);
  for (int j = 0; j < 4; ++j) {
    CF_EXPECT_NEAR(back.cameras[1].intrinsics[j], c1.intrinsics[j], 1e-9);
    CF_EXPECT_NEAR(back.cameras[0].distortion_coeffs[j], c0.distortion_coeffs[j], 1e-9);
  }
  CF_EXPECT_TRUE(back.cameras[0].width == 640 && back.cameras[0].height == 480);
  for (int r = 0; r < 4; ++r)
    for (int cc = 0; cc < 4; ++cc)
      CF_EXPECT_NEAR(back.cameras[1].T_cn_cnm1(r, cc), c1.T_cn_cnm1(r, cc), 1e-9);
}

// Reference extrinsics T_ck_c0 convert to Kalibr's consecutive chain and re-compose correctly.
CF_TEST(kalibr_consecutive_extrinsics_recompose) {
  Sophus::SE3d T_c1_c0(Sophus::SO3d::exp(Eigen::Vector3d(0.02, -0.01, 0.015)),
                       Eigen::Vector3d(-0.10, 0.005, 0.002));
  Sophus::SE3d T_c2_c0(Sophus::SO3d::exp(Eigen::Vector3d(-0.015, 0.02, -0.01)),
                       Eigen::Vector3d(0.10, -0.004, 0.003));
  auto consec = consecutiveFromReference({T_c1_c0, T_c2_c0});  // [T_c1_c0, T_c2_c1]
  CF_EXPECT_TRUE(consec.size() == 2);

  // cam0 -> cam2 via the chain (T_c2_c1 * T_c1_c0) must equal the reference T_c2_c0.
  Sophus::SE3d chained = Sophus::SE3d(consec[1]) * Sophus::SE3d(consec[0]);
  CF_EXPECT_TRUE((chained * T_c2_c0.inverse()).log().norm() < 1e-9);
  // First link is exactly T_c1_c0.
  CF_EXPECT_TRUE((Sophus::SE3d(consec[0]) * T_c1_c0.inverse()).log().norm() < 1e-9);
}

// The Isaac URDF emitter (io #10) already handles an N-camera rig: one fixed joint + optical
// sub-frame per camera.
CF_TEST(isaac_urdf_emits_multi_camera_rig) {
  RigDescription rig;
  for (const char* n : {"front", "left", "right"}) {
    CameraExtrinsic c;
    c.name = n;
    c.T_base_camera = Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.1, 0.0, 0.2));
    rig.cameras.push_back(c);
  }
  std::string urdf = toIsaacUrdf(rig);
  CF_EXPECT_TRUE(contains(urdf, "<joint name=\"base_link_to_front\""));
  CF_EXPECT_TRUE(contains(urdf, "<joint name=\"base_link_to_left\""));
  CF_EXPECT_TRUE(contains(urdf, "<joint name=\"base_link_to_right\""));
  CF_EXPECT_TRUE(contains(urdf, "<link name=\"right_optical\"/>"));
}
