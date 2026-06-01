#pragma once
// Minimal dependency-free test harness for CalibForge (host-buildable with g++).
// Tests self-register via CF_TEST; test_main.cpp calls cf_test::run_all().
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <vector>

namespace cf_test {

struct Case {
  std::string name;
  std::function<void(bool&)> fn;
};

inline std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}

struct Reg {
  Reg(const std::string& name, std::function<void(bool&)> fn) {
    registry().push_back({name, std::move(fn)});
  }
};

inline int run_all() {
  int failed = 0;
  for (auto& c : registry()) {
    bool ok = true;
    try {
      c.fn(ok);
    } catch (const std::exception& e) {
      ok = false;
      std::printf("  threw std::exception: %s\n", e.what());
    } catch (...) {
      ok = false;
      std::printf("  threw unknown exception\n");
    }
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", c.name.c_str());
    if (!ok) ++failed;
  }
  std::printf("\n%zu tests, %d failed\n", registry().size(), failed);
  return failed == 0 ? 0 : 1;
}

}  // namespace cf_test

#define CF_TEST(name)                                            \
  static void name(bool&);                                       \
  static ::cf_test::Reg cf_reg_##name(#name, name);              \
  static void name(bool& _ok)

#define CF_EXPECT_NEAR(a, b, tol)                                                  \
  do {                                                                             \
    double _a = (a), _b = (b), _t = (tol);                                         \
    if (std::fabs(_a - _b) > _t) {                                                 \
      _ok = false;                                                                 \
      std::printf("  EXPECT_NEAR failed: %s (%.9g) vs %s (%.9g), tol %.3g @ %s:%d\n", \
                  #a, _a, #b, _b, _t, __FILE__, __LINE__);                          \
    }                                                                              \
  } while (0)

#define CF_EXPECT_TRUE(x)                                            \
  do {                                                               \
    if (!(x)) {                                                      \
      _ok = false;                                                   \
      std::printf("  EXPECT_TRUE failed: %s @ %s:%d\n", #x, __FILE__, __LINE__); \
    }                                                                \
  } while (0)
