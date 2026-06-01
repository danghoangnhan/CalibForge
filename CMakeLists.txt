# CalibForge — top-level build (SKELETON)
#
# Goal of this file at the scaffold stage: prove the "one source -> server + Jetson"
# multi-arch build logic configures cleanly. It does NOT yet build library targets.
#
# It gracefully degrades when no CUDA toolkit is present (e.g. a CI host or a dev
# laptop) so contributors can still configure and work on host-side code; the GPU
# targets are enabled only when a CUDA compiler is found.

cmake_minimum_required(VERSION 3.24)  # 3.24+: native CMAKE_CUDA_ARCHITECTURES handling

project(CalibForge
  VERSION 0.0.1
  DESCRIPTION "Unified GPU geometric camera calibration: estimate + apply, edge + server"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- The single-source multi-arch matrix (edit here as new Jetson/server GPUs ship) -----
# sm_72 Xavier · sm_87 Orin (primary edge) · sm_80/90 server (A100/H100) · sm_86/89 desktop/L4
set(CALIBFORGE_CUDA_ARCH_DEFAULT "72;80;86;87;89;90"
    CACHE STRING "Default -gencode arch matrix (server + Jetson in one build)")

# --- Detect CUDA; enable the GPU half only if a compiler is actually available ----------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${CALIBFORGE_CUDA_ARCH_DEFAULT}")
  endif()
  set(CALIBFORGE_HAS_CUDA ON)
  message(STATUS "CalibForge: CUDA found -> building GPU targets for sm_{${CMAKE_CUDA_ARCHITECTURES}}")
  # Trivial probe TU: proves the arch matrix compiles end-to-end on a CUDA host.
  add_executable(calibforge_arch_probe src/arch_probe.cu)
else()
  set(CALIBFORGE_HAS_CUDA OFF)
  message(WARNING
    "CalibForge: no CUDA toolkit found -> configuring HOST-ONLY. "
    "GPU targets (the arch matrix sm_{${CALIBFORGE_CUDA_ARCH_DEFAULT}}) are skipped. "
    "Install the CUDA toolkit to build the GPU half.")
endif()

# --- Module skeleton (interface-only at this stage; headers under <module>/include) ------
# core/ detect/ solve/ pipelines/ apply/ io/ bindings/  — see CLAUDE.md for responsibilities.
add_library(calibforge_core INTERFACE)
target_include_directories(calibforge_core INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/core/include
  ${CMAKE_CURRENT_SOURCE_DIR}/solve/include)
target_compile_features(calibforge_core INTERFACE cxx_std_17)

# --- Eigen (header-only) for the CPU solver. Permissive (MPL-2.0) per docs/DEPENDENCIES.md.
#     Populate + add the include dir directly (don't run Eigen's own CMake — header-only). ---
include(FetchContent)
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  message(STATUS "CalibForge: fetching Eigen 3.4.0 (header-only) ...")
  FetchContent_Populate(eigen)
endif()
add_library(calibforge_eigen INTERFACE)
target_include_directories(calibforge_eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})

# --- Sophus (header-only SE(3)/SO(3)). Permissive (MIT) per docs/DEPENDENCIES.md.
#     SOPHUS_USE_BASIC_LOGGING avoids the optional fmt transitive dependency. ---
FetchContent_Declare(sophus
  GIT_REPOSITORY https://github.com/strasdat/Sophus.git
  GIT_TAG 1.22.10
  GIT_SHALLOW TRUE)
FetchContent_GetProperties(sophus)
if(NOT sophus_POPULATED)
  message(STATUS "CalibForge: fetching Sophus 1.22.10 (header-only) ...")
  FetchContent_Populate(sophus)
endif()
add_library(calibforge_sophus INTERFACE)
target_include_directories(calibforge_sophus SYSTEM INTERFACE ${sophus_SOURCE_DIR})
target_compile_definitions(calibforge_sophus INTERFACE SOPHUS_USE_BASIC_LOGGING)
target_link_libraries(calibforge_sophus INTERFACE calibforge_eigen)

# --- Tests (host-only; no CUDA needed). Run: cmake --build build && ctest --test-dir build ---
enable_testing()
add_executable(calibforge_tests
  tests/test_main.cpp
  tests/test_pinhole_camera.cpp
  tests/test_brown_conrady_camera.cpp
  tests/test_least_squares.cpp
  tests/test_calibrate_single.cpp
  tests/test_observability.cpp)
target_link_libraries(calibforge_tests PRIVATE calibforge_core calibforge_eigen calibforge_sophus)
target_include_directories(calibforge_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
add_test(NAME calibforge_tests COMMAND calibforge_tests)

message(STATUS "CalibForge ${PROJECT_VERSION} configured (CUDA=${CALIBFORGE_HAS_CUDA}). Core math is header-only; CPU tests build host-only.")
