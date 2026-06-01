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

message(STATUS "CalibForge ${PROJECT_VERSION} configured (CUDA=${CALIBFORGE_HAS_CUDA}). Scaffold only — no library targets yet.")
