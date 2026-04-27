# Plan-R1 TensorRT Export - Top-level CMake
#
# This project builds:
# 1. Custom TensorRT plugin (libscatter_add_plugin.so)
# 2. C++ inference library and demo

cmake_minimum_required(VERSION 3.18)
project(planr1_tensorrt VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Options
option(BUILD_PLUGIN "Build custom TensorRT plugin" ON)
option(BUILD_CPP "Build C++ inference library" ON)

# Find CUDA
find_package(CUDA REQUIRED)
message(STATUS "Found CUDA: ${CUDA_VERSION}")
message(STATUS "CUDA include: ${CUDA_INCLUDE_DIRS}")

# Find TensorRT
if(DEFINED ENV{TENSORRT_ROOT})
    set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
elseif(NOT DEFINED TENSORRT_ROOT)
    # Try common locations
    if(EXISTS "/usr/include/NvInfer.h")
        set(TENSORRT_ROOT "/usr")
    elseif(EXISTS "/opt/tensorrt")
        set(TENSORRT_ROOT "/opt/tensorrt")
    else()
        message(FATAL_ERROR "TENSORRT_ROOT not set. Please specify:\n"
                            "  cmake -DTENSORRT_ROOT=/path/to/tensorrt ..")
    endif()
endif()

message(STATUS "TENSORRT_ROOT: ${TENSORRT_ROOT}")

# TensorRT include directories
set(TENSORRT_INCLUDE_DIRS
    ${TENSORRT_ROOT}/include
    ${TENSORRT_ROOT}/include/x86_64-linux-gnu
)

# TensorRT libraries
find_library(TENSORRT_NVINFER nvinfer 
    PATHS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64 /usr/lib/x86_64-linux-gnu)
find_library(TENSORRT_NVINFER_PLUGIN nvinfer_plugin 
    PATHS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64 /usr/lib/x86_64-linux-gnu)
find_library(TENSORRT_NVONNXPARSER nvonnxparser 
    PATHS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64 /usr/lib/x86_64-linux-gnu)

if(NOT TENSORRT_NVINFER)
    message(FATAL_ERROR "Could not find nvinfer library")
endif()

message(STATUS "TENSORRT_NVINFER: ${TENSORRT_NVINFER}")
message(STATUS "TENSORRT_NVINFER_PLUGIN: ${TENSORRT_NVINFER_PLUGIN}")

# ============================================================================
# Build Custom Plugin
# ============================================================================

if(BUILD_PLUGIN)
    message(STATUS "Building TensorRT plugin...")
    
    add_library(scatter_add_plugin SHARED
        plugins/scatter_add_kernel.cu
        plugins/scatter_add_plugin.cpp
    )
    
    target_include_directories(scatter_add_plugin
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/plugins
            ${TENSORRT_INCLUDE_DIRS}
            ${CUDA_INCLUDE_DIRS}
    )
    
    target_link_libraries(scatter_add_plugin
        ${TENSORRT_NVINFER}
        ${CUDA_LIBRARIES}
    )
    
    set_target_properties(scatter_add_plugin PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    # Install plugin
    install(TARGETS scatter_add_plugin
        LIBRARY DESTINATION lib
    )
endif()

# ============================================================================
# Build C++ Inference Library
# ============================================================================

if(BUILD_CPP)
    message(STATUS "Building C++ inference library...")
    
    add_library(planr1_tensorrt
        cpp/src/planr1_tensorrt.cpp
    )
    
    target_include_directories(planr1_tensorrt
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${TENSORRT_INCLUDE_DIRS}
            ${CUDA_INCLUDE_DIRS}
    )
    
    target_link_libraries(planr1_tensorrt
        PUBLIC
            ${TENSORRT_NVINFER}
            ${CUDA_LIBRARIES}
            ${CMAKE_DL_LIBS}
    )
    
    # Build demo executable
    add_executable(planr1_tensorrt_demo
        cpp/src/main.cpp
    )
    
    target_link_libraries(planr1_tensorrt_demo
        PRIVATE
            planr1_tensorrt
    )
    
    target_include_directories(planr1_tensorrt_demo
        PRIVATE
            ${TENSORRT_INCLUDE_DIRS}
            ${CUDA_INCLUDE_DIRS}
    )
    
    # Install
    install(TARGETS planr1_tensorrt planr1_tensorrt_demo
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    
    install(DIRECTORY cpp/include/
        DESTINATION include
    )
endif()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "╔══════════════════════════════════════════════════════════╗")
message(STATUS "║     Plan-R1 TensorRT Build Configuration                 ║")
message(STATUS "╠══════════════════════════════════════════════════════════╣")
message(STATUS "║  CMAKE_BUILD_TYPE:     ${CMAKE_BUILD_TYPE}")
message(STATUS "║  CUDA_VERSION:         ${CUDA_VERSION}")
message(STATUS "║  TENSORRT_ROOT:        ${TENSORRT_ROOT}")
message(STATUS "║  BUILD_PLUGIN:         ${BUILD_PLUGIN}")
message(STATUS "║  BUILD_CPP:            ${BUILD_CPP}")
message(STATUS "╚══════════════════════════════════════════════════════════╝")
message(STATUS "")
