cmake_minimum_required(VERSION 3.14)
project(planr1_inference VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# Options
# ============================================================
option(USE_CUDA "Enable CUDA support for ONNX Runtime" ON)

# ============================================================
# Find ONNX Runtime
# ============================================================
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
endif()

if(NOT DEFINED ONNXRUNTIME_ROOT)
    if(EXISTS "/usr/local/include/onnxruntime")
        set(ONNXRUNTIME_ROOT "/usr/local")
    elseif(EXISTS "/opt/onnxruntime")
        set(ONNXRUNTIME_ROOT "/opt/onnxruntime")
    else()
        message(FATAL_ERROR "ONNXRUNTIME_ROOT not set. Please specify:\n"
                            "  cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..")
    endif()
endif()

message(STATUS "ONNXRUNTIME_ROOT: ${ONNXRUNTIME_ROOT}")

set(ONNXRUNTIME_INCLUDE_DIRS 
    ${ONNXRUNTIME_ROOT}/include
    ${ONNXRUNTIME_ROOT}/include/onnxruntime
    ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session
)

find_library(ONNXRUNTIME_LIBRARY 
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_ROOT}/lib ${ONNXRUNTIME_ROOT}/lib64
    REQUIRED
)

message(STATUS "ONNXRUNTIME_LIBRARY: ${ONNXRUNTIME_LIBRARY}")

# ============================================================
# Library: planr1_inference
# ============================================================
add_library(planr1_inference
    src/planr1_inference.cpp
)

target_include_directories(planr1_inference
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(planr1_inference
    PUBLIC
        ${ONNXRUNTIME_LIBRARY}
)

if(USE_CUDA)
    target_compile_definitions(planr1_inference PUBLIC USE_CUDA)
endif()

# ============================================================
# Executable: planr1_demo
# ============================================================
add_executable(planr1_demo
    src/main.cpp
)

target_link_libraries(planr1_demo
    PRIVATE
        planr1_inference
)

target_include_directories(planr1_demo
    PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIRS}
)

# ============================================================
# Install
# ============================================================
install(TARGETS planr1_inference planr1_demo
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)

# ============================================================
# Summary
# ============================================================
message(STATUS "")
message(STATUS "╔═══════════════════════════════════════════════════╗")
message(STATUS "║     Plan-R1 ONNX Runtime Inference               ║")
message(STATUS "╠═══════════════════════════════════════════════════╣")
message(STATUS "║ CMAKE_BUILD_TYPE:   ${CMAKE_BUILD_TYPE}")
message(STATUS "║ USE_CUDA:           ${USE_CUDA}")
message(STATUS "║ ONNXRUNTIME_ROOT:   ${ONNXRUNTIME_ROOT}")
message(STATUS "╠═══════════════════════════════════════════════════╣")
message(STATUS "║ Targets:                                         ║")
message(STATUS "║   - planr1_inference (library)                   ║")
message(STATUS "║   - planr1_demo (executable)                     ║")
message(STATUS "╚═══════════════════════════════════════════════════╝")
message(STATUS "")
