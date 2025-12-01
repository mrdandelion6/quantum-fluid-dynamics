cmake_minimum_required(VERSION 3.18)
project(FluidSimulation LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture for GTX 1080 Ti (Pascal - compute capability 6.1)
set(CMAKE_CUDA_ARCHITECTURES 61)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    /usr/include  # For GLM (header-only)
)

# Source files
set(SOURCES
    src/main.cpp
    src/renderer.cpp
    src/fluid_solver.cu
)

# Create executable
add_executable(fluid_sim ${SOURCES})

# Link libraries
target_link_libraries(fluid_sim
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    GLEW::GLEW
    glfw
    cudart
)

# Set CUDA flags
set_target_properties(fluid_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Compiler flags
target_compile_options(fluid_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
    >
)

message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
