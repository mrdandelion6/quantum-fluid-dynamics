#!/bin/bash

echo "======================================"
echo "3D Fluid Simulation - Build Script"
echo "======================================"
echo ""

# Check for required packages
echo "Checking dependencies..."

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA not found. Please install CUDA toolkit."
    exit 1
fi

echo "✓ CUDA found"

# Check for required libraries
missing_deps=()

# Check for GLFW (either X11 or Wayland)
if ! pkg-config --exists glfw3; then
    missing_deps+=("glfw")
else
    echo "✓ GLFW found"
fi

# Check for GLEW
if ! pkg-config --exists glew; then
    missing_deps+=("glew")
else
    echo "✓ GLEW found"
fi

# Check for GLM (it doesn't have pkg-config, check for header instead)
if [ ! -f "/usr/include/glm/glm.hpp" ]; then
    missing_deps+=("glm")
else
    echo "✓ GLM found"
fi

if [ ${#missing_deps[@]} -ne 0 ]; then
    echo ""
    echo "ERROR: Missing dependencies: ${missing_deps[*]}"
    echo ""
    echo "On Arch Linux, install with:"
    echo "  For Wayland: sudo pacman -S glfw-wayland glew glm"
    echo "  For X11:     sudo pacman -S glfw-x11 glew glm"
    echo ""
    exit 1
fi

# Detect display server
if [ -n "$WAYLAND_DISPLAY" ]; then
    echo "✓ Wayland detected"
    DISPLAY_SERVER="Wayland"
elif [ -n "$DISPLAY" ]; then
    echo "✓ X11 detected"
    DISPLAY_SERVER="X11"
else
    echo "⚠ Warning: Could not detect display server"
    DISPLAY_SERVER="Unknown"
fi

echo ""

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Run CMake
echo "Running CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: CMake configuration failed!"
    exit 1
fi

# Build
echo ""
echo "Building..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Build failed!"
    exit 1
fi

echo ""
echo "======================================"
echo "✓ Build successful!"
echo "======================================"
echo ""
echo "Display Server: $DISPLAY_SERVER"
echo ""
echo "To run the simulation:"
echo "  cd build && ./fluid_sim"
echo ""
echo "Or simply run:"
echo "  ./build.sh run"
echo ""

# If 'run' argument is passed, run the simulation
if [ "$1" == "run" ]; then
    echo "Starting simulation..."
    echo ""
    ./fluid_sim
fi
