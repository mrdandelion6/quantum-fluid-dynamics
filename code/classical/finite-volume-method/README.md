# 3D Incompressible Fluid Simulation

A real-time 3D fluid simulator implementing the **Navier-Stokes equations** for  an **incompressible Newtonian fluid** using CUDA for computation and OpenGL for visualization. Made entirely by me during my research on CFD for a thesis course (PHY473) at the University of Toronto.

## Features

- **Full 3D Navier-Stokes solver** on GPU (CUDA)
- **Incompressible flow** with pressure projection
- **Real-time visualization** with OpenGL
- **10,000 particle tracers** for flow visualization
- **Interactive camera** controls
- Optimized for NVIDIA GTX 1080 Ti

## Physics Implemented

### Incompressible Navier-Stokes Equations:

```
Continuity:  ∇·u = 0
Momentum:    ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u
```

### Numerical Method:

- **Spatial discretization**: Finite Volume Method (FVM)
- **Time integration**: Semi-Lagrangian advection
- **Pressure solve**: Jacobi iteration for Poisson equation
- **Incompressibility**: Projection method

### Algorithm Steps (per timestep):

1. **Advection**: Semi-Lagrangian method for convection term
2. **Diffusion**: Implicit viscous diffusion (Jacobi iterations)
3. **Projection**: Pressure Poisson solve + velocity correction
4. **Boundary conditions**: No-slip walls
5. **Particle update**: Tracer particles follow velocity field

## Requirements

### Hardware:

- NVIDIA GPU with CUDA support (tested on GTX 1080 Ti)
- 4GB+ VRAM recommended

### Software (Arch Linux):

```bash
# Install dependencies
sudo pacman -S cuda glfw-x11 glew glm cmake base-devel
```

### Packages needed:

- `cuda` - NVIDIA CUDA toolkit
- `glfw-x11` - OpenGL window library
- `glew` - OpenGL Extension Wrangler
- `glm` - OpenGL Mathematics
- `cmake` - Build system
- `base-devel` - C++ compiler and tools

## Building

### Quick build:

```bash
./build.sh
```

### Build and run:

```bash
./build.sh run
```

### Manual build:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./fluid_sim
```

## Controls

- **Left mouse drag**: Rotate camera around scene
- **ESC**: Exit simulation

## Parameters

You can modify simulation parameters in `src/main.cpp`:

```cpp
const int NX = 64, NY = 64, NZ = 64;  // Grid resolution
const float DX = 0.02f;                 // Cell size
const float DT = 0.016f;                // Time step (~60 FPS)
const float NU = 0.0001f;              // Viscosity
```

### Performance vs Quality:

| Grid Size | Memory  | Performance | Quality     |
| --------- | ------- | ----------- | ----------- |
| 32³       | ~1 MB   | >200 FPS    | Low detail  |
| 64³       | ~8 MB   | ~60 FPS     | Good        |
| 128³      | ~64 MB  | ~15 FPS     | High detail |
| 256³      | ~512 MB | ~2 FPS      | Very high   |

## Code Structure

```
fluid_sim/
├── CMakeLists.txt           # Build configuration
├── build.sh                 # Build script
├── README.md               # This file
└── src/
    ├── main.cpp            # Entry point, main loop
    ├── fluid_solver.cu     # CUDA fluid solver implementation
    ├── fluid_solver.cuh    # Fluid solver header
    ├── renderer.cpp        # OpenGL rendering
    └── renderer.h          # Renderer header
```

## How It Works

### 1. Grid Setup (FVM)

The domain is divided into a 3D grid of cells:

```
Each cell stores:
  - u[i][j][k] = x-velocity
  - v[i][j][k] = y-velocity
  - w[i][j][k] = z-velocity
  - p[i][j][k] = pressure
```

### 2. CUDA Kernels

**advectKernel**: Semi-Lagrangian advection

```cuda
- Trace particle backwards in time
- Interpolate velocity at that position
- Update current cell with interpolated value
```

**diffuseKernel**: Viscous diffusion

```cuda
- Jacobi iteration for implicit diffusion
- Laplacian stencil (7-point in 3D)
```

**projectKernel**: Enforce incompressibility

```cuda
- Compute divergence of velocity
- Solve Poisson equation for pressure
- Correct velocity: u = u* - ∇p
```

**updateParticlesKernel**: Move tracer particles

```cuda
- Sample velocity at particle position
- Integrate position: pos += vel * dt
- Bounce off boundaries
```

### 3. CUDA-OpenGL Interop

Particles are stored in a shared buffer:

- CUDA writes particle positions
- OpenGL reads and renders them
- Zero-copy transfer (same GPU memory!)

```cpp
// Register VBO with CUDA
cudaGraphicsGLRegisterBuffer(&resource, vbo, ...);

// Update from CUDA side
cudaGraphicsMapResources(...);
cudaMemcpy(vbo_ptr, particle_data, ...);
cudaGraphicsUnmapResources(...);

// Render from OpenGL side
glDrawArrays(GL_POINTS, 0, num_particles);
```

## Troubleshooting

### Build errors:

**"nvcc not found"**

```bash
# Add CUDA to PATH
export PATH=/opt/cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH
```

**"GLFW/GLEW not found"**

```bash
sudo pacman -S glfw-x11 glew
```

### Runtime errors:

**"Failed to create GLFW window"**

- Check if you're running on Wayland (use X11)
- Or install `glfw-wayland` instead of `glfw-x11`

**Low FPS:**

- Reduce grid resolution (NX, NY, NZ)
- Reduce number of particles
- Check GPU utilization with `nvidia-smi`

**Crashes:**

- Check CUDA compute capability matches CMakeLists.txt
- GTX 1080 Ti is compute capability 6.1 (Pascal)

## Performance Notes

On GTX 1080 Ti:

- **64³ grid**: ~60 FPS (real-time)
- **128³ grid**: ~15 FPS (high quality)
- **10,000 particles**: minimal overhead

Bottleneck is typically the **pressure Poisson solve** (30 Jacobi iterations).

## Future Enhancements

Possible improvements:

- [ ] Better initial conditions (vortex rings, etc.)
- [ ] Interactive fluid sources (mouse injection)
- [ ] Multigrid solver for pressure (faster)
- [ ] Smoke density visualization
- [ ] Vorticity confinement (maintain turbulence)
- [ ] Volume rendering instead of particles
- [ ] Export simulation data

## References

- **Navier-Stokes equations**: https://en.wikipedia.org/wiki/Navier–Stokes_equations
- **Finite Volume Method**: Anderson, "Computational Fluid Dynamics"
- **GPU Fluids**: Harris, "Fast Fluid Dynamics Simulation on the GPU"
- **CUDA-GL Interop**: NVIDIA CUDA Programming Guide
