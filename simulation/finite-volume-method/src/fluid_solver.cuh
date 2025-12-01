#ifndef FLUID_SOLVER_CUH
#define FLUID_SOLVER_CUH

#include <cuda_runtime.h>
// DO NOT include cuda_gl_interop.h here - will be included in .cu file after
// GLEW

class FluidSolver {
  public:
    // Constructor
    FluidSolver(int nx, int ny, int nz, float dx, float dt, float nu);

    // Destructor
    ~FluidSolver();

    // Simulation step
    void step();

    // Reset simulation
    void reset();

    // OpenGL interop
    void registerGLBuffer(unsigned int vbo);
    void unregisterGLBuffer();
    void updateGLBuffer();

    // Getters
    int getNumParticles() const { return num_particles; }
    int getNX() const { return nx; }
    int getNY() const { return ny; }
    int getNZ() const { return nz; }
    float getDX() const { return dx; }

  private:
    // Grid dimensions
    int nx, ny, nz;
    int grid_size;

    // Physical parameters
    float dx, dy, dz; // Grid spacing
    float dt;         // Time step
    float nu;         // Kinematic viscosity

    // Device memory for velocity fields
    float *d_u;     // x-velocity
    float *d_v;     // y-velocity
    float *d_w;     // z-velocity
    float *d_u_new; // Temporary for updates
    float *d_v_new;
    float *d_w_new;
    float *d_p; // Pressure

    // Particle data
    int num_particles;
    float *d_particle_pos; // Particle positions (x,y,z interleaved)
    float *d_particle_vel; // Particle velocities

    // OpenGL interop
    bool vbo_registered;
    struct cudaGraphicsResource
        *cuda_vbo_resource; // Forward declaration instead of include

    // Internal methods
    void allocateMemory();
    void freeMemory();
    void project();
    void updateParticles();
};

#endif // FLUID_SOLVER_CUH
