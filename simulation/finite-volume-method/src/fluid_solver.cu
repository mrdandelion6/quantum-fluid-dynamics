#include "fluid_solver.cuh"
#include <cuda_gl_interop.h>
#include <math.h>
#include <stdio.h>

#define IX(i, j, k) ((i) + (nx) * (j) + (nx) * (ny) * (k))
#define CLAMP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

// CUDA kernel for advection (convection term)
__global__ void advectKernel(float *u, float *v, float *w, float *u_new,
                             float *v_new, float *w_new, int nx, int ny, int nz,
                             float dt, float dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i <= 0 || i >= nx - 1 || j <= 0 || j >= ny - 1 || k <= 0 || k >= nz - 1)
        return;

    int idx = i + nx * j + nx * ny * k;

    // Semi-Lagrangian advection (trace particle backwards)
    float x = i - dt * u[idx] / dx;
    float y = j - dt * v[idx] / dx;
    float z = k - dt * w[idx] / dx;

    // Clamp to grid boundaries
    x = fmaxf(0.5f, fminf(nx - 1.5f, x));
    y = fmaxf(0.5f, fminf(ny - 1.5f, y));
    z = fmaxf(0.5f, fminf(nz - 1.5f, z));

    // Trilinear interpolation
    int i0 = (int)x, i1 = i0 + 1;
    int j0 = (int)y, j1 = j0 + 1;
    int k0 = (int)z, k1 = k0 + 1;

    float sx = x - i0, sy = y - j0, sz = z - k0;
    float tx = 1.0f - sx, ty = 1.0f - sy, tz = 1.0f - sz;

    // Interpolate u
    u_new[idx] = tx * ty * tz * u[i0 + nx * j0 + nx * ny * k0] +
                 sx * ty * tz * u[i1 + nx * j0 + nx * ny * k0] +
                 tx * sy * tz * u[i0 + nx * j1 + nx * ny * k0] +
                 sx * sy * tz * u[i1 + nx * j1 + nx * ny * k0] +
                 tx * ty * sz * u[i0 + nx * j0 + nx * ny * k1] +
                 sx * ty * sz * u[i1 + nx * j0 + nx * ny * k1] +
                 tx * sy * sz * u[i0 + nx * j1 + nx * ny * k1] +
                 sx * sy * sz * u[i1 + nx * j1 + nx * ny * k1];

    // Interpolate v
    v_new[idx] = tx * ty * tz * v[i0 + nx * j0 + nx * ny * k0] +
                 sx * ty * tz * v[i1 + nx * j0 + nx * ny * k0] +
                 tx * sy * tz * v[i0 + nx * j1 + nx * ny * k0] +
                 sx * sy * tz * v[i1 + nx * j1 + nx * ny * k0] +
                 tx * ty * sz * v[i0 + nx * j0 + nx * ny * k1] +
                 sx * ty * sz * v[i1 + nx * j0 + nx * ny * k1] +
                 tx * sy * sz * v[i0 + nx * j1 + nx * ny * k1] +
                 sx * sy * sz * v[i1 + nx * j1 + nx * ny * k1];

    // Interpolate w
    w_new[idx] = tx * ty * tz * w[i0 + nx * j0 + nx * ny * k0] +
                 sx * ty * tz * w[i1 + nx * j0 + nx * ny * k0] +
                 tx * sy * tz * w[i0 + nx * j1 + nx * ny * k0] +
                 sx * sy * tz * w[i1 + nx * j1 + nx * ny * k0] +
                 tx * ty * sz * w[i0 + nx * j0 + nx * ny * k1] +
                 sx * ty * sz * w[i1 + nx * j0 + nx * ny * k1] +
                 tx * sy * sz * w[i0 + nx * j1 + nx * ny * k1] +
                 sx * sy * sz * w[i1 + nx * j1 + nx * ny * k1];
}

// CUDA kernel for diffusion (viscous term) - Jacobi iteration
__global__ void diffuseKernel(float *u, float *v, float *w, float *u_new,
                              float *v_new, float *w_new, int nx, int ny,
                              int nz, float alpha, float rbeta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i <= 0 || i >= nx - 1 || j <= 0 || j >= ny - 1 || k <= 0 || k >= nz - 1)
        return;

    int idx = i + nx * j + nx * ny * k;

    // Jacobi iteration for implicit diffusion
    u_new[idx] = (u[idx] + alpha * (u_new[(i - 1) + nx * j + nx * ny * k] +
                                    u_new[(i + 1) + nx * j + nx * ny * k] +
                                    u_new[i + nx * (j - 1) + nx * ny * k] +
                                    u_new[i + nx * (j + 1) + nx * ny * k] +
                                    u_new[i + nx * j + nx * ny * (k - 1)] +
                                    u_new[i + nx * j + nx * ny * (k + 1)])) *
                 rbeta;

    v_new[idx] = (v[idx] + alpha * (v_new[(i - 1) + nx * j + nx * ny * k] +
                                    v_new[(i + 1) + nx * j + nx * ny * k] +
                                    v_new[i + nx * (j - 1) + nx * ny * k] +
                                    v_new[i + nx * (j + 1) + nx * ny * k] +
                                    v_new[i + nx * j + nx * ny * (k - 1)] +
                                    v_new[i + nx * j + nx * ny * (k + 1)])) *
                 rbeta;

    w_new[idx] = (w[idx] + alpha * (w_new[(i - 1) + nx * j + nx * ny * k] +
                                    w_new[(i + 1) + nx * j + nx * ny * k] +
                                    w_new[i + nx * (j - 1) + nx * ny * k] +
                                    w_new[i + nx * (j + 1) + nx * ny * k] +
                                    w_new[i + nx * j + nx * ny * (k - 1)] +
                                    w_new[i + nx * j + nx * ny * (k + 1)])) *
                 rbeta;
}

// CUDA kernel for computing divergence
__global__ void computeDivergenceKernel(float *u, float *v, float *w,
                                        float *div, int nx, int ny, int nz,
                                        float dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i <= 0 || i >= nx - 1 || j <= 0 || j >= ny - 1 || k <= 0 || k >= nz - 1)
        return;

    int idx = i + nx * j + nx * ny * k;

    div[idx] =
        -0.5f * dx *
        (u[(i + 1) + nx * j + nx * ny * k] - u[(i - 1) + nx * j + nx * ny * k] +
         v[i + nx * (j + 1) + nx * ny * k] - v[i + nx * (j - 1) + nx * ny * k] +
         w[i + nx * j + nx * ny * (k + 1)] - w[i + nx * j + nx * ny * (k - 1)]);
}

// CUDA kernel for pressure solve (Jacobi iteration)
__global__ void solvePressureKernel(float *p, float *p_new, float *div, int nx,
                                    int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i <= 0 || i >= nx - 1 || j <= 0 || j >= ny - 1 || k <= 0 || k >= nz - 1)
        return;

    int idx = i + nx * j + nx * ny * k;

    p_new[idx] =
        (div[idx] + p[(i - 1) + nx * j + nx * ny * k] +
         p[(i + 1) + nx * j + nx * ny * k] + p[i + nx * (j - 1) + nx * ny * k] +
         p[i + nx * (j + 1) + nx * ny * k] + p[i + nx * j + nx * ny * (k - 1)] +
         p[i + nx * j + nx * ny * (k + 1)]) /
        6.0f;
}

// CUDA kernel for pressure projection
__global__ void projectKernel(float *u, float *v, float *w, float *p, int nx,
                              int ny, int nz, float dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i <= 0 || i >= nx - 1 || j <= 0 || j >= ny - 1 || k <= 0 || k >= nz - 1)
        return;

    int idx = i + nx * j + nx * ny * k;

    u[idx] -= 0.5f *
              (p[(i + 1) + nx * j + nx * ny * k] -
               p[(i - 1) + nx * j + nx * ny * k]) /
              dx;
    v[idx] -= 0.5f *
              (p[i + nx * (j + 1) + nx * ny * k] -
               p[i + nx * (j - 1) + nx * ny * k]) /
              dx;
    w[idx] -= 0.5f *
              (p[i + nx * j + nx * ny * (k + 1)] -
               p[i + nx * j + nx * ny * (k - 1)]) /
              dx;
}

// CUDA kernel for boundary conditions
__global__ void boundaryKernel(float *u, float *v, float *w, int nx, int ny,
                               int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz)
        return;

    // No-slip boundaries (walls)
    if (i == 0 || i == nx - 1) {
        int idx = i + nx * j + nx * ny * k;
        u[idx] = 0.0f;
        v[idx] = 0.0f;
        w[idx] = 0.0f;
    }
    if (j == 0 || j == ny - 1) {
        int idx = i + nx * j + nx * ny * k;
        u[idx] = 0.0f;
        v[idx] = 0.0f;
        w[idx] = 0.0f;
    }
    if (k == 0 || k == nz - 1) {
        int idx = i + nx * j + nx * ny * k;
        u[idx] = 0.0f;
        v[idx] = 0.0f;
        w[idx] = 0.0f;
    }
}

// NEW: CUDA kernel for continuous source (inject velocity every frame)
__global__ void addSourceKernel(float *u, float *v, float *w, int nx, int ny,
                                int nz, float dx, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz)
        return;

    float cx = nx * dx * 0.5f;
    float cy = ny * dx * 0.5f;

    // Only add source in bottom quarter
    if (k < nz / 4) {
        float x = i * dx;
        float y = j * dx;

        float dx_pos = x - cx;
        float dy_pos = y - cy;
        float r = sqrtf(dx_pos * dx_pos + dy_pos * dy_pos);

        if (r < 0.4f) {
            int idx = i + nx * j + nx * ny * k;
            float strength = 0.15f * expf(-r * r / 0.1f) * dt;

            // Add continuous upward velocity
            w[idx] += strength;
            // Add slight rotation
            u[idx] += -dy_pos * strength * 0.5f;
            v[idx] += dx_pos * strength * 0.5f;
        }
    }
}

// CUDA kernel for updating particle positions
__global__ void updateParticlesKernel(float *pos, float *vel, float *u,
                                      float *v, float *w, int num_particles,
                                      int nx, int ny, int nz, float dt,
                                      float dx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles)
        return;

    float px = pos[idx * 3 + 0];
    float py = pos[idx * 3 + 1];
    float pz = pos[idx * 3 + 2];

    // Sample velocity at particle position
    int i = (int)(px / dx);
    int j = (int)(py / dx);
    int k = (int)(pz / dx);

    i = max(0, min(nx - 2, i));
    j = max(0, min(ny - 2, j));
    k = max(0, min(nz - 2, k));

    int grid_idx = i + nx * j + nx * ny * k;

    float vx = u[grid_idx];
    float vy = v[grid_idx];
    float vz = w[grid_idx];

    // Update particle position
    px += vx * dt;
    py += vy * dt;
    pz += vz * dt;

    // Bounce off boundaries
    if (px < 0.0f) {
        px = 0.0f;
        vx = -vx;
    }
    if (px > (nx - 1) * dx) {
        px = (nx - 1) * dx;
        vx = -vx;
    }
    if (py < 0.0f) {
        py = 0.0f;
        vy = -vy;
    }
    if (py > (ny - 1) * dx) {
        py = (ny - 1) * dx;
        vy = -vy;
    }
    if (pz < 0.0f) {
        pz = 0.0f;
        vz = -vz;
    }
    if (pz > (nz - 1) * dx) {
        pz = (nz - 1) * dx;
        vz = -vz;
    }

    pos[idx * 3 + 0] = px;
    pos[idx * 3 + 1] = py;
    pos[idx * 3 + 2] = pz;

    vel[idx * 3 + 0] = vx;
    vel[idx * 3 + 1] = vy;
    vel[idx * 3 + 2] = vz;
}

// Constructor
FluidSolver::FluidSolver(int nx, int ny, int nz, float dx, float dt, float nu)
    : nx(nx), ny(ny), nz(nz), dx(dx), dy(dx), dz(dx), dt(dt), nu(nu),
      vbo_registered(false), cuda_vbo_resource(nullptr) {

    grid_size = nx * ny * nz;
    allocateMemory();
    reset();
}

// Destructor
FluidSolver::~FluidSolver() {
    if (vbo_registered) {
        unregisterGLBuffer();
    }
    freeMemory();
}

void FluidSolver::allocateMemory() {
    size_t bytes = grid_size * sizeof(float);

    cudaMalloc(&d_u, bytes);
    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_w, bytes);
    cudaMalloc(&d_u_new, bytes);
    cudaMalloc(&d_v_new, bytes);
    cudaMalloc(&d_w_new, bytes);
    cudaMalloc(&d_p, bytes);

    // Allocate particles for visualization
    num_particles = 10000;
    cudaMalloc(&d_particle_pos, num_particles * 3 * sizeof(float));
    cudaMalloc(&d_particle_vel, num_particles * 3 * sizeof(float));
}

void FluidSolver::freeMemory() {
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_u_new);
    cudaFree(d_v_new);
    cudaFree(d_w_new);
    cudaFree(d_p);
    cudaFree(d_particle_pos);
    cudaFree(d_particle_vel);
}

void FluidSolver::reset() {
    cudaMemset(d_u, 0, grid_size * sizeof(float));
    cudaMemset(d_v, 0, grid_size * sizeof(float));
    cudaMemset(d_w, 0, grid_size * sizeof(float));
    cudaMemset(d_p, 0, grid_size * sizeof(float));

    // Initialize particles at the bottom center where source will be
    float *h_pos = new float[num_particles * 3];
    float cx = nx * dx * 0.5f;
    float cy = ny * dx * 0.5f;

    for (int i = 0; i < num_particles; i++) {
        // Spawn particles in a circle at the bottom
        float r = (rand() / (float)RAND_MAX) * 0.3f;
        float theta = (rand() / (float)RAND_MAX) * 2.0f * 3.14159f;

        h_pos[i * 3 + 0] = cx + r * cosf(theta);
        h_pos[i * 3 + 1] = cy + r * sinf(theta);
        h_pos[i * 3 + 2] = (rand() / (float)RAND_MAX) * 0.2f; // Near bottom
    }
    cudaMemcpy(d_particle_pos, h_pos, num_particles * 3 * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemset(d_particle_vel, 0, num_particles * 3 * sizeof(float));
    delete[] h_pos;
}

void FluidSolver::step() {
    dim3 threads(8, 8, 8);
    dim3 blocks((nx + threads.x - 1) / threads.x,
                (ny + threads.y - 1) / threads.y,
                (nz + threads.z - 1) / threads.z);

    // 0. ADD SOURCE (inject velocity continuously every frame)
    addSourceKernel<<<blocks, threads>>>(d_u, d_v, d_w, nx, ny, nz, dx, dt);
    cudaDeviceSynchronize();

    // 1. Advection step
    advectKernel<<<blocks, threads>>>(d_u, d_v, d_w, d_u_new, d_v_new, d_w_new,
                                      nx, ny, nz, dt, dx);
    cudaDeviceSynchronize();

    // Swap pointers
    float *temp;
    temp = d_u;
    d_u = d_u_new;
    d_u_new = temp;
    temp = d_v;
    d_v = d_v_new;
    d_v_new = temp;
    temp = d_w;
    d_w = d_w_new;
    d_w_new = temp;

    // 2. Diffusion step (implicit - Jacobi iteration)
    float alpha = dt * nu / (dx * dx);
    float rbeta = 1.0f / (1.0f + 6.0f * alpha);

    for (int iter = 0; iter < 20; iter++) {
        diffuseKernel<<<blocks, threads>>>(d_u, d_v, d_w, d_u_new, d_v_new,
                                           d_w_new, nx, ny, nz, alpha, rbeta);
        cudaDeviceSynchronize();

        temp = d_u;
        d_u = d_u_new;
        d_u_new = temp;
        temp = d_v;
        d_v = d_v_new;
        d_v_new = temp;
        temp = d_w;
        d_w = d_w_new;
        d_w_new = temp;
    }

    // 3. Projection step (enforce incompressibility)
    project();

    // 4. Boundary conditions
    boundaryKernel<<<blocks, threads>>>(d_u, d_v, d_w, nx, ny, nz);
    cudaDeviceSynchronize();

    // 5. Update particle positions
    updateParticles();
}

void FluidSolver::project() {
    dim3 threads(8, 8, 8);
    dim3 blocks((nx + threads.x - 1) / threads.x,
                (ny + threads.y - 1) / threads.y,
                (nz + threads.z - 1) / threads.z);

    // Compute divergence
    computeDivergenceKernel<<<blocks, threads>>>(d_u, d_v, d_w, d_p, nx, ny, nz,
                                                 dx);
    cudaDeviceSynchronize();

    // Solve pressure Poisson equation (Jacobi iteration)
    cudaMemset(d_u_new, 0,
               grid_size *
                   sizeof(float)); // Use u_new as temporary for new pressure

    for (int iter = 0; iter < 30; iter++) {
        solvePressureKernel<<<blocks, threads>>>(d_p, d_u_new, d_p, nx, ny, nz);
        cudaDeviceSynchronize();

        float *temp = d_p;
        d_p = d_u_new;
        d_u_new = temp;
    }

    // Project velocities
    projectKernel<<<blocks, threads>>>(d_u, d_v, d_w, d_p, nx, ny, nz, dx);
    cudaDeviceSynchronize();
}

void FluidSolver::updateParticles() {
    int threads = 256;
    int blocks = (num_particles + threads - 1) / threads;

    updateParticlesKernel<<<blocks, threads>>>(d_particle_pos, d_particle_vel,
                                               d_u, d_v, d_w, num_particles, nx,
                                               ny, nz, dt, dx);
    cudaDeviceSynchronize();

    // RESPAWN particles that escaped
    float *h_pos = new float[num_particles * 3];
    cudaMemcpy(h_pos, d_particle_pos, num_particles * 3 * sizeof(float),
               cudaMemcpyDeviceToHost);

    float max_x = (nx - 1) * dx;
    float max_y = (ny - 1) * dx;
    float max_z = (nz - 1) * dx;
    float cx = nx * dx * 0.5f;
    float cy = ny * dx * 0.5f;

    for (int i = 0; i < num_particles; i++) {
        float x = h_pos[i * 3 + 0];
        float y = h_pos[i * 3 + 1];
        float z = h_pos[i * 3 + 2];

        // If particle reached top or escaped, respawn at bottom
        if (z > max_z * 0.9f || z < 0 || x < 0 || x > max_x || y < 0 ||
            y > max_y) {

            // Respawn near center bottom
            float r = (rand() / (float)RAND_MAX) * 0.3f;
            float theta = (rand() / (float)RAND_MAX) * 2.0f * 3.14159f;

            h_pos[i * 3 + 0] = cx + r * cosf(theta);
            h_pos[i * 3 + 1] = cy + r * sinf(theta);
            h_pos[i * 3 + 2] = 0.05f;
        }
    }

    cudaMemcpy(d_particle_pos, h_pos, num_particles * 3 * sizeof(float),
               cudaMemcpyHostToDevice);
    delete[] h_pos;
}

void FluidSolver::registerGLBuffer(unsigned int vbo) {
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
                                 cudaGraphicsMapFlagsWriteDiscard);
    vbo_registered = true;
}

void FluidSolver::unregisterGLBuffer() {
    if (vbo_registered) {
        cudaGraphicsUnregisterResource(cuda_vbo_resource);
        vbo_registered = false;
    }
}

void FluidSolver::updateGLBuffer() {
    if (!vbo_registered)
        return;

    // Map OpenGL buffer to CUDA
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);

    size_t num_bytes;
    float *d_vbo_ptr;
    cudaGraphicsResourceGetMappedPointer((void **)&d_vbo_ptr, &num_bytes,
                                         cuda_vbo_resource);

    // Copy particle positions to VBO
    cudaMemcpy(d_vbo_ptr, d_particle_pos, num_particles * 3 * sizeof(float),
               cudaMemcpyDeviceToDevice);

    // Unmap buffer
    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
}
