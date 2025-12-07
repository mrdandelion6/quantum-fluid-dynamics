#include "fluid_solver.cuh"
#include "renderer.h"
#include <chrono>
#include <iostream>

int main() {
    // Simulation parameters
    const int NX = 64, NY = 64, NZ = 64;
    const float DX = 0.02f;   // Grid spacing
    const float DT = 0.016f;  // ~60 FPS
    const float NU = 0.0001f; // Kinematic viscosity

    std::cout << "Initializing 3D Fluid Simulation...\n";
    std::cout << "Grid: " << NX << "x" << NY << "x" << NZ << "\n";
    std::cout << "Total cells: " << (NX * NY * NZ) << "\n\n";

    // Create fluid solver
    FluidSolver solver(NX, NY, NZ, DX, DT, NU);

    // Create renderer
    Renderer renderer(1280, 720);
    if (!renderer.init()) {
        std::cerr << "Failed to initialize renderer\n";
        return -1;
    }

    // Register OpenGL buffer with CUDA
    solver.registerGLBuffer(renderer.getVBO());

    std::cout << "Starting simulation...\n";
    std::cout << "Controls:\n";
    std::cout << "  - Left mouse drag: Rotate camera\n";
    std::cout << "  - ESC: Exit\n\n";

    // Main loop
    int frame_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (!renderer.shouldClose()) {
        // Step fluid simulation
        solver.step();

        // Update OpenGL buffer with new particle positions
        solver.updateGLBuffer();

        // Render
        renderer.render(solver.getNumParticles());
        renderer.pollEvents();

        // Print FPS every second
        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        float elapsed =
            std::chrono::duration<float>(current_time - start_time).count();

        if (elapsed >= 1.0f) {
            float fps = frame_count / elapsed;
            std::cout << "FPS: " << fps
                      << " | Particles: " << solver.getNumParticles() << "\r"
                      << std::flush;
            frame_count = 0;
            start_time = current_time;
        }
    }

    std::cout << "\nSimulation ended.\n";
    return 0;
}
