#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class Renderer {
  public:
    Renderer(int width, int height);
    ~Renderer();

    bool init();
    void render(int num_particles);
    bool shouldClose();
    void pollEvents();

    GLuint getVBO() { return vbo; }

    void setViewMatrix(const glm::mat4 &view) { view_matrix = view; }
    void setProjectionMatrix(const glm::mat4 &proj) {
        projection_matrix = proj;
    }

  private:
    GLFWwindow *window;
    int width, height;

    // OpenGL objects
    GLuint vao, vbo;
    GLuint shader_program;

    // Matrices
    glm::mat4 view_matrix;
    glm::mat4 projection_matrix;
    glm::mat4 model_matrix;

    // Camera
    float camera_distance;
    float camera_angle_x, camera_angle_y;

    void createShaders();
    void setupBuffers(int max_particles);
    void handleInput();
};
