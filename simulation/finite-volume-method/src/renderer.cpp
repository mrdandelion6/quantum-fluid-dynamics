#include "renderer.h"
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <sstream>

static std::string loadShader(const char *filepath) {
    std::ifstream file(filepath);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

static void error_callback(int error, const char *description) {
    fprintf(stderr, "GLFW Error: %s\n", description);
}

Renderer::Renderer(int width, int height)
    : width(width), height(height), window(nullptr), camera_distance(5.0f),
      camera_angle_x(0.0f), camera_angle_y(0.0f) {}

Renderer::~Renderer() {
    if (window) {
        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &vbo);
        glDeleteProgram(shader_program);
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

bool Renderer::init() {
    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    window = glfwCreateWindow(width, height, "3D Fluid Simulation", nullptr,
                              nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // VSync

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        return false;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glPointSize(3.0f);

    createShaders();
    setupBuffers(10000); // Max particles

    // Setup matrices
    projection_matrix = glm::perspective(glm::radians(45.0f),
                                         (float)width / height, 0.1f, 100.0f);
    model_matrix = glm::mat4(1.0f);

    return true;
}

void Renderer::createShaders() {
    // Vertex shader
    const char *vertex_shader_src = R"(
        #version 430 core
        layout (location = 0) in vec3 position;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        out vec3 frag_pos;
        
        void main() {
            frag_pos = position;
            gl_Position = projection * view * model * vec4(position, 1.0);
            gl_PointSize = 5.0;
        }
    )";

    // Fragment shader
    const char *fragment_shader_src = R"(
        #version 430 core
        in vec3 frag_pos;
        out vec4 FragColor;
        
        void main() {
            // Color based on position
            vec3 color = normalize(frag_pos) * 0.5 + 0.5;
            FragColor = vec4(color, 0.8);
        }
    )";

    // Compile vertex shader
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_src, nullptr);
    glCompileShader(vertex_shader);

    GLint success;
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(vertex_shader, 512, nullptr, log);
        std::cerr << "Vertex shader compilation failed:\n" << log << std::endl;
    }

    // Compile fragment shader
    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_src, nullptr);
    glCompileShader(fragment_shader);

    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(fragment_shader, 512, nullptr, log);
        std::cerr << "Fragment shader compilation failed:\n"
                  << log << std::endl;
    }

    // Link shaders
    shader_program = glCreateProgram();
    glAttachShader(shader_program, vertex_shader);
    glAttachShader(shader_program, fragment_shader);
    glLinkProgram(shader_program);

    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(shader_program, 512, nullptr, log);
        std::cerr << "Shader linking failed:\n" << log << std::endl;
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
}

void Renderer::setupBuffers(int max_particles) {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // Allocate buffer
    glBufferData(GL_ARRAY_BUFFER, max_particles * 3 * sizeof(float), nullptr,
                 GL_DYNAMIC_DRAW);

    // Setup vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                          (void *)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
}

void Renderer::handleInput() {
    static double last_x = 0.0, last_y = 0.0;
    static bool first_mouse = true;

    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        if (first_mouse) {
            last_x = xpos;
            last_y = ypos;
            first_mouse = false;
        }

        float xoffset = (xpos - last_x) * 0.5f;
        float yoffset = (last_y - ypos) * 0.5f;

        camera_angle_x += yoffset;
        camera_angle_y += xoffset;

        camera_angle_x = glm::clamp(camera_angle_x, -89.0f, 89.0f);
    } else {
        first_mouse = true;
    }

    // Zoom with scroll
    static double scroll_offset = 0.0;
    // (You'd need to set up a scroll callback in GLFW)

    last_x = xpos;
    last_y = ypos;

    // Update view matrix
    glm::vec3 camera_pos =
        glm::vec3(camera_distance * cos(glm::radians(camera_angle_y)) *
                      cos(glm::radians(camera_angle_x)),
                  camera_distance * sin(glm::radians(camera_angle_x)),
                  camera_distance * sin(glm::radians(camera_angle_y)) *
                      cos(glm::radians(camera_angle_x)));

    view_matrix =
        glm::lookAt(camera_pos, glm::vec3(0.5f), glm::vec3(0.0f, 1.0f, 0.0f));
}

void Renderer::render(int num_particles) {
    handleInput();

    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shader_program);

    // Set uniforms
    GLuint model_loc = glGetUniformLocation(shader_program, "model");
    GLuint view_loc = glGetUniformLocation(shader_program, "view");
    GLuint proj_loc = glGetUniformLocation(shader_program, "projection");

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm::value_ptr(model_matrix));
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm::value_ptr(view_matrix));
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE,
                       glm::value_ptr(projection_matrix));

    // Draw particles
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, num_particles);
    glBindVertexArray(0);

    glfwSwapBuffers(window);
}

bool Renderer::shouldClose() { return glfwWindowShouldClose(window); }

void Renderer::pollEvents() {
    glfwPollEvents();

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}
