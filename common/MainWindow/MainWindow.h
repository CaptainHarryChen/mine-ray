#pragma once

#include "gdt/math/AffineSpace.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

using namespace gdt;
using namespace std;

class MainWindow
{
    static MainWindow *instance;

protected:
    GLFWwindow *handle = nullptr;

public:
    MainWindow(const string &title);
    ~MainWindow();

    vec2i getMousePos() const;

    virtual void render(double deltaTime) {}
    virtual void draw() {}
    virtual void resize(const vec2i &newSize) {}
    virtual void keyEvent(int key, int action, int mods) {}
    virtual void mouseMotionEvent(const vec2i &newPos) {}
    virtual void mouseButtonEvent(int button, int action, int mods) {}

    void run();

private:
    static void glfw_error_callback(int error, const char *description)
    {
        fprintf(stderr, "Error: %s\n", description);
    }

    static void windowResize(GLFWwindow *window, int width, int height)
    {
        instance->resize(vec2i(width, height));
    }

    static void windowKeyEvent(GLFWwindow *window, int key, int scancode, int action, int mods)
    {
        instance->keyEvent(key, action, mods);
    }

    static void windowMouseMotionEvent(GLFWwindow *window, double x, double y)
    {
        instance->mouseMotionEvent(vec2i((int)x, (int)y));
    }

    static void windowMouseButtonEvent(GLFWwindow *window, int button, int action, int mods)
    {
        instance->mouseButtonEvent(button, action, mods);
    }
};
