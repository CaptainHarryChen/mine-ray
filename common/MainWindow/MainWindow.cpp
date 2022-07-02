#include "MainWindow.h"

MainWindow *MainWindow::instance;

MainWindow::MainWindow(const string &title)
{
    if (instance)
    {
        fprintf(stderr, "Error: Only one window can be created! \n");
        exit(EXIT_FAILURE);
    }
    instance = this;

    glfwSetErrorCallback(glfw_error_callback);
    // glfwInitHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);

    if (!glfwInit())
        exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

    handle = glfwCreateWindow(1200, 800, title.c_str(), NULL, NULL);
    if (!handle)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwSetWindowUserPointer(handle, this);
    glfwMakeContextCurrent(handle);
    glfwSwapInterval(1);
}

MainWindow::~MainWindow()
{
    glfwDestroyWindow(handle);
    glfwTerminate();
}

void MainWindow::run()
{
    int width, height;
    glfwGetFramebufferSize(handle, &width, &height);
    resize(vec2i(width, height));

    glfwSetFramebufferSizeCallback(handle, windowResize);
    glfwSetKeyCallback(handle, windowKeyEvent);
    glfwSetCursorPosCallback(handle, windowMouseMotionEvent);
    glfwSetMouseButtonCallback(handle, windowMouseButtonEvent);

    double lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(handle))
    {
        double nowTime = glfwGetTime();
        double delta = nowTime - lastTime;
        lastTime = nowTime;
        render(delta);
        draw();

        glfwSwapBuffers(handle);
        glfwPollEvents();
    }
}

vec2i MainWindow::getMousePos() const
{
    double x, y;
    glfwGetCursorPos(handle, &x, &y);
    return vec2i((int)x, (int)y);
}