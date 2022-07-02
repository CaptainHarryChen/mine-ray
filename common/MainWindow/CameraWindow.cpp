#include "CameraWindow.h"

CameraWindow::CameraWindow(const Camera &initCamera, const string &title) : MainWindow(title)
{
    camera = initCamera;
}

void CameraWindow::setMoveSpeed(float speed)
{
    moveSpeed = speed;
}

void CameraWindow::setRotateSpeed(float speed)
{
    rotateSpeed = speed;
}


void CameraWindow::mouseButtonEvent(int button, int action, int mods)
{
    bool pressed = (action == GLFW_PRESS);
    switch (button)
    {
    case GLFW_MOUSE_BUTTON_RIGHT:
        lastMousePos = getMousePos();
        rightButtonPressed = pressed;
        if (pressed)
            glfwSetInputMode(handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        else
            glfwSetInputMode(handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        break;
    }
}

void CameraWindow::mouseMotionEvent(const vec2i &newPos)
{
    if (rightButtonPressed)
    {
        vec2i delta = newPos - lastMousePos;
        lastMousePos = newPos;
        // printf("%d %d\n", delta.x, delta.y);
        float rad_u = -M_PI / 180.f * delta.x * rotateSpeed;
        float rad_v = -M_PI / 180.f * delta.y * rotateSpeed;
        linear3f frame = linear3f(normalize(cross(camera.forw, camera.up)), normalize(camera.up), normalize(camera.forw));
        frame = linear3f::rotate(frame.vy, rad_u) * linear3f::rotate(frame.vx, rad_v) * frame;

        if (fabsf(dot(frame.vz, vec3f(0, 1, 0))) > 1e-6f)
        {
            frame.vx = normalize(cross(vec3f(0, 1, 0), frame.vz));
            frame.vy = normalize(cross(frame.vz, frame.vx));
        }

        camera.up = frame.vy;
        camera.forw = frame.vz;
    }
}

void CameraWindow::keyEvent(int key, int action, int mods)
{
    bool pressed = (action != GLFW_RELEASE);
    switch (key)
    {
    case GLFW_KEY_W:
        movingForward = pressed;
        break;
    case GLFW_KEY_S:
        movingBack = pressed;
        break;
    case GLFW_KEY_A:
        movingLeft = pressed;
        break;
    case GLFW_KEY_D:
        movingRight = pressed;
        break;
    case GLFW_KEY_LEFT_CONTROL:
        movingDown = pressed;
        break;
    case GLFW_KEY_LEFT_SHIFT:
        movingUp = pressed;
        break;
    }
}

void CameraWindow::render(double deltaTime)
{
    MainWindow::render(deltaTime);
    vec3f velocity(0.0f);
    if (movingForward)
        velocity += normalize(camera.forw);
    if (movingBack)
        velocity -= normalize(camera.forw);
    if (movingUp)
        velocity += normalize(camera.up);
    if (movingDown)
        velocity -= normalize(camera.up);
    if (movingRight)
        velocity -= normalize(cross(camera.up, camera.forw));
    if (movingLeft)
        velocity += normalize(cross(camera.up, camera.forw));
    if (length(velocity) != 0.0f)
    {
        velocity = normalize(velocity) * moveSpeed;
        camera.pos += velocity * (float)deltaTime;
    }
}
