#pragma once

#include "MainWindow.h"

struct Camera
{
    vec3f pos;
    vec3f forw;
    vec3f up;
};

class CameraWindow: public MainWindow
{
    float moveSpeed = 2.5f;
    float rotateSpeed = 0.15f;

    vec2i lastMousePos;
    bool rightButtonPressed = false;
    bool movingForward = false;
    bool movingBack = false;
    bool movingLeft = false;
    bool movingRight = false;
    bool movingUp = false;
    bool movingDown = false;

public:
    Camera camera;

    CameraWindow(const Camera &initCamera, const string &title);
    
    void setMoveSpeed(float speed);
    void setRotateSpeed(float speed);

    virtual void mouseButtonEvent(int button, int action, int mods) override;
    virtual void mouseMotionEvent(const vec2i &delta) override;
    virtual void keyEvent(int key, int action, int mods) override;
    virtual void render(double deltaTime) override;
};
