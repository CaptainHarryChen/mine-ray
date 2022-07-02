#include "optixRenderer/OptixRenderer.h"
#include "mainWindow/CameraWindow.h"
#include <GL/gl.h>

struct SampleWindow : public CameraWindow
{
	SampleWindow(const std::string &title,
					const Model *model,
					const Camera &camera,
					const QuadLight &light)
		: CameraWindow(camera, title),
			sample(model, &light)
	{
		sample.setCamera(camera);
	}

	virtual void render(double deltaTime) override
	{
		CameraWindow::render(deltaTime);
		sample.setCamera(camera);
		sample.render();
	}

	virtual void draw() override
	{
		sample.downloadPixels(pixels.data());
		if (fbTexture == 0)
			glGenTextures(1, &fbTexture);

		glBindTexture(GL_TEXTURE_2D, fbTexture);
		GLenum texFormat = GL_RGBA;
		GLenum texelType = GL_UNSIGNED_BYTE;
		glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
						texelType, pixels.data());

		glDisable(GL_LIGHTING);
		glColor3f(1, 1, 1);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, fbTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glDisable(GL_DEPTH_TEST);

		glViewport(0, 0, fbSize.x, fbSize.y);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

		glBegin(GL_QUADS);
		{
			glTexCoord2f(0.f, 0.f);
			glVertex3f(0.f, 0.f, 0.f);

			glTexCoord2f(0.f, 1.f);
			glVertex3f(0.f, (float)fbSize.y, 0.f);

			glTexCoord2f(1.f, 1.f);
			glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

			glTexCoord2f(1.f, 0.f);
			glVertex3f((float)fbSize.x, 0.f, 0.f);
		}
		glEnd();
	}

	virtual void resize(const vec2i &newSize)
	{
		fbSize = newSize;
		sample.resize(newSize);
		pixels.resize(newSize.x * newSize.y);
	}

	vec2i fbSize;
	GLuint fbTexture{0};
	OptixRenderer sample;
	std::vector<uint32_t> pixels;
};

int main(int ac, char **av)
{
	std::string mod0 = "../../models/cbox/CornellBox-Original.obj";
	std::string mod1 = "../../models/cbox_empty/CornellBox-Empty-RG.obj";
	std::string mod2 = "../../models/cbox_glossy/CornellBox-Glossy.obj";
	std::string mod3 = "../../models/cbox_mirror/CornellBox-Mirror.obj";
	std::string mod4 = "../../models/cbox_sphere/CornellBox-Sphere.obj";
	std::string mod5 = "../../models/cbox_water/CornellBox-Water.obj";
	std::string mod6 = "../../models/trans/trans.obj";
	try
	{
		Model *model = loadOBJ(mod4);
		Camera camera = {/*from*/ vec3f(0.0f, 0.0f, 5.0f),
							/* at */ model->bounds.center() - vec3f(0, 0, 5),
							/* up */ vec3f(0.f, 1.f, 0.f)};


		const float light_height = 1.56f;
		const float light_size = 0.2f;
		const float light_power = 2.5f;
		const float light_width = 1.5f;
		QuadLight light = {/* origin */ vec3f(0 - light_size, light_height, -light_size),
							/* edge 1 */ vec3f(light_width * light_size, 0, 0),
							/* edge 2 */ vec3f(0, 0, light_width * light_size),
							/* power */ vec3f(light_power)};

		SampleWindow *window = new SampleWindow("Mirror Example",
												model, camera, light);
		window->setMoveSpeed(2.5f);
		window->run();
	}
	catch (std::runtime_error &e)
	{
		std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
					<< GDT_TERMINAL_DEFAULT << std::endl;
		std::cout << "Load Model Failed!" << std::endl;
		exit(1);
	}
	return 0;
}
