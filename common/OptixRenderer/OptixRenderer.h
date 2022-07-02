#pragma once

#include "CUDABuffer.h"
#include "Model.h"
#include "MainWindow/CameraWindow.h"
#include "LaunchParams.h"

class OptixRenderer
{
public:
	OptixRenderer(const Model *model, const QuadLight *light);

	void render();
	void resize(const vec2i &newSize);
	void downloadPixels(uint32_t h_pixels[]);
	void setCamera(const Camera &camera);

protected:

	void initOptix();

	void createContext();
	void createModule();
	void createRaygenPrograms();
	void createMissPrograms();
	void createHitgroupPrograms();
	void createPipeline();
	void buildSBT();
	OptixTraversableHandle buildAccel();
	void createTextures();
	void createLight(const QuadLight *light);

protected:

	CUcontext cudaContext;
	CUstream stream;
	cudaDeviceProp deviceProps;

	OptixDeviceContext optixContext;

	OptixPipeline pipeline;
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	OptixPipelineLinkOptions pipelineLinkOptions = {};

	OptixModule module;
	OptixModuleCompileOptions moduleCompileOptions = {};

	std::vector<OptixProgramGroup> raygenPGs;
	CUDABuffer raygenRecordsBuffer;
	std::vector<OptixProgramGroup> missPGs;
	CUDABuffer missRecordsBuffer;
	std::vector<OptixProgramGroup> hitgroupPGs;
	CUDABuffer hitgroupRecordsBuffer;
	OptixShaderBindingTable sbt = {};

	LaunchParams launchParams;
	CUDABuffer launchParamsBuffer;

	CUDABuffer colorBuffer;

	Camera lastSetCamera;

	const Model *model;

	std::vector<CUDABuffer> vertexBuffer;
	std::vector<CUDABuffer> normalBuffer;
	std::vector<CUDABuffer> texcoordBuffer;
	std::vector<CUDABuffer> indexBuffer;

	CUDABuffer asBuffer;

	std::vector<cudaArray_t> textureArrays;
	std::vector<cudaTextureObject_t> textureObjects;
};
