#include "OptixRenderer.h"

#include <optix_function_table_definition.h>

using namespace std;

// .cu代码编译出来的ptx代码
extern "C" char embedded_ptx_code[];

template <class T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) Record
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef Record<void *> RaygenRecord;
typedef Record<void *> MissRecord;
typedef Record<TriangleMeshSBTData> HitgroupRecord;

static void context_log_cb(unsigned int level, const char *tag, const char *message, void *)
{
	fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

OptixRenderer::OptixRenderer(const Model *model, const QuadLight *light) : model(model)
{
	initOptix();

	cout << "creating optix context ..." << endl;
	createContext();
	cout << "setting up module ..." << endl;
	createModule();

	cout << "creating raygen programs ..." << endl;
	createRaygenPrograms();
	cout << "creating miss programs ..." << endl;
	createMissPrograms();
	cout << "creating hitgroup programs ..." << endl;
	createHitgroupPrograms();

	launchParams.traversable = buildAccel();

	cout << "setting up optix pipeline ..." << endl;
	createPipeline();

	createTextures();
	createLight(light);
	cout << "building SBT ..." << endl;
	buildSBT();

	launchParamsBuffer.alloc(sizeof(launchParams));
	cout << "context, module, pipeline, etc, all set up ..." << endl;

	cout << GDT_TERMINAL_GREEN;
	cout << "Optix 7 Sample fully set up" << endl;
	cout << GDT_TERMINAL_DEFAULT;
}

/**
 * @brief 初始化Optix和cuda
 */
void OptixRenderer::initOptix()
{
	cout << "Initializing optix..." << endl;

	// 初始化cuda环境
	cudaFree(0);
	// 检查cuda设备
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if (numDevices == 0)
		throw runtime_error("No CUDA capable devices found!");
	cout << "Found " << numDevices << " CUDA devices" << endl;

	// 初始化Optix
	OPTIX_CHECK(optixInit());
	cout << GDT_TERMINAL_GREEN << "Successfully initialized optix" << GDT_TERMINAL_DEFAULT << endl;
}

/**
 * @brief 创建Optix上下文
 */
void OptixRenderer::createContext()
{
	const int deviceID = 0;
	CUDA_CHECK(SetDevice(deviceID));
	CUDA_CHECK(StreamCreate(&stream));

	cudaGetDeviceProperties(&deviceProps, deviceID);
	cout << "running on device: " << deviceProps.name << endl;

	CUresult cuRes = cuCtxGetCurrent(&cudaContext);
	if (cuRes != CUDA_SUCCESS)
		fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

	OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
	OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
}

/**
 * @brief 将.cu模块添加进来，设置相关参数（如LaunchParams）
 */
void OptixRenderer::createModule()
{
	moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.numPayloadValues = 2;
	pipelineCompileOptions.numAttributeValues = 2;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	// 传入 .cu 里的参数使用的变量名
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
	// 最多递归调用的光线追踪次数（似乎没用）
	pipelineLinkOptions.maxTraceDepth = 2;
	// .cu的ptx代码
	const string ptxCode = embedded_ptx_code;

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
										 &moduleCompileOptions,
										 &pipelineCompileOptions,
										 ptxCode.c_str(),
										 ptxCode.size(),
										 log, &sizeof_log,
										 &module));
	if (sizeof_log > 1)
		PRINT(log);
}

/**
 * @brief 指定.cu中的光线生成函数raygen
 *
 * 只从摄像机生成一种光线，即使用一个raygen函数
 */
void OptixRenderer::createRaygenPrograms()
{
	raygenPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc.raygen.module = module;
	pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(optixContext,
										&pgDesc,
										1,
										&pgOptions,
										log, &sizeof_log,
										&raygenPGs[0]));
	if (sizeof_log > 1)
		PRINT(log);
}

/**
 * @brief 指定.cu中的光线没有击中任何物体调用的函数miss
 *
 * 存在两种光线，需要两个miss函数
 * 1. 摄像机发出的光线没有碰到任何物体 radiance
 * 2. 摄像机碰到物体后，从该点向光源发出的光线（检查光源与该点直接是否有遮挡） shadow
 */
void OptixRenderer::createMissPrograms()
{
	missPGs.resize(RAY_TYPE_COUNT);

	char log[2048];
	size_t sizeof_log = sizeof(log);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	pgDesc.miss.module = module;

	pgDesc.miss.entryFunctionName = "__miss__radiance";
	OPTIX_CHECK(optixProgramGroupCreate(optixContext,
										&pgDesc,
										1,
										&pgOptions,
										log, &sizeof_log,
										&missPGs[RADIANCE_RAY_TYPE]));
	if (sizeof_log > 1)
		PRINT(log);

	pgDesc.miss.entryFunctionName = "__miss__shadow";
	OPTIX_CHECK(optixProgramGroupCreate(optixContext,
										&pgDesc,
										1,
										&pgOptions,
										log, &sizeof_log,
										&missPGs[SHADOW_RAY_TYPE]));
	if (sizeof_log > 1)
		PRINT(log);
}

/**
 * @brief 指定.cu中的光线击中物体时调用的函数
 *
 * 存在两种光线，每种光线都要closethit和anyhit两种函数
 * 1. 摄像机发出的光线没有碰到任何物体 radiance
 * 2. 摄像机碰到物体后，从该点向光源发出的光线（检查光源与该点直接是否有遮挡） shadow
 */
void OptixRenderer::createHitgroupPrograms()
{
	hitgroupPGs.resize(RAY_TYPE_COUNT);

	char log[2048];
	size_t sizeof_log = sizeof(log);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDesc.hitgroup.moduleCH = module;
	pgDesc.hitgroup.moduleAH = module;

	pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
	OPTIX_CHECK(optixProgramGroupCreate(optixContext,
										&pgDesc,
										1,
										&pgOptions,
										log, &sizeof_log,
										&hitgroupPGs[RADIANCE_RAY_TYPE]));
	if (sizeof_log > 1)
		PRINT(log);

	// shadow光线的碰撞判断没有实际作用，因为只使用miss来判断该点是否能被光源照到
	pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";
	OPTIX_CHECK(optixProgramGroupCreate(optixContext,
										&pgDesc,
										1,
										&pgOptions,
										log, &sizeof_log,
										&hitgroupPGs[SHADOW_RAY_TYPE]));
	if (sizeof_log > 1)
		PRINT(log);
}

/**
 * @brief 指定用于计算光线碰撞的物体即算法
 *
 * @return OptixTraversableHandle
 */
OptixTraversableHandle OptixRenderer::buildAccel()
{
	OptixTraversableHandle asHandle{0};

	// 导入模型和材质.
	int numMeshes = (int)model->meshes.size();
	vertexBuffer.resize(numMeshes);
	normalBuffer.resize(numMeshes);
	texcoordBuffer.resize(numMeshes);
	indexBuffer.resize(numMeshes);

	vector<OptixBuildInput> triangleInput(numMeshes);
	vector<CUdeviceptr> d_vertices(numMeshes);
	vector<CUdeviceptr> d_indices(numMeshes);
	vector<uint32_t> triangleInputFlags(numMeshes);

	for (int meshID = 0; meshID < numMeshes; meshID++)
	{
		TriangleMesh &mesh = *model->meshes[meshID];
		vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
		indexBuffer[meshID].alloc_and_upload(mesh.index);
		if (!mesh.normal.empty())
			normalBuffer[meshID].alloc_and_upload(mesh.normal);
		if (!mesh.texcoord.empty())
			texcoordBuffer[meshID].alloc_and_upload(mesh.texcoord);

		triangleInput[meshID] = {};
		triangleInput[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
		d_indices[meshID] = indexBuffer[meshID].d_pointer();

		triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(vec3f);
		triangleInput[meshID].triangleArray.numVertices = (int)mesh.vertex.size();
		triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

		triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(vec3i);
		triangleInput[meshID].triangleArray.numIndexTriplets = (int)mesh.index.size();
		triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

		triangleInputFlags[meshID] = 0;

		triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
		triangleInput[meshID].triangleArray.numSbtRecords = 1;
		triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
		triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
		triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
	}

	// 计算显存使用量.
	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accelOptions.motionOptions.numKeys = 1;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blasBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
											 &accelOptions,
											 triangleInput.data(),
											 (int)numMeshes, // num_build_inputs
											 &blasBufferSizes));

	// 开好现存空间.
	CUDABuffer compactedSizeBuffer;
	compactedSizeBuffer.alloc(sizeof(uint64_t));

	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactedSizeBuffer.d_pointer();

	// 将模型放入，构建相交算法.
	CUDABuffer tempBuffer;
	tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

	CUDABuffer outputBuffer;
	outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

	OPTIX_CHECK(optixAccelBuild(optixContext,
								/* stream */ 0,
								&accelOptions,
								triangleInput.data(),
								(int)numMeshes,
								tempBuffer.d_pointer(),
								tempBuffer.sizeInBytes,

								outputBuffer.d_pointer(),
								outputBuffer.sizeInBytes,

								&asHandle,

								&emitDesc, 1));
	CUDA_SYNC_CHECK();

	uint64_t compactedSize;
	compactedSizeBuffer.download(&compactedSize, 1);

	asBuffer.alloc(compactedSize);
	OPTIX_CHECK(optixAccelCompact(optixContext,
								  /*stream:*/ 0,
								  asHandle,
								  asBuffer.d_pointer(),
								  asBuffer.sizeInBytes,
								  &asHandle));
	CUDA_SYNC_CHECK();

	outputBuffer.free();
	tempBuffer.free();
	compactedSizeBuffer.free();

	return asHandle;
}

/**
 * @brief 创建光线追踪算法的流水线，将各种.cu里的函数组合
 */
void OptixRenderer::createPipeline()
{
	vector<OptixProgramGroup> programGroups;
	for (auto pg : raygenPGs)
		programGroups.push_back(pg);
	for (auto pg : hitgroupPGs)
		programGroups.push_back(pg);
	for (auto pg : missPGs)
		programGroups.push_back(pg);

	char log[2048];
	size_t sizeof_log = sizeof(log);
	PING;
	PRINT(programGroups.size());
	OPTIX_CHECK(optixPipelineCreate(optixContext,
									&pipelineCompileOptions,
									&pipelineLinkOptions,
									programGroups.data(),
									(int)programGroups.size(),
									log, &sizeof_log,
									&pipeline));
	if (sizeof_log > 1)
		PRINT(log);

	// 设置栈空间大小 （删去没有问题）
	OptixStackSizes stack_sizes = {};
	for (auto pg : raygenPGs)
		OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes));
	for (auto pg : hitgroupPGs)
		OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes));
	for (auto pg : missPGs)
		OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes));

	uint32_t max_trace_depth = 2;
	uint32_t max_cc_depth = 0;
	uint32_t max_dc_depth = 0;
	uint32_t direct_callable_stack_size_from_traversal;
	uint32_t direct_callable_stack_size_from_state;
	uint32_t continuation_stack_size;
	OPTIX_CHECK(optixUtilComputeStackSizes(
		&stack_sizes,
		max_trace_depth,
		max_cc_depth,
		max_dc_depth,
		&direct_callable_stack_size_from_traversal,
		&direct_callable_stack_size_from_state,
		&continuation_stack_size));

	const uint32_t max_traversal_depth = 1;
	OPTIX_CHECK(optixPipelineSetStackSize(
		pipeline,
		direct_callable_stack_size_from_traversal,
		direct_callable_stack_size_from_state,
		continuation_stack_size,
		max_traversal_depth));
}

/**
 * @brief 加载纹理
 */
void OptixRenderer::createTextures()
{
	int numTextures = (int)model->textures.size();

	textureArrays.resize(numTextures);
	textureObjects.resize(numTextures);

	for (int textureID = 0; textureID < numTextures; textureID++)
	{
		auto texture = model->textures[textureID];

		cudaResourceDesc res_desc = {};

		cudaChannelFormatDesc channel_desc;
		int32_t width = texture->resolution.x;
		int32_t height = texture->resolution.y;
		int32_t numComponents = 4;
		int32_t pitch = width * numComponents * sizeof(uint8_t);
		channel_desc = cudaCreateChannelDesc<uchar4>();

		cudaArray_t &pixelArray = textureArrays[textureID];
		CUDA_CHECK(MallocArray(&pixelArray,
							   &channel_desc,
							   width, height));

		CUDA_CHECK(Memcpy2DToArray(pixelArray,
								   /* offset */ 0, 0,
								   texture->pixel,
								   pitch, pitch, height,
								   cudaMemcpyHostToDevice));

		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = pixelArray;

		cudaTextureDesc tex_desc = {};
		tex_desc.addressMode[0] = cudaAddressModeWrap;
		tex_desc.addressMode[1] = cudaAddressModeWrap;
		tex_desc.filterMode = cudaFilterModeLinear;
		tex_desc.readMode = cudaReadModeNormalizedFloat;
		tex_desc.normalizedCoords = 1;
		tex_desc.maxAnisotropy = 1;
		tex_desc.maxMipmapLevelClamp = 99;
		tex_desc.minMipmapLevelClamp = 0;
		tex_desc.mipmapFilterMode = cudaFilterModePoint;
		tex_desc.borderColor[0] = 1.0f;
		tex_desc.sRGB = 0;

		// Create texture object
		cudaTextureObject_t cuda_tex = 0;
		CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
		textureObjects[textureID] = cuda_tex;
	}
}

void OptixRenderer::createLight(const QuadLight *light)
{
	if (!light)
		return;
	launchParams.light.origin = light->origin;
	launchParams.light.du = light->du;
	launchParams.light.dv = light->dv;
	launchParams.light.power = light->power;
}

/**
 * @brief 构建SBT着色器绑定列表(shader binding table)
 */
void OptixRenderer::buildSBT()
{
	// 光线生成阶段不需要多余SBT记录信息，留空nullptr
	vector<RaygenRecord> raygenRecords;
	for (int i = 0; i < raygenPGs.size(); i++)
	{
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
		rec.data = nullptr;
		raygenRecords.push_back(rec);
	}
	raygenRecordsBuffer.alloc_and_upload(raygenRecords);
	sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

	// 光线不击中物体函数不需要多余SBT记录信息，留空nullptr.
	vector<MissRecord> missRecords;
	for (int i = 0; i < missPGs.size(); i++)
	{
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
		rec.data = nullptr; /* for now ... */
		missRecords.push_back(rec);
	}
	missRecordsBuffer.alloc_and_upload(missRecords);
	sbt.missRecordBase = missRecordsBuffer.d_pointer();
	sbt.missRecordStrideInBytes = sizeof(MissRecord);
	sbt.missRecordCount = (int)missRecords.size();

	// 将模型即材质数据放入sbt，使得光线击中物体时，能够读取物体信息.
	int numObjects = (int)model->meshes.size();
	vector<HitgroupRecord> hitgroupRecords;
	for (int meshID = 0; meshID < numObjects; meshID++)
	{
		for (int rayID = 0; rayID < RAY_TYPE_COUNT; rayID++)
		{
			auto mesh = model->meshes[meshID];

			HitgroupRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[rayID], &rec));
			rec.data.color = mesh->diffuse;
			rec.data.trans = mesh->trans;
			rec.data.ior = mesh->ior;
			if (mesh->diffuseTextureID >= 0)
			{
				rec.data.hasTexture = true;
				rec.data.texture = textureObjects[mesh->diffuseTextureID];
			}
			else
			{
				rec.data.hasTexture = false;
			}
			rec.data.index = (vec3i *)indexBuffer[meshID].d_pointer();
			rec.data.vertex = (vec3f *)vertexBuffer[meshID].d_pointer();
			rec.data.normal = (vec3f *)normalBuffer[meshID].d_pointer();
			rec.data.texcoord = (vec2f *)texcoordBuffer[meshID].d_pointer();
			hitgroupRecords.push_back(rec);
		}
	}
	hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
	sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
	sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

/**
 * @brief 使用光追渲染一帧
 */
void OptixRenderer::render()
{
	if (launchParams.frame.size.x == 0)
		return;

	launchParamsBuffer.upload(&launchParams, 1);
	launchParams.frame.accumID++;
	OPTIX_CHECK(optixLaunch(pipeline, stream,
							launchParamsBuffer.d_pointer(),
							launchParamsBuffer.sizeInBytes,
							&sbt,
							launchParams.frame.size.x,
							launchParams.frame.size.y,
							1));
	CUDA_SYNC_CHECK();
}

/**
 * @brief 设置新的摄像机，并计算相机的三维坐标轴方向，传入optix里
 *
 * @param camera
 */
void OptixRenderer::setCamera(const Camera &camera)
{
	lastSetCamera = camera;
	launchParams.camera.position = camera.pos;
	launchParams.camera.direction = normalize(camera.forw);
	const float cosFovy = 0.66f;
	const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);
	launchParams.camera.horizontal = cosFovy * aspect * normalize(cross(launchParams.camera.direction, camera.up));
	launchParams.camera.vertical = cosFovy * normalize(cross(launchParams.camera.horizontal,
															 launchParams.camera.direction));
}

/**
 * @brief 设置窗口的大小
 *
 * @param newSize
 */
void OptixRenderer::resize(const vec2i &newSize)
{
	if (newSize.x == 0 || newSize.y == 0)
		return;

	colorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));
	launchParams.frame.size = newSize;
	launchParams.frame.colorBuffer = (uint32_t *)colorBuffer.d_pointer();

	setCamera(lastSetCamera);
}

/**
 * @brief 将光追算法计算的结果像素下载进cpu内存（然后使用opengl绘制在屏幕）
 *
 * @param h_pixels
 */
void OptixRenderer::downloadPixels(uint32_t h_pixels[])
{
	colorBuffer.download(h_pixels,
						 launchParams.frame.size.x * launchParams.frame.size.y);
}
