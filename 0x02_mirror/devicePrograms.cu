#include <optix_device.h>
#include <cuda_runtime.h>

#include "optixRenderer/LaunchParams.h"
#include "gdt/random/random.h"

#define LIGHT_MASK 1
#define NUM_LIGHT_SAMPLES 1
#define NUM_PIXEL_SAMPLES 5

typedef gdt::LCG<16> Random;

extern "C" __constant__ LaunchParams optixLaunchParams;

struct PRD
{
	bool done;
	Random random;
	vec3f origin;
	vec3f direction;
	vec3f pixelColor;
	vec3f attenuation;
};

static __forceinline__ __device__ void *unpackPointer(uint32_t i0, uint32_t i1)
{
	const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
	void *ptr = reinterpret_cast<void *>(uptr);
	return ptr;
}

static __forceinline__ __device__ void packPointer(void *ptr, uint32_t &i0, uint32_t &i1)
{
	const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

template <typename T>
static __forceinline__ __device__ T *getPRD()
{
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return reinterpret_cast<T *>(unpackPointer(u0, u1));
}

extern "C" __global__ void __closesthit__shadow()
{
}

extern "C" __global__ void __closesthit__radiance()
{
	const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData *)optixGetSbtDataPointer();
	PRD &prd = *getPRD<PRD>();

	const int primID = optixGetPrimitiveIndex();
	const vec3i index = sbtData.index[primID];
	const float u = optixGetTriangleBarycentrics().x;
	const float v = optixGetTriangleBarycentrics().y;

	const vec3f &A = sbtData.vertex[index.x];
	const vec3f &B = sbtData.vertex[index.y];
	const vec3f &C = sbtData.vertex[index.z];
	vec3f Ng = cross(B - A, C - A);
	vec3f Ns = (sbtData.normal)
				? ((1.f - u - v) * sbtData.normal[index.x] + u * sbtData.normal[index.y] + v * sbtData.normal[index.z])
				: Ng;

	const vec3f rayDir = optixGetWorldRayDirection();

	Ng = normalize(Ng);
	Ns = normalize(Ns);

	vec3f diffuseColor = sbtData.color;
	if (sbtData.hasTexture && sbtData.texcoord)
	{
		const vec2f tc = (1.f - u - v) * sbtData.texcoord[index.x] + u * sbtData.texcoord[index.y] + v * sbtData.texcoord[index.z];

		vec4f fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
		diffuseColor *= (vec3f)fromTexture;
	}

	vec3f pixelColor = (0.1f + 0.2f * fabsf(dot(Ns, rayDir))) * diffuseColor;

	const vec3f surfPos = (1.f - u - v) * sbtData.vertex[index.x] + u * sbtData.vertex[index.y] + v * sbtData.vertex[index.z];
	const vec3f OriPos = optixGetWorldRayOrigin();
	const vec3f InterPos = OriPos + optixGetRayTmax() * rayDir;

	prd.origin = InterPos;
	prd.direction = rayDir - 2.f * dot(rayDir, Ns) * Ns;
	prd.direction = normalize(prd.direction);

	const int numLightSamples = NUM_LIGHT_SAMPLES;
	for (int lightSampleID = 0; lightSampleID < numLightSamples; lightSampleID++)
	{
		const vec3f lightPos = optixLaunchParams.light.origin + prd.random() * optixLaunchParams.light.du + prd.random() * optixLaunchParams.light.dv;
		vec3f lightDir = lightPos - surfPos;
		float lightDist = gdt::length(lightDir);
		lightDir = normalize(lightDir);

		const float NdotL = dot(lightDir, Ns);
		if (NdotL >= 0.f)
		{
			vec3f lightVisibility = 0.f;
			uint32_t u0, u1;
			packPointer(&lightVisibility, u0, u1);
			optixTrace(optixLaunchParams.traversable,
						surfPos + 1e-3f * Ns,
						lightDir,
						1e-3f,					  // tmin
						lightDist * (1.f - 1e-3f), // tmax
						0.0f,					  // rayTime
						OptixVisibilityMask(LIGHT_MASK),
						OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
						SHADOW_RAY_TYPE, // SBT offset
						RAY_TYPE_COUNT,	// SBT stride
						SHADOW_RAY_TYPE, // missSBTIndex
						u0, u1);
			pixelColor += lightVisibility * optixLaunchParams.light.power * diffuseColor * (NdotL / (lightDist * lightDist * numLightSamples));
		}
	}

	prd.pixelColor = pixelColor;
}

extern "C" __global__ void __anyhit__radiance()
{
}

extern "C" __global__ void __anyhit__shadow()
{
}

extern "C" __global__ void __miss__radiance()
{
	PRD &prd = *getPRD<PRD>();
	prd.pixelColor = vec3f(0.f);
	prd.done = true;
}

extern "C" __global__ void __miss__shadow()
{
	vec3f &prd = *(vec3f *)getPRD<vec3f>();
	prd = vec3f(0.3f);
}

extern "C" __global__ void __raygen__renderFrame()
{
	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;
	const int accumID = optixLaunchParams.frame.accumID;
	const auto &camera = optixLaunchParams.camera;

	int numPixelSamples = NUM_PIXEL_SAMPLES;

	vec3f pixelColor = 0.f;
	for (int sampleID = 0; sampleID < numPixelSamples; sampleID++)
	{
		PRD prd;
		prd.random.init(ix + accumID * optixLaunchParams.frame.size.x,
						iy + accumID * optixLaunchParams.frame.size.y);
		prd.pixelColor = vec3f(0.f);
		prd.done = false;
		prd.origin = camera.direction;

		uint32_t u0, u1;
		packPointer(&prd, u0, u1);

		const vec2f screen(vec2f(ix + prd.random(), iy + prd.random()) / vec2f(optixLaunchParams.frame.size));

		vec3f rayDir = normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);
		prd.direction = rayDir;

		vec3f ray_ori = camera.position;
		vec3f ray_dir = rayDir;
		int depth = 0;
		for( ;; ){
			optixTrace(optixLaunchParams.traversable,
					ray_ori,
					ray_dir,
					0.01f,	  // tmin
					1e16f, // tmax
					0.0f,  // rayTime
					OptixVisibilityMask(LIGHT_MASK),
					OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
					RADIANCE_RAY_TYPE,			  // SBT offset
					RAY_TYPE_COUNT,				  // SBT stride
					RADIANCE_RAY_TYPE,			  // missSBTIndex
					u0, u1);
			pixelColor += prd.pixelColor;
			if(prd.done || depth >= 10)
				break;
			ray_ori = prd.origin;
			ray_dir = prd.direction;
			++depth;
		}
	}

	const int r = int(255.99f * min(pixelColor.x / numPixelSamples, 1.f));
	const int g = int(255.99f * min(pixelColor.y / numPixelSamples, 1.f));
	const int b = int(255.99f * min(pixelColor.z / numPixelSamples, 1.f));

	const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

	const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
	optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}
