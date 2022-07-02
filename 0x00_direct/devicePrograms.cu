#include <optix_device.h>
#include <cuda_runtime.h>

#include "OptixRenderer/LaunchParams.h"

extern "C" __constant__ LaunchParams optixLaunchParams;

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

	if (dot(rayDir, Ng) > 0.f)
		Ng = -Ng;
	Ng = normalize(Ng);

	if (dot(Ng, Ns) < 0.f)
		Ns -= 2.f * dot(Ng, Ns) * Ng;
	Ns = normalize(Ns);

	vec3f diffuseColor = sbtData.color;
	if (sbtData.hasTexture && sbtData.texcoord)
	{
		const vec2f tc = (1.f - u - v) * sbtData.texcoord[index.x] + u * sbtData.texcoord[index.y] + v * sbtData.texcoord[index.z];

		vec4f fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
		diffuseColor *= (vec3f)fromTexture;
	}

	const vec3f surfPos = (1.f - u - v) * sbtData.vertex[index.x] + u * sbtData.vertex[index.y] + v * sbtData.vertex[index.z];
	const vec3f lightPos(0.0f, 1.98f, 0.0f);
	const vec3f lightDir = lightPos - surfPos;

	vec3f lightVisibility = 0.f;

	uint32_t u0, u1;
	packPointer(&lightVisibility, u0, u1);
	optixTrace(optixLaunchParams.traversable,
				surfPos + 1e-3f * Ng,
				lightDir,
				1e-3f,		// tmin
				1.f - 1e-3f, // tmax
				0.0f,		// rayTime
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
				SHADOW_RAY_TYPE, // SBT offset
				RAY_TYPE_COUNT,	// SBT stride
				SHADOW_RAY_TYPE, // missSBTIndex
				u0, u1);

	const float cosDN = 0.1f + .8f * fabsf(dot(rayDir, Ns));

	vec3f &prd = *(vec3f *)getPRD<vec3f>();
	prd = (.1f + (.2f + .8f * lightVisibility) * cosDN) * diffuseColor;
}

extern "C" __global__ void __anyhit__radiance()
{
}

extern "C" __global__ void __anyhit__shadow()
{ 
}

extern "C" __global__ void __miss__radiance()
{
	vec3f &prd = *(vec3f *)getPRD<vec3f>();
	prd = vec3f(0.f);
}

extern "C" __global__ void __miss__shadow()
{
	vec3f &prd = *(vec3f *)getPRD<vec3f>();
	prd = vec3f(1.f);
}

extern "C" __global__ void __raygen__renderFrame()
{
	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;

	const auto &camera = optixLaunchParams.camera;

	vec3f pixelColorPRD = vec3f(0.f);

	uint32_t u0, u1;
	packPointer(&pixelColorPRD, u0, u1);

	const vec2f screen(vec2f(ix + .5f, iy + .5f) / vec2f(optixLaunchParams.frame.size));

	vec3f rayDir = normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);
	
	optixTrace(optixLaunchParams.traversable,
				camera.position,
				rayDir,
				0.f,	  // tmin
				1e20f, // tmax
				0.0f,  // rayTime
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
				RADIANCE_RAY_TYPE,			  // SBT offset
				RAY_TYPE_COUNT,				  // SBT stride
				RADIANCE_RAY_TYPE,			  // missSBTIndex
				u0, u1);

	const int r = int(255.99f * pixelColorPRD.x);
	const int g = int(255.99f * pixelColorPRD.y);
	const int b = int(255.99f * pixelColorPRD.z);

	const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

	const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
	optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}
