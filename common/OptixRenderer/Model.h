#pragma once

#include "gdt/math/AffineSpace.h"
#include <vector>

using namespace gdt;

/**
 * @brief 存储使用同一材质的一个模型
 * 将模型拆成一堆三角形，记录每个三角形的顶点下标，每个点记录位置、法向量、材质坐标。
 * 材质中记录了颜色diffuse，透明度trans，折射率ior
 */
struct TriangleMesh
{
	std::vector<vec3f> vertex;
	std::vector<vec3f> normal;
	std::vector<vec2f> texcoord;
	std::vector<vec3i> index;

	// material data:
	vec3f diffuse;
	vec3f trans;
	float ior;
	int diffuseTextureID{-1};
};

/**
 * @brief 简易光照模型
 * 
 */
struct QuadLight
{
	vec3f origin, du, dv, power;
};

/**
 * @brief 存储纹理
 */
struct Texture
{
	~Texture()
	{
		if (pixel)
			delete[] pixel;
	}

	uint32_t *pixel{nullptr};
	vec2i resolution{-1};
};

/**
 * @brief 整个地图的模型，包括所有三角形组成的结构和纹理
 */
struct Model
{
	~Model()
	{
		for (auto mesh : meshes)
			delete mesh;
		for (auto texture : textures)
			delete texture;
	}

	std::vector<TriangleMesh *> meshes;
	std::vector<Texture *> textures;
	//! bounding box of all vertices in the model
	box3f bounds;
};

/**
 * @brief 从文件里读取模型
 * 
 * @param objFile 模型文件路径
 * @return Model* 返回模型
 */
Model *loadOBJ(const std::string &objFile);
