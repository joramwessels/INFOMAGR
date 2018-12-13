#pragma once

struct AABB		// 6*4 = 24 bytes
{
	float xmin;
	float xmax;
	float ymin;
	float ymax;
	float zmin;
	float zmax;

	AABB()
	{
	}

	AABB(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
	{
		this->xmin = xmin;
		this->xmax = xmax;
		this->ymin = ymin;
		this->ymax = ymax;
		this->zmin = zmin;
		this->zmax = zmax;
	}

	vec3 Midpoint() {
		return vec3((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2);
	}
};

struct BVHNode		// 32 bytes
{
	AABB bounds;	// 24 bytes
	int leftFirst;	// 4 bytes
	int count;		// 4 bytes

	BVHNode()
	{
	}
};

class BVH
{
public:
	BVH();
	~BVH();
	void Build(Geometry** scene, int no_elements);
	void load(char* filename);
	void save(char* filename);
private:
	//Scene information
	Geometry** scene;
	int totalNoElements = 0;

	//BVH Construction
	BVHNode* pool;
	uint* orderedIndices;

	AABB calculateAABB(uint* indices, int start, int no_elements);

	//Final BVH
	BVHNode* root;
};

