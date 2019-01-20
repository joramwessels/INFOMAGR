#pragma once
struct BVHNode;
int sortOnAxis(int axis, float splitplane, uint* orderedIndices, float* scene, float* pool, int index); //Forward declaration because I'm too lazy to fix the order

class BVH
{
public:
	BVH();
	~BVH();
	void Build(float* scene, int no_elements);
	void load(char * filename, int totalNoElements, float* scene);
	void save(char* filename);

	//Scene information
	float* scene;
	int totalNoElements = 0;

	//BVH Construction
	//__declspec(align(64)) BVHNode* pool;
	__declspec(align(64)) float* pool;
	uint poolPtr;
	uint* orderedIndices;

	AABB calculateAABB(uint* indices, int start, int no_elements);

	//Final BVH
	float* root;
	int depth = 0;
};


class ParentBVH : public BVH {
public:
	void join2BVHs(BVH* bvh1, BVH* bvh2);

	BVH* left;
	BVH* right;

	bool doTranslateLeft = false;
	bool doTranslateRight = false;

	vec3 translateLeft = { 0, 0, 0 };
	vec3 translateRight = { 0, 0, 0 };

	Collision Traverse(float* ray_ptr, BVHNode* node);
};

void storeBoundsInFloatArray(float* pool, int startIndex, AABB aabb);

//Will return 0 for x, 1 for y, 2 for z.
int calculateSplitAxis(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax);

float calculateAABBArea(float* pool, int index);

float calculateSplitCost(int axis, float splitplane, BVH* bvh, int index);


float calculateSplitPosition(int axis, BVH* bvh, int index);

void subdivideBVHNode(BVH* bvh, int index, int recursiondepth = 0);

float IntersectAABB(float* ray_ptr, float* BVHNode);







Collision TraverseBVHNode(float* ray_ptr, float* pool, uint* orderedIndices, float* scene, int index);