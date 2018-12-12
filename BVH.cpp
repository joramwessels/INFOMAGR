#include "precomp.h"
#include "BVH.h"
#include <numeric>

//----------------------------------------------
// TODO:
// - Plane to AABB (in Plane.cpp/Plane::GetAABB)
// - Recursive tree building while sorting indices
//		(the problem here is how to track the right
//		 child if the nodes aren't sorted yet)
// - Load & Save
//----------------------------------------------

BVH::BVH()
{
}


BVH::~BVH()
{
}

// Builds a BVH tree using recursive splits given a list of geometric objects
void BVH::Build(Geometry* scene, int no_elements)
{
	// Creating initial object indices
	std::vector<int> v(no_elements);
	std::iota(std::begin(v), std::end(v), 0);
	int* indices = &v[0];
	// Creating root node
	BVHNode* leafs = calculateLeafNodes(scene, no_elements);
	BVHNode root = BVHNode();
	root.bounds = calculateAABB(leafs, no_elements);
	root.count = 2;

		// TODO: recursively build tree
}

// Converts a scene of geometric objects into an array of BVH leaf nodes
BVHNode* calculateLeafNodes(Geometry* scene, int no_elements)
{
	BVHNode* leafs = (BVHNode*) malloc(sizeof(BVHNode) * no_elements);
	for (int i = 0; i < no_elements; i++) {
		leafs[i] = BVHNode();
		leafs[i].count = 1;
		leafs[i].leftFirst = i;
		leafs[i].bounds = scene[i].GetAABB();
	}
	return leafs;
}

// Calculates the smallest surrounding AABB for the given array of nodes
AABB calculateAABB(BVHNode* nodes, int no_elements)
{
	float xmin = nodes[0].bounds.xmin, xmax = nodes[0].bounds.xmax, ymin = nodes[0].bounds.ymin;
	float ymax = nodes[0].bounds.ymax, zmin = nodes[0].bounds.zmin, zmax = nodes[0].bounds.zmax;
	for (int i = 1; i < no_elements; i++) {
		if (nodes[i].bounds.xmin < xmin) xmin = nodes[i].bounds.xmin;
		if (nodes[i].bounds.xmax > xmax) xmax = nodes[i].bounds.xmax;
		if (nodes[i].bounds.ymin < ymin) ymin = nodes[i].bounds.ymin;
		if (nodes[i].bounds.ymax > ymax) ymax = nodes[i].bounds.ymax;
		if (nodes[i].bounds.zmin < zmin) zmin = nodes[i].bounds.zmin;
		if (nodes[i].bounds.zmax > zmax) zmax = nodes[i].bounds.zmax;
	}
	return AABB(xmin, xmax, ymin, ymax, zmin, zmax);
}