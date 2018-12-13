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
void BVH::Build(Geometry** scene, int no_elements)
{
	/*// Creating initial object indices
	std::vector<int> v(no_elements);
	std::iota(std::begin(v), std::end(v), 0);
	int* indices = &v[0];
	// Creating root node
	BVHNode* leafs = calculateLeafNodes(scene, no_elements);
	BVHNode root = BVHNode();
	root.bounds = calculateAABB(leafs, no_elements);
	root.count = 2;*/

	//I modified this a little bit, see slide 31 from lecture 2 :)

	totalNoElements = no_elements;
	this->scene = scene;

	//Create BVHnode pool, to pick BVHnodes from later
	pool = new BVHNode[no_elements * 2 - 1];
	orderedIndices = new uint[no_elements];

	//Create indices array
	for (size_t i = 0; i < no_elements; i++)
	{
		orderedIndices[i] = i;
	}

	//the root is the first bvhnode
	root = &pool[0];
	root->leftFirst = 0;
	root->count = no_elements;
	root->bounds = calculateAABB(orderedIndices, root->leftFirst, root->count);
	root->subdivide(); //TODO: make :P

		// TODO: recursively build tree
}

/*
// I don't think we need this -M
// Converts a scene of geometric objects into an array of BVH leaf nodes
BVHNode* BVH::calculateLeafNodes(Geometry* scene, int no_elements)
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
*/


// Calculates the smallest surrounding AABB for the given array of nodes
AABB BVH::calculateAABB(uint* indices, int start, int no_elements)
{
	float xmin = scene[indices[0]]->aabb.xmin, xmax = scene[indices[0]]->aabb.xmax, ymin = scene[indices[0]]->aabb.ymin;
	float ymax = scene[indices[0]]->aabb.ymax, zmin = scene[indices[0]]->aabb.zmin, zmax = scene[indices[0]]->aabb.zmax;
	for (int i = start; i < start + no_elements; i++) {
		AABB thisGeometryAABB = scene[indices[i]]->aabb;
		if (thisGeometryAABB.xmin < xmin) xmin = thisGeometryAABB.xmin;
		if (thisGeometryAABB.xmax > xmax) xmax = thisGeometryAABB.xmax;
		if (thisGeometryAABB.ymin < ymin) ymin = thisGeometryAABB.ymin;
		if (thisGeometryAABB.ymax > ymax) ymax = thisGeometryAABB.ymax;
		if (thisGeometryAABB.zmin < zmin) zmin = thisGeometryAABB.zmin;
		if (thisGeometryAABB.zmax > zmax) zmax = thisGeometryAABB.zmax;
	}
	return AABB(xmin, xmax, ymin, ymax, zmin, zmax);
}