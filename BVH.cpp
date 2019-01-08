#include "precomp.h"

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
void BVH::Build(float* scene, int no_elements)
{
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
	poolPtr = 2;

	root->leftFirst = 0;
	root->count = no_elements;
	root->bounds = calculateAABB(orderedIndices, root->leftFirst, root->count);
	root->subdivide(this); 
}


// Calculates the smallest surrounding AABB for the given array of nodes
AABB BVH::calculateAABB(uint* indices, int start, int no_elements)
{
	if (no_elements == 0) {
		printf("This shouldn't happen! \n");
	}

	float xmin = scene[(indices[start] * FLOATS_PER_TRIANGLE) + T_AABBMINX];
	float xmax = scene[(indices[start] * FLOATS_PER_TRIANGLE) + T_AABBMAXX];
	float ymin = scene[(indices[start] * FLOATS_PER_TRIANGLE) + T_AABBMINY];
	float ymax = scene[(indices[start] * FLOATS_PER_TRIANGLE) + T_AABBMAXY];
	float zmin = scene[(indices[start] * FLOATS_PER_TRIANGLE) + T_AABBMINZ];
	float zmax = scene[(indices[start] * FLOATS_PER_TRIANGLE) + T_AABBMAXZ];
	for (int i = start + 1; i < (start + no_elements); i++) {
		xmin = min(xmin, scene[(indices[i] * FLOATS_PER_TRIANGLE) + T_AABBMINX]);
		xmax = max(xmax, scene[(indices[i] * FLOATS_PER_TRIANGLE) + T_AABBMAXX]);
		ymin = min(ymin, scene[(indices[i] * FLOATS_PER_TRIANGLE) + T_AABBMINY]);
		ymax = max(ymax, scene[(indices[i] * FLOATS_PER_TRIANGLE) + T_AABBMAXY]);
		zmin = min(zmin, scene[(indices[i] * FLOATS_PER_TRIANGLE) + T_AABBMINZ]);
		zmax = max(zmax, scene[(indices[i] * FLOATS_PER_TRIANGLE) + T_AABBMAXZ]);

		/*
		AABB thisGeometryAABB = scene[indices[i]]->aabb;
		if (thisGeometryAABB.xmin < xmin) xmin = thisGeometryAABB.xmin;
		if (thisGeometryAABB.xmax > xmax) xmax = thisGeometryAABB.xmax;
		if (thisGeometryAABB.ymin < ymin) ymin = thisGeometryAABB.ymin;
		if (thisGeometryAABB.ymax > ymax) ymax = thisGeometryAABB.ymax;
		if (thisGeometryAABB.zmin < zmin) zmin = thisGeometryAABB.zmin;
		if (thisGeometryAABB.zmax > zmax) zmax = thisGeometryAABB.zmax;*/
	}
	return AABB(xmin, xmax, ymin, ymax, zmin, zmax);
}

// Recursively traverses the BVH tree from the given node to find a collision. Returns collision with t = -1 if none were found.
Collision BVH::Traverse(Ray* ray, BVHNode* node)
{
	Collision closest;
	closest.t = -1;
	ray->bvhtraversals++;

		// If leaf
		if (node->count != 0)
		{
			float closestdist = 0xffffff;

			// Find closest collision
			for (int i = 0; i < node->count; i++)
			{
				//Collision collision = scene[orderedIndices[node->leftFirst + i]]->Intersect(*ray);
				Collision collision = intersectTriangle(orderedIndices[node->leftFirst + i], *ray, scene);
				float dist = collision.t;
				if (dist != -1 && dist < closestdist)
				{
					//Collision. Check if closest
					closest = collision;
					closestdist = dist;
				}
			}
			//printf("leaf: collision at %f \n", closest.t);
			return closest;
		}
		// If node
		else
		{
			// Check both children and return the closest collision if both intersected
			AABB::AABBIntersection tleft = pool[node->leftFirst].bounds.Intersects(*ray);
			AABB::AABBIntersection tright = pool[node->leftFirst + 1].bounds.Intersects(*ray);
			
			int flip = 0;
			float tEntryFarNode = tright.tEntry;
			if (tright.tEntry < tleft.tEntry) { flip = 1; tEntryFarNode = tleft.tEntry; };

			Collision colclose, colfar;
			colclose.t = -1;
			colfar.t = -1;

			if ((tleft.intersects && !flip) || (flip && tright.intersects)) {
				colclose = Traverse(ray, &(pool[node->leftFirst + flip]));
				if (colclose.t < tEntryFarNode && colclose.t > 0) {
					return colclose;
				}
			}
			if ((tright.intersects && !flip) || (tleft.intersects && flip)) {
				colfar = Traverse(ray, &(pool[node->leftFirst + (1 - flip)]));
			}

			if (colfar.t == -1) return colclose;
			if (colclose.t == -1) return colfar;
			return (colclose.t < colfar.t ? colclose : colfar);
		}
	
	return closest;
}

void BVH::load(char * filename, int totalNoElements, float* scene)
{
	this->totalNoElements = totalNoElements;
	this->scene = scene;

	int metadata[2]; //no elements & poolptr
	ifstream inFile(filename, ios::in | ios::binary);
	inFile.read((char*)metadata, sizeof(int) * 2);

	if (metadata[0] != totalNoElements) {
		printf("Number of elements in bvh does not match the given scene!");
		exit(-1);
	}

	this->poolPtr = metadata[1];
	orderedIndices = new uint[poolPtr];
	inFile.read((char*)orderedIndices, sizeof(uint) * totalNoElements);

	pool = new BVHNode[poolPtr];
	inFile.read((char*)pool, sizeof(BVHNode) * poolPtr);
	root = pool;
	
	printf("BVH loaded: %s \n", filename);


}

void BVH::save(char * filename)
{
	printf("Saving bvh to %s... \n", filename);
	ofstream outFile(filename, ios::out | ios::binary);
	
	//Save the number of elements and number of bvhnodes (=poolptr) to the file 
	int metadata[2];
	metadata[0] = totalNoElements;
	metadata[1] = poolPtr;
	
	outFile.write((char*)metadata, sizeof(int) * 2);
	outFile.write((char*)orderedIndices, sizeof(uint) * totalNoElements);
	outFile.write((char*)pool, sizeof(BVHNode) * poolPtr);


	outFile.close();
	printf("BVH saved as %s \n", filename);

}

void ParentBVH::join2BVHs(BVH * bvh1, BVH * bvh2)
{
	left = bvh1;
	right = bvh2;
}

Collision ParentBVH::Traverse(Ray* ray, BVHNode* node)
{
	Ray rayright = *ray;
	Ray rayleft = *ray;

	rayright.Origin += translateRight;
	rayleft.Origin += translateLeft;

	Collision coll1 = left->Traverse(&rayleft, left->root);
	coll1.Pos -= translateLeft;

	Collision coll2 = right->Traverse(&rayright, right->root);
	coll2.Pos -= translateRight;

	if ((coll2.t > 0 && coll2.t < coll1.t) || coll1.t < 0) coll1 = coll2;

	return coll1;
}
