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
	//pool = new BVHNode[no_elements * 2 - 1];
	pool = new float[(no_elements * 2 - 1) * B_SIZE];
	orderedIndices = new uint[no_elements];

	//Create indices array
	for (size_t i = 0; i < no_elements; i++)
	{
		orderedIndices[i] = i;
	}

	//the root is the first bvhnode
	root = &pool[0];
	poolPtr = 2;

	root[B_LEFTFIRST] = 0;
	root[B_COUNT] = no_elements;
	AABB rootaabb = calculateAABB(orderedIndices, root[B_LEFTFIRST], root[B_COUNT]);
	//root->bounds = calculateAABB(orderedIndices, root->leftFirst, root->count);

	root[B_AABB_MINX] = rootaabb.xmin;
	root[B_AABB_MAXX] = rootaabb.xmax;
	root[B_AABB_MINY] = rootaabb.ymin;
	root[B_AABB_MAXY] = rootaabb.ymax;
	root[B_AABB_MINZ] = rootaabb.zmin;
	root[B_AABB_MAXZ] = rootaabb.zmax;

	subdivideBVHNode(this, 0); 
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
Collision TraverseBVHNode(float* ray_ptr, float* pool, uint* orderedIndices, float* scene, int index)
{
	Collision closest;
	closest.t = -1;

	ray_ptr[R_BVHTRA]++;

	vec3 ray_origin =    { ray_ptr[R_OX], ray_ptr[R_OY], ray_ptr[R_OZ] };
	vec3 ray_direction = { ray_ptr[R_DX], ray_ptr[R_DY], ray_ptr[R_DZ] };

		// If leaf
		if (pool[index + B_COUNT] != 0)
		{
			float closestdist = 0xffffff;

			// Find closest collision
			for (int i = 0; i < pool[index + B_COUNT]; i++)
			{
				//Collision collision = scene[orderedIndices[node->leftFirst + i]]->Intersect(*ray);
				Collision collision = intersectTriangle(orderedIndices[(int)pool[index + B_LEFTFIRST] + i], ray_origin, ray_direction, scene);
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
			//AABB::AABBIntersection tleft = pool[(int)pool[index + B_LEFTFIRST]].bounds.Intersects(ray_origin, ray_direction);
			//AABB::AABBIntersection tright = pool[node->leftFirst + 1].bounds.Intersects(ray_origin, ray_direction);
			
			float tleft = IntersectAABB(ray_ptr, pool + ((index + B_LEFTFIRST) * B_SIZE));
			float tright = IntersectAABB(ray_ptr, pool + ((index + B_LEFTFIRST + 1) * B_SIZE));

			int flip = 0;
			float tEntryFarNode = tright;
			if (tright < tleft && tright > 0) { flip = 1; tEntryFarNode = tleft; };

			Collision colclose, colfar;
			colclose.t = -1;
			colfar.t = -1;

			if ((tleft > 0 && !flip) || (flip && tright > 0)) {
				//colclose = Traverse(ray_ptr, &(pool[node->leftFirst + flip]));
				colclose = TraverseBVHNode(ray_ptr, pool, orderedIndices, scene, (pool[index + B_LEFTFIRST] + flip) * B_SIZE);
				if (colclose.t < tEntryFarNode && colclose.t > 0) {
					return colclose;
				}
			}
			if ((tright > 0 && !flip) || (tleft > 0 && flip)) {
				//colfar = Traverse(ray_ptr, &(pool[node->leftFirst + (1 - flip)]));
				colfar = TraverseBVHNode(ray_ptr, pool, orderedIndices, scene, (pool[index + B_LEFTFIRST] + (1 - flip)) * B_SIZE);

			}

			if (colfar.t == -1) return colclose;
			if (colclose.t == -1) return colfar;
			return (colclose.t < colfar.t ? colclose : colfar);
		}
	
	return closest;
}

void BVH::load(char * filename, int totalNoElements, float* scene)
{
	//TODO: Fix this
	/*this->totalNoElements = totalNoElements;
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
	*/

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
	//outFile.write((char*)pool, sizeof(BVHNode) * poolPtr);


	outFile.close();
	printf("BVH saved as %s \n", filename);

}

void ParentBVH::join2BVHs(BVH * bvh1, BVH * bvh2)
{
	left = bvh1;
	right = bvh2;
}

Collision ParentBVH::Traverse(float* ray_ptr, BVHNode* node)
{
	float *rayright, *rayleft;
	rayright = new float[R_SIZE]; 
	rayleft = new float[R_SIZE];

	memcpy(rayright, ray_ptr, sizeof(float) * R_SIZE);
	memcpy(rayleft, ray_ptr, sizeof(float) * R_SIZE);

	rayright[R_OX] += translateRight.x;
	rayright[R_OY] += translateRight.y;
	rayright[R_OZ] += translateRight.z;
	rayleft[R_OX] += translateLeft.x;
	rayleft[R_OY] += translateLeft.y;
	rayleft[R_OZ] += translateLeft.z;

	//Collision coll1 = left->Traverse(rayleft, left->root);
	//coll1.Pos -= translateLeft;

	//Collision coll2 = right->Traverse(rayright, right->root);
	//coll2.Pos -= translateRight;

	//if ((coll2.t > 0 && coll2.t < coll1.t) || coll1.t < 0) coll1 = coll2;

	//return coll1;
	return Collision();
}

//Partitions into left right on splitplane, returns first for right node.
// O(n) :)
int sortOnAxis(int axis, float splitplane, uint* orderedIndices, float* scene, float* pool, int index) {
	int leftFirst = pool[index + B_LEFTFIRST];
	int count = pool[index + B_COUNT];

	int start = leftFirst;
	int end = start + count;

	bool sorted = false;

	int left = start;
	int right = end - 1;

	while (!sorted)
	{
		while (calculateTriangleAABBMidpoint(orderedIndices[left], scene)[axis] < splitplane) {
			left++;
			if (left >= (end - 1)) {
				return -1;
			}

		}

		while (calculateTriangleAABBMidpoint(orderedIndices[right], scene)[axis] >= splitplane) {
			right--;
			if (right <= start) {
				return -1;
			}

		}

		if (left >= right) {
			sorted = true;
		}
		else {
			int temp = orderedIndices[left];
			orderedIndices[left] = orderedIndices[right];
			orderedIndices[right] = temp;
		}

	}
	return left;
}

float calculateAABBArea(float* pool, int index) {
	float xmin = pool[index + B_AABB_MINX];
	float xmax = pool[index + B_AABB_MAXX];
	float ymin = pool[index + B_AABB_MINY];
	float ymax = pool[index + B_AABB_MAXY];
	float zmin = pool[index + B_AABB_MINZ];
	float zmax = pool[index + B_AABB_MAXZ];


	float diffx = xmax - xmin;
	float diffy = ymax - ymin;
	float diffz = zmax - zmin;

	return 2 * ((diffx * diffy) + (diffy * diffz) + (diffx * diffz));
}

float calculateSplitCost(int axis, float splitplane, BVH* bvh, int index) {
	int leftFirst = bvh->pool[index + B_LEFTFIRST];
	int count = bvh->pool[index + B_COUNT];

	int rightfirst = sortOnAxis(axis, splitplane, bvh->orderedIndices, bvh->scene, bvh->pool, index);

	if (rightfirst == -1) return -1;

	int Nleft = rightfirst - leftFirst;
	int Nright = count - Nleft;

	//One of the leafs empty, split will never be worth it.
	if (Nleft <= 0 || Nright <= 0) return -1;

	float Aleft = bvh->calculateAABB(bvh->orderedIndices, leftFirst, Nleft).Area();
	float Aright = bvh->calculateAABB(bvh->orderedIndices, rightfirst, Nright).Area();

	return (Aleft * Nleft) + (Aright * Nright);
}


float calculateSplitPosition(int axis, BVH* bvh, int index) {
	int count = bvh->pool[index + B_COUNT];
	int leftFirst = bvh->pool[index + B_LEFTFIRST];

	float diff;
	float base;

	float smallest = -1;
	float highest = -1;

	//Set this to 1 to use midpointsplit
#if 0
	for (size_t i = 0; i < count; i++)
	{
		//vec3 mid = bvh->scene[bvh->orderedIndices[leftFirst + i]]->aabb.Midpoint();
		vec3 mid = calculateTriangleAABBMidpoint(bvh->orderedIndices[leftFirst + i], bvh->scene);
		if (mid[axis] > highest || i == 0)
		{
			highest = mid[axis];
		}
		if (mid[axis] < smallest || i == 0)
		{
			smallest = mid[axis];
		}
	}

	diff = highest - smallest;
	base = smallest;

	//MIDPOINTSPLIT
	//return base + (diff / 2.0f);
	return sortOnAxis(axis, base + (diff / 2.0f), bvh->orderedIndices, bvh->scene);
#else
		//SAH
	int numbins = 10;

	float mincost = calculateAABBArea(bvh->pool, index) * count;
	float bestPositionSoFar = -1;

	// Find the minimum and maximum midpoint of the child objects
	for (size_t i = 0; i < count; i++)
	{
		//vec3 mid = bvh->scene[bvh->orderedIndices[leftFirst + i]]->aabb.Midpoint();
		vec3 mid = calculateTriangleAABBMidpoint(bvh->orderedIndices[leftFirst + i], bvh->scene);
		if (i == 0 || mid[axis] > highest)
		{
			highest = mid[axis];
		}
		if (i == 0 || mid[axis] < smallest)
		{
			smallest = mid[axis];
		}
	}

	diff = highest - smallest;
	base = smallest;

	float binsize = diff / (float)numbins;

	//evaluate all split planes
	for (size_t i = 1; i < numbins; i++)
	{
		float currentPosition = base + (i * binsize);
		float cost = calculateSplitCost(axis, currentPosition, bvh, index);

		if (cost < mincost && cost > 0) {
			mincost = cost;
			bestPositionSoFar = currentPosition;
		}
	}

	int chosenaxis = axis;

	//Also evaluate the SAH for the other axes. Set to false to just check the dominant axis. (Slightly faster, but the resulting bvh will be of lesser quality)
	bool checkotheraxis = true;
	if (checkotheraxis) {

		//loop over all axis
		for (int axisi = 0; axisi < 3; axisi++)
		{
			if (axisi == axis) continue; //We already checked this axis
			for (size_t i = 0; i < count; i++)
			{
				//vec3 mid = bvh->scene[bvh->orderedIndices[leftFirst + i]]->aabb.Midpoint();
				vec3 mid = calculateTriangleAABBMidpoint(bvh->orderedIndices[leftFirst + i], bvh->scene);
				if (mid[axisi] > highest || i == 0)
				{
					highest = mid[axisi];
				}
				if (mid[axisi] < smallest || i == 0)
				{
					smallest = mid[axisi];
				}
			}

			diff = highest - smallest;
			base = smallest;

			float binsize = diff / (float)numbins;

			//evaluate all split planes
			for (size_t i = 1; i < numbins; i++)
			{
				float currentPosition = base + (i * binsize);
				float cost = calculateSplitCost(axisi, currentPosition, bvh, index);


				if (cost < mincost && cost > 0) {
					//printf("Found smaller cost on axis %i instead of %i: %f instead of %f \n", axis, axisi, cost, mincost);
					mincost = cost;
					bestPositionSoFar = currentPosition;
					chosenaxis = axisi;
				}
			}
		}
	}

	//Check if this split will actually help
	if (bestPositionSoFar == -1) {
		//printf("Not splitting. count: %i \n", count);
		return -1;
	}
	else {
		//printf("Selected splitposition: %f \n", bestPositionSoFar);
		//printf("Given axis: %i, chosen axis: %i \n", axis, chosenaxis);
		axis = chosenaxis;
		return sortOnAxis(axis, bestPositionSoFar, bvh->orderedIndices, bvh->scene, bvh->pool, index);
	}
#endif
}

void subdivideBVHNode(BVH* bvh, int index, int recursiondepth) {
	int count = bvh->pool[index + B_COUNT];
	int leftFirst = bvh->pool[index + B_LEFTFIRST];

	bool debugprints = false;
	if (debugprints) printf("\n*** Subdividing BVHNode on level %i. Count: %i ***\n", recursiondepth, count);

	//Just to keep track of the bvh depth. Not used, other than to print it
	if (recursiondepth > bvh->depth) bvh->depth = recursiondepth;

	/*if (debugprints) {
		printf("\nBounding box: \n");
		printf("xmin: %f \n", bounds.xmin);
		printf("xmax: %f \n", bounds.xmax);
		printf("ymin: %f \n", bounds.ymin);
		printf("ymax: %f \n", bounds.ymax);
		printf("zmin: %f \n", bounds.zmin);
		printf("zmax: %f \n\n", bounds.zmax);
	}*/

	if (count < 3) {
		if (debugprints) {
			printf("This is a leaf node, containing %i items (orderedindex: geometryindex) ", count);

			for (size_t i = 0; i < count; i++)
			{
				printf("%i: ", leftFirst + i);
				printf("%i, ", bvh->orderedIndices[leftFirst + i]);
			}
			printf("\n");
		}
		return;
	}

	int axis = calculateSplitAxis(bvh->pool[index + B_AABB_MINX], bvh->pool[index + B_AABB_MAXX], bvh->pool[index + B_AABB_MINY], bvh->pool[index + B_AABB_MAXY], bvh->pool[index + B_AABB_MINZ], bvh->pool[index + B_AABB_MAXZ]);

	if (debugprints) printf("Selected axis %i \n", axis);

	int firstForRightChild = calculateSplitPosition(axis, bvh, index);
	if (firstForRightChild == -1) return; //Splitting this node will not make things better



	if (debugprints) printf("First for right child: %i \n", firstForRightChild);

	int leftchild = bvh->poolPtr++ * B_SIZE;
	int rightchild = bvh->poolPtr++ * B_SIZE; //For right child.

	//Create the left child
	bvh->pool[leftchild + B_LEFTFIRST] = leftFirst;
	bvh->pool[leftchild + B_COUNT] = firstForRightChild - leftFirst;
	if (debugprints) printf("Set count of leftchild to %i \n", bvh->pool[leftchild + B_COUNT]);

	if (bvh->pool[leftchild + B_COUNT] == 0 || count - bvh->pool[leftchild + B_COUNT] == 0) {
		if (true)
		{
			printf("Zero in this child, all %i items (orderedindex: geometryindex) ", count);

			for (size_t i = 0; i < count; i++)
			{
				printf("%i: ", leftFirst + i);
				printf("%i, ", bvh->orderedIndices[leftFirst + i]);
			}
			printf("\n");
		}
		//Apparently the centers are too close together or something. devide equally between L / R
		//Totally not a hack.
		//This shouldn't happen anymore. Just here for reassurance
		bvh->pool[leftchild + B_COUNT] = count / 2;
		firstForRightChild = leftFirst + bvh->pool[leftchild + B_COUNT];
	}

	AABB leftchildaabb = bvh->calculateAABB(bvh->orderedIndices, bvh->pool[leftchild + B_LEFTFIRST], bvh->pool[leftchild + B_COUNT]);

	storeBoundsInFloatArray(bvh->pool, leftchild, leftchildaabb);

	//Create the right child
	bvh->pool[rightchild + B_LEFTFIRST] = firstForRightChild;
	bvh->pool[rightchild + B_COUNT] = count - bvh->pool[leftchild + B_COUNT];
	if (debugprints) printf("Set count of rightchild to %i \n", bvh->pool[rightchild + B_COUNT]);

	AABB rightchildaabb = bvh->calculateAABB(bvh->orderedIndices, firstForRightChild, bvh->pool[rightchild + B_COUNT]);

	storeBoundsInFloatArray(bvh->pool, rightchild, rightchildaabb);

	//bvh->pool[rightchild].bounds = bvh->calculateAABB(bvh->orderedIndices, firstForRightChild, bvh->pool[rightchild].count);

	if (debugprints) printf("My count: %i, Left count: %i, Right count: %i", count, bvh->pool[leftchild + B_COUNT], bvh->pool[rightchild + B_COUNT]);


	//Subdivide the children
	if (debugprints) printf("Starting subdivide of left child on level %i \n", recursiondepth);
	if (debugprints) printf("Starting subdivide of right child on level %i \n", recursiondepth);

	subdivideBVHNode(bvh, leftchild, recursiondepth + 1);

	bvh->pool[index + B_LEFTFIRST] = leftchild;
	count = 0; //Set count to 0, because this node is no longer a leaf node.
	subdivideBVHNode(bvh, rightchild, recursiondepth + 1);

	if (debugprints) printf("Node on level %i done. \n", recursiondepth);

}


int calculateSplitAxis(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax) {
	float xdiff = xmax - xmin;
	float ydiff = ymax - ymin;
	float zdiff = zmax - zmin;

	int axis = 0;
	float max = xdiff;
	if (ydiff > max) { axis = 1; max = ydiff; }
	if (zdiff > max) axis = 2;

	return axis;
}

float IntersectAABB(float* ray_ptr, float* BVHNode)
{
	float xmin = BVHNode[B_AABB_MINX];
	float xmax = BVHNode[B_AABB_MAXX];
	float ymin = BVHNode[B_AABB_MINY];
	float ymax = BVHNode[B_AABB_MAXY];
	float zmin = BVHNode[B_AABB_MINZ];
	float zmax = BVHNode[B_AABB_MAXZ];

	float dirX = ray_ptr[R_DX];
	float dirY = ray_ptr[R_DY];
	float dirZ = ray_ptr[R_DZ];
	float OX = ray_ptr[R_OX];
	float OY = ray_ptr[R_OY];
	float OZ = ray_ptr[R_OZ];

	float invDirX = 1 / dirX;
	float tmin = (xmin - OX) * invDirX;
	float tmax = (xmax - OX) * invDirX;

	if (tmin > tmax) swap(tmin, tmax);

	float invDirY = 1 / dirY;
	float tymin = (ymin - OY) * invDirY;
	float tymax = (ymax - OY) * invDirY;

	if (tymin > tymax) swap(tymin, tymax);

	if ((tmin > tymax) || (tymin > tmax))
		return -1;

	tmin = max(tmin, tymin);
	tmax = min(tymax, tmax);

	float invDirZ = 1 / dirZ;
	float tzmin = (zmin - OZ) * invDirZ;
	float tzmax = (zmax - OZ) * invDirZ;

	if (tzmin > tzmax) swap(tzmin, tzmax);

	if ((tmin > tzmax) || (tzmin > tmax))
		return -1;

	tmin = max(tmin, tzmin);
	tmax = min(tzmax, tmax);

	if (tmax < 0) return -1;

	return tmin;
}

void storeBoundsInFloatArray(float* pool, int startIndex, AABB aabb) {
	pool[startIndex + B_AABB_MINX] = aabb.xmin;
	pool[startIndex + B_AABB_MAXX] = aabb.xmax;
	pool[startIndex + B_AABB_MINY] = aabb.ymin;
	pool[startIndex + B_AABB_MAXY] = aabb.ymax;
	pool[startIndex + B_AABB_MINZ] = aabb.zmin;
	pool[startIndex + B_AABB_MAXZ] = aabb.zmax;

}