#pragma once
struct BVHNode;


class BVH
{
public:
	BVH();
	~BVH();
	void Build(float* scene, int no_elements);
	virtual Collision Traverse(float* ray_ptr, BVHNode* node);
	void load(char * filename, int totalNoElements, float* scene);
	void save(char* filename);

	//Scene information
	float* scene;
	int totalNoElements = 0;

	//BVH Construction
	__declspec(align(64)) BVHNode* pool;
	uint poolPtr;
	uint* orderedIndices;

	AABB calculateAABB(uint* indices, int start, int no_elements);

	//Final BVH
	BVHNode* root;
	int depth;
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

	Collision Traverse(float* ray_ptr, BVHNode* node) override;
};

struct BVHNode		// 32 bytes
{
	AABB bounds;	// 24 bytes
	int leftFirst;	// 4 bytes
	int count;		// 4 bytes

	BVHNode()
	{
	}

	void subdivide(BVH* bvh, int recursiondepth = 0) {
		bool debugprints = false;
		if (debugprints) printf("\n*** Subdividing BVHNode on level %i. Count: %i ***\n", recursiondepth, count);

		//Just to keep track of the bvh depth. Not used, other than to print it
		if (recursiondepth > bvh->depth) bvh->depth = recursiondepth;

		if (debugprints) {
			printf("\nBounding box: \n");
			printf("xmin: %f \n", bounds.xmin);
			printf("xmax: %f \n", bounds.xmax);
			printf("ymin: %f \n", bounds.ymin);
			printf("ymax: %f \n", bounds.ymax);
			printf("zmin: %f \n", bounds.zmin);
			printf("zmax: %f \n\n", bounds.zmax);
		}

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

		int axis = calculateSplitAxis();
		if (debugprints) printf("Selected axis %i \n", axis);

		int firstForRightChild = calculateSplitPosition(axis, bvh);
		if (firstForRightChild == -1) return; //Splitting this node will not make things better



		if (debugprints) printf("First for right child: %i \n", firstForRightChild);

		int leftchild = bvh->poolPtr++;
		int rightchild = bvh->poolPtr++; //For right child.

		//Create the left child
		bvh->pool[leftchild].leftFirst = leftFirst;
		bvh->pool[leftchild].count = firstForRightChild - leftFirst;
		if (debugprints) printf("Set count of leftchild to %i \n", bvh->pool[leftchild].count);

		if (bvh->pool[leftchild].count == 0 || count - bvh->pool[leftchild].count == 0) {
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
			bvh->pool[leftchild].count = count / 2;
			firstForRightChild = leftFirst + bvh->pool[leftchild].count;
		}

		bvh->pool[leftchild].bounds = bvh->calculateAABB(bvh->orderedIndices, bvh->pool[leftchild].leftFirst, bvh->pool[leftchild].count);

		//Create the right child
		bvh->pool[rightchild].leftFirst = firstForRightChild;
		bvh->pool[rightchild].count = count - bvh->pool[leftchild].count;
		if (debugprints) printf("Set count of rightchild to %i \n", bvh->pool[rightchild].count);

		bvh->pool[rightchild].bounds = bvh->calculateAABB(bvh->orderedIndices, firstForRightChild, bvh->pool[rightchild].count);

		if (debugprints) printf("My count: %i, Left count: %i, Right count: %i", count, bvh->pool[leftchild].count, bvh->pool[rightchild].count);


		//Subdivide the children
		if (debugprints) printf("Starting subdivide of left child on level %i \n", recursiondepth);
		bvh->pool[leftchild].subdivide(bvh, recursiondepth + 1);
		if (debugprints) printf("Starting subdivide of right child on level %i \n", recursiondepth);

		bvh->pool[rightchild].subdivide(bvh, recursiondepth + 1);
		leftFirst = leftchild;
		count = 0; //Set count to 0, because this node is no longer a leaf node.

		if (debugprints) printf("Node on level %i done. \n", recursiondepth);

	}

	//Will return 0 for x, 1 for y, 2 for z.
	int calculateSplitAxis() {
		float xdiff = bounds.xmax - bounds.xmin;
		float ydiff = bounds.ymax - bounds.ymin;
		float zdiff = bounds.zmax - bounds.zmin;

		int axis = 0;
		float max = xdiff;
		if (ydiff > max) { axis = 1; max = ydiff; }
		if (zdiff > max) axis = 2;

		return axis;
	}

	float calculateSplitPosition(int axis, BVH* bvh) {
		
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

		float mincost = bounds.Area() * count;
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
			float cost = calculateSplitCost(axis, currentPosition, bvh);

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
					float cost = calculateSplitCost(axisi, currentPosition, bvh);


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
			return sortOnAxis(axis, bestPositionSoFar, bvh->orderedIndices, bvh->scene);
		}
#endif
	}

	float calculateSplitCost(int axis, float splitplane, BVH* bvh) {
		int rightfirst = sortOnAxis(axis, splitplane, bvh->orderedIndices, bvh->scene);

		if (rightfirst == -1) return -1;

		int Nleft = rightfirst - leftFirst;
		int Nright = count - Nleft;

		//One of the leafs empty, split will never be worth it.
		if (Nleft <= 0 || Nright <= 0) return -1;

		float Aleft = bvh->calculateAABB(bvh->orderedIndices, leftFirst, Nleft).Area();
		float Aright = bvh->calculateAABB(bvh->orderedIndices, rightfirst, Nright).Area();

		return (Aleft * Nleft) + (Aright * Nright);
	}

	//Partitions into left right on splitplane, returns first for right node.
	// O(n) :)
	int sortOnAxis(int axis, float splitplane, uint* orderedIndices, float* scene) {
		int start = leftFirst;
		int end = start + count;

		bool sorted = false;
		
		int left = start;
		int right = end - 1;
		
		while(!sorted)
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
};


