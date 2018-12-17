#pragma once
struct BVHNode;


class BVH
{
public:
	BVH();
	~BVH();
	void Build(Geometry** scene, int no_elements);
	Collision Traverse(Ray ray, BVHNode* node);
	void load(char* filename);
	void save(char* filename);

	//Scene information
	Geometry** scene;
	int totalNoElements = 0;

	//BVH Construction
	BVHNode* pool;
	uint poolPtr;
	uint* orderedIndices;

	AABB calculateAABB(uint* indices, int start, int no_elements);

	//Final BVH
	BVHNode* root;
	int depth;
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

		float splitposition = calculateSplitPosition(axis);
		if (debugprints) printf("Splitposition %f \n", splitposition);

		int firstForRightChild = sortOnAxis(axis, splitposition, bvh->orderedIndices, bvh->scene);

		int leftchild = bvh->poolPtr++;
		int rightchild = bvh->poolPtr++; //For right child.

		//Create the left child
		bvh->pool[leftchild].leftFirst = leftFirst;
		bvh->pool[leftchild].count = firstForRightChild - leftFirst;
		if (debugprints) printf("Set count of leftchild to %i \n", bvh->pool[leftchild].count);

		if (bvh->pool[leftchild].count == 0 || count - bvh->pool[leftchild].count == 0) {
			if (debugprints)
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

	//Currently midpoint-split. TODO: surface area heuristic or something
	float calculateSplitPosition(int axis) {
		
		float diff;
		float base;

		switch (axis)
		{
		case 0:
			diff = bounds.xmax - bounds.xmin;
			base = bounds.xmin;
			break;
		case 1:
			diff = bounds.ymax - bounds.ymin;
			base = bounds.ymin;
			break;
		case 2:
			diff = bounds.zmax - bounds.zmin;
			base = bounds.zmin;
			break;
		default:
			break;
		}
		//printf("Diff: %f \n", diff);

		return base + (diff / 2.0f);
	}

	//Partitions into left right on splitplane, returns first for right node
	int sortOnAxis(int axis, float splitplane, uint* orderedIndices, Geometry** scene) {
		int start = leftFirst;
		int end = start + count;

		bool sorted = false;
		
		int left = start;
		int right = end - 1;
		
		while(!sorted)
		{
			while (scene[orderedIndices[left]]->aabb.Midpoint()[axis] < splitplane) {
				left++;
			}

			while (scene[orderedIndices[right]]->aabb.Midpoint()[axis] >= splitplane) {
				right--;
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


