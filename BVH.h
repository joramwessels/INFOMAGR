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

	void subdivide() {
		int axis = calculateSplitAxis();
		float splitposition = calculateSplitPosition(axis);
		//sortOnAxis(axis) //TODO: sort the elements on the selected axis (use the midpoint of the aabb). Or just put all elements that have the splitaxis < splitposition in the front. No real sorting needed.
		//TODO: create child nodes, and subdevide them
	}

	//Will return 0 for x, 1 for y, 2 for z.
	int calculateSplitAxis(){
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
	//TODO this is wrong. It should be a split in space, not just half/half in number of objects. I think..
	float calculateSplitPosition(int axis) {
		float diff;
		switch (axis)
		{
		case 0:
			diff = bounds.xmax - bounds.xmin;
			break;
		case 1:
			diff = bounds.ymax - bounds.ymin;
			break;
		case 2:
			diff = bounds.zmax - bounds.zmin;
			break;
		default:
			break;
		}
		return diff / 2.0f;
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

