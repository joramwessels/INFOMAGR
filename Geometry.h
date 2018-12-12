#pragma once
#include "BVH.h"

//This is a base class used for geometry objects.
class Geometry
{
public:
	Geometry();
	~Geometry();
	//Color color;

	Material material;

	virtual Collision Intersect(Ray ray, bool isShadowray = false);
	virtual AABB GetAABB();

private:
};

