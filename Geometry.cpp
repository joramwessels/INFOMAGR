#include "precomp.h"


Geometry::Geometry()
{
}


Geometry::~Geometry()
{
}

Collision Geometry::Intersect(Ray ray, bool isShadowray)
{
	printf("This function should not be called");
	return Collision();
}

AABB Geometry::GetAABB()
{
	printf("This function should not be called");
	return AABB();
}