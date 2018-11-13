#pragma once
#include "Geometry.h"
class Plane :
	public Geometry
{
public:
	Plane(vec3 N, float d, uint color);
	~Plane();

	Collision Intersect(Ray ray) override;


private:
	vec3 N;
	float d;
};

