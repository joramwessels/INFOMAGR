#pragma once
#include "Geometry.h"
class Plane :
	public Geometry
{
public:
	Plane(vec3 N, float d, uint color);
	~Plane();

	float Intersect(Ray ray) override;


private:
	vec3 N;
	float d;
};

