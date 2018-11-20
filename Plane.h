#pragma once
#include "Geometry.h"
class Plane :
	public Geometry
{
public:
	Plane(vec3 N, float d, Material material);
	~Plane();

	Collision Intersect(Ray ray) override;


private:
	vec3 N;
	float d;

	vec3 Xaxis; //For texture alignment
	vec3 Yaxis;
};

