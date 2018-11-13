#include "precomp.h"
#include "Plane.h"



Plane::Plane(vec3 N, float d, uint color)
{
	this->N = N.normalized();
	this->d = d;
	this->color.from_uint(color);

}

Plane::~Plane()
{
}

float Plane::Intersect(Ray ray)
{
	float t = -(dot(ray.Origin, N) + d) / (dot(ray.Direction, N));
	if (t < 0) return -1.0f;
	else return t;
}
