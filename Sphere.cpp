#include "precomp.h"
#include "Sphere.h"


Sphere::Sphere(vec3 position, float r, uint color)
{
	this->position = position;
	r2 = r * r;
	this->color.from_uint(color);
}

float Sphere::Intersect(Ray ray)
{
	vec3 c = position - ray.Origin;
	float t = dot(c, ray.Direction);
	vec3 q = c - (t * ray.Direction);
	float p2 = dot(q, q);

	if (p2 > r2) return -1.0f; //No collision

	t -= sqrt(r2 - p2);
	if(t > 0) return t; //Collision a t.
	else return -1.0f;
}


