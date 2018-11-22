#include "precomp.h"



Triangle::Triangle(vec3 v1, vec3 v2, vec3 v3)
{
	this->v1 = v1;
	this->v2 = v2;
	this->v0 = v0;

	e0 = v1 - v0;
	e1 = v2 - v0;
	e2 = v0 - v2;

	N = cross(e0, e1);
	N.normalize();

	D = dot(N, v0);
}

Triangle::~Triangle()
{
}

Collision Triangle::Intersect(Ray ray, bool shatterray)
{
	//printf("Triangle intersect \n");

	Collision collision;
	collision.t = -1;
	float NdotR = dot(ray.Direction, N);
	if (NdotR == 0) return collision; //Ray parrallel to plane, would cause division by 0

	float t = -(dot(ray.Origin, N) + D) / (NdotR);

	//printf("T: %f", t);

	//From https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
	if (t > 0)
	{
		vec3 P = ray.Origin + t * ray.Direction;
		if (dot(N, cross(e0, (P - v0))) > 0 && dot(N, cross(e0, (P - v1))) > 0 && dot(N, cross(e0, (P - v2))) > 0)
		{
			//Collision
			collision.t = t;
			collision.colorAt = 0xffffff;
			collision.other = this;
			collision.N = N;
			collision.Pos = P;
			return collision;
		}
		//printf("no collision \n");
	}
	return collision;
}