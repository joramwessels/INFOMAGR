#include "precomp.h"


//Use CCW ordering for the vertices! Otherwise the normal will be flipped.
Triangle::Triangle(vec3 v0, vec3 v1, vec3 v2, Material material)
{
	this->material = material;

	this->v1 = v1;
	this->v2 = v2;
	this->v0 = v0;

	e0 = v1 - v0;
	e1 = v2 - v1;
	e2 = v0 - v2;

	N = cross(e0, e1);
	N.normalize();

	D = -dot(N, v0);
	//printf("Triangle N: %f, %f, %f. D: %f \n", N.x, N.y, N.z, D);

}

Triangle::Triangle(vec3 v0, vec3 v1, vec3 v2, vec3 N, Material material)
{
	this->material = material;

	this->v1 = v1;
	this->v2 = v2;
	this->v0 = v0;

	e0 = v1 - v0;
	e1 = v2 - v1;
	e2 = v0 - v2;

	this->N = N.normalized();

	//N = cross(e0, e1);
	//N.normalize();

	D = -dot(N, v0);

}

Triangle::~Triangle()
{
}

Collision Triangle::Intersect(Ray ray, bool isShadowray)
{
	Collision collision;
	collision.t = -1;
	float NdotR = dot(ray.Direction, N);
	if (NdotR == 0) return collision; //Ray parrallel to plane, would cause division by 0

	float t = -(dot(ray.Origin, N) + D) / (NdotR);

	//From https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
	if (t > 0)
	{
		vec3 P = ray.Origin + t * ray.Direction;
		if (dot(N, cross(e0, (P - v0))) > 0 && dot(N, cross(e1, (P - v1))) > 0 && dot(N, cross(e2, (P - v2))) > 0)
		{
			//Collision
			collision.t = t;
	
			if (isShadowray) {
				return collision;
			}

			collision.colorAt = material.color;
			collision.other = this;
			if (NdotR > 0) collision.N = -N;
			else collision.N = N;
			collision.Pos = P;
			return collision;
		}
	}
	return collision;
}

AABB Triangle::GetAABB()
{
	return AABB(
		(v0.x <= v1.x && v0.x <= v2.x ? v0.x : (v1.x <= v0.x && v1.x <= v2.x ? v1.x : v2.x)),
		(v0.x >= v1.x && v0.x >= v2.x ? v0.x : (v1.x >= v0.x && v1.x >= v2.x ? v1.x : v2.x)),
		(v0.y <= v1.y && v0.y <= v2.y ? v0.y : (v1.y <= v0.y && v1.y <= v2.y ? v1.y : v2.y)),
		(v0.y >= v1.y && v0.y >= v2.y ? v0.y : (v1.y >= v0.y && v1.y >= v2.y ? v1.y : v2.y)),
		(v0.z <= v1.z && v0.z <= v2.z ? v0.z : (v1.z <= v0.z && v1.z <= v2.z ? v1.z : v2.z)),
		(v0.z >= v1.z && v0.z >= v2.z ? v0.z : (v1.z >= v0.z && v1.z >= v2.z ? v1.z : v2.z))
	);
}