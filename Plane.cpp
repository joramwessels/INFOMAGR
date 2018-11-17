#include "precomp.h"
#include "Plane.h"

Plane::Plane(vec3 N, float d, Material material)
{
	this->N = N.normalized();
	this->d = d;
	this->material = material;

}

Plane::~Plane()
{
}

Collision Plane::Intersect(Ray ray)
{
	float t = -(dot(ray.Origin, N) + d) / (dot(ray.Direction, N));

	Collision collision;
	if (t < 0) {
		collision.t = -1;
		return collision;
	}
	else
	{
		collision.N = -N; //TODO: this is not always correct. If the ray comes from the other side, it should be N
		collision.other = this;
		collision.Pos = ray.Origin + t * ray.Direction;
		collision.t = t;
		return collision;
	}
}
