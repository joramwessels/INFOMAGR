#include "precomp.h"
#include "Sphere.h"


Sphere::Sphere(vec3 position, float r, uint color)
{
	this->position = position;
	r2 = r * r;
	this->color.from_uint(color);
}

Collision Sphere::Intersect(Ray ray)
{
	Collision collision;

	vec3 c = position - ray.Origin;
	float t = dot(c, ray.Direction);
	vec3 q = c - (t * ray.Direction);
	float p2 = dot(q, q);

	if (p2 > r2) {
		collision.t = -1; //No collision
		return collision;
	}

	t -= sqrt(r2 - p2);
	if (t > 0)
	{
		collision.t = t;
		collision.other = this;
		collision.Pos = ray.Origin + t * ray.Direction;
		collision.N = (collision.Pos - position).normalized();
		return collision; //Collision at t.
	}
	else {
		collision.t = -1; //No collision
	}
	return collision;
}


