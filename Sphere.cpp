#include "Sphere.h"
#include "precomp.h"

Sphere::Sphere( vec3 position, float r, Material material )
{
	this->position = position;
	r2 = r * r;
	this->material = material;
}

Collision Sphere::Intersect( Ray ray )
{
	Collision collision;

	// If the origin of the ray is inside the sphere
	if ( (ray.Origin - position).length() < sqrt(r2) )
	{
		vec3 PO = position - ray.Origin;
		collision.t = 2 * sqrt( r2 ) * dot( PO.normalized(), ray.Direction );
		collision.Pos = ray.Origin + collision.t * ray.Direction;
		collision.other = this;
		collision.N = ( position - collision.Pos ).normalized();

		if ( material.texturetype == Material::COLOR )
		{
			collision.colorAt = material.color;
		}
		return collision;
	}

	vec3 c = position - ray.Origin;
	float t = dot( c, ray.Direction );
	vec3 q = c - ( t * ray.Direction );
	float p2 = dot( q, q );

	if ( p2 > r2 )
	{
		collision.t = -1; //No collision
		return collision;
	}

	t -= sqrt( r2 - p2 );
	if ( t > 0 )
	{
		collision.t = t;
		collision.other = this;
		collision.Pos = ray.Origin + t * ray.Direction;
		collision.N = ( collision.Pos - position ).normalized();

		if ( material.texturetype == Material::COLOR )
		{
			collision.colorAt = material.color;
		}

		return collision; //Collision at t.
	}
	else
	{
		collision.t = -1; //No collision
	}
	return collision;
}
