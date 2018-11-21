#include "Sphere.h"
#include "precomp.h"

Sphere::Sphere( vec3 position, float r, Material material )
{
	this->position = position;
	r2 = r * r;
	this->material = material;
}

Collision Sphere::Intersect( Ray ray, bool shatterray )
{
	Collision collision;

	// If the origin of the ray is inside the sphere
	if ( (ray.Origin - position).sqrLentgh() < r2 )
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
		if (shatterray) {
			return collision; //We just want to know if it hits *something*
		}
		collision.t = t;
		collision.other = this;
		collision.Pos = ray.Origin + t * ray.Direction;
		collision.N = ( collision.Pos - position ).normalized();

		if ( material.texturetype == Material::COLOR )
		{
			collision.colorAt = material.color;
		}
		else {
			float u = 0.5 + (atan2f(-collision.N.z, -collision.N.x) / (2 * PI));
			float v = 0.5 - (asinf(-collision.N.y) / PI);

			if (material.texturetype == Material::CHECKERBOARD)
			{
				
				//printf("u: %f, v: %f \n", u, v);
					if ((int)(v * 10) % 2 == (int)(u * 10) % 2) {
						collision.colorAt = material.color2;
					}

				else {
					collision.colorAt = material.color;
				}
			}
			if (material.texturetype == Material::TEXTURE) {
				collision.colorAt = material.texture->GetBuffer()[(int)(material.texture->GetWidth() * u) + (int)(material.texture->GetHeight() * v) * material.texture->GetPitch()];
			}
		}

		return collision; //Collision at t.
	}
	else
	{
		collision.t = -1; //No collision
	}
	return collision;
}
