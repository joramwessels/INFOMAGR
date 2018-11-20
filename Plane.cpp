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

		if (material.texturetype == Material::COLOR)
		{
			collision.colorAt = material.color;
		}
		else
		{
			vec3 uv = collision.Pos - d * N;

			float u = fmod(uv.x, 1);
			float v = fmod(uv.z, 1);

			if (u < 0) {
				u = 1 + u;
			}
			if (v < 0) {
				v = 1 + v;
			}


			//if (u < 0) u = 

			//printf("x: %f, y: %f, z:%f \n", uv.x, uv.y, uv.z);

			if (material.texturetype == Material::CHECKERBOARD)
			{

				if ((u > 0.5 && v < 0.5) || (u < 0.5 && v > 0.5))
				{
					collision.colorAt = material.color;
				}
				else{
					collision.colorAt = material.color2;
				}

				/*
				//printf("u: %f, v: %f \n", u, v);
				if ((int)abs(uv.x) % 2 == (int)abs(uv.z) % 2) {
					collision.colorAt = material.color2;
				}

				else {
					collision.colorAt = material.color;
				}*/
			}
		}

		return collision;
	}
}
