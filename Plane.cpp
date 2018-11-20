#include "precomp.h"
#include "Plane.h"

Plane::Plane(vec3 N, float d, Material material)
{
	this->N = N.normalized();
	this->d = d;
	this->material = material;

	Xaxis.x = N.x * cosf(90 * PI / (180)) + N.z * sinf(90 * PI / (180));
	Xaxis.y = N.y;
	Xaxis.z = -(N.x * sinf(90 * PI / (180))) + N.z * cos(90 * PI / (180));

	if (Xaxis.x == N.x && Xaxis.y == N.y && Xaxis.z == N.z) {
		Xaxis.x = N.x;
		Xaxis.y = N.y * cosf(90 * PI / (180)) - N.z * sinf(90 * PI / (180));
		Xaxis.z = N.y * sinf(90 * PI / (180)) + N.z * cos(90 * PI / (180));
	}

	Yaxis = cross(N, Xaxis);

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

			float un = dot(collision.Pos, Xaxis);
			float vn = dot(collision.Pos, Yaxis);

			float u = fmod(un, 1);
			float v = fmod(vn, 1);

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
