#include "precomp.h"
#include "Plane.h"

Plane::Plane(vec3 N, float d, Material material)
{
	this->N = N.normalized();
	this->d = d;
	this->material = material;

	//Rotate the normal 90 degrees along the Y axis to get the X-axis local to the plane. Used for textures
	Xaxis.x = N.x * cosf(90 * PI / (180)) + N.z * sinf(90 * PI / (180));
	Xaxis.y = N.y;
	Xaxis.z = -(N.x * sinf(90 * PI / (180))) + N.z * cos(90 * PI / (180));

	//If Xaxis == N then we need to rotate along a different axis
	if (Xaxis.x == N.x && Xaxis.y == N.y && Xaxis.z == N.z) {
		Xaxis.x = N.x;
		Xaxis.y = N.y * cosf(90 * PI / (180)) - N.z * sinf(90 * PI / (180));
		Xaxis.z = N.y * sinf(90 * PI / (180)) + N.z * cosf(90 * PI / (180));
	}

	Yaxis = cross(N, Xaxis);
	calculateAABB();
}

Plane::~Plane()
{
}

Collision Plane::Intersect(Ray ray, bool isShadowray)
{
	Collision collision;
	collision.t = -1;

	float NdotR = dot(ray.Direction, N);
	if (NdotR == 0) return collision;

	float t = -(dot(ray.Origin, N) + d) / (NdotR);

	if (t < 0 || isinf(t)) {
		return collision;
	}
	else
	{
		if (isShadowray) {
			collision.t = 2; //We just want to know if it hits *something*
			return collision;
		}


		if (NdotR > 0) collision.N = -N;
		else collision.N = N; 

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

			if (material.texturetype == Material::CHECKERBOARD)
			{

				if ((u > 0.5 && v < 0.5) || (u < 0.5 && v > 0.5))
				{
					collision.colorAt = material.color;
				}
				else{
					collision.colorAt = material.color2;
				}
			}
			if (material.texturetype == Material::TEXTURE) {
					collision.colorAt = material.texture->GetBuffer()[(int)(material.texture->GetWidth() * u) + (int)(material.texture->GetHeight() * v) * material.texture->GetPitch()];
				
			}
		}

		return collision;
	}
}

void Plane::calculateAABB()
{
	// TODO
	aabb = AABB(-20, 20, -20, 20, -3, -3);
	//aabb = AABB(0,1,0,1,0,1);
}