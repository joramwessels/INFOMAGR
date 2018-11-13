#pragma once



class Sphere
{
public:
	Sphere(vec3 position, float r, uint color);
	float Intersect(Ray ray);
	Color color;

private:
	vec3 position;
	float r2; // r^2
};

