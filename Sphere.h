#pragma once

class Sphere : public Geometry
{
public:
	Sphere(vec3 position, float r, uint color);
	float Intersect(Ray ray) override;

private:
	float r2; // r^2
	vec3 position;
};

