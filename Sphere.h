#pragma once

class Sphere : public Geometry
{
public:
	Sphere(vec3 position, float r, Material material);
	Collision Intersect(Ray ray, bool isShadowray = false) override;
	AABB GetAABB();

private:
	float r2; // r^2
	vec3 position;

	void calculateAABB() override;
};

