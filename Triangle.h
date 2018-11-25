#pragma once

class Triangle :
	public Geometry
{
public:
	Triangle(vec3 v0, vec3 v1, vec3 v2, Material material);
	Triangle(vec3 v0, vec3 v1, vec3 v2, vec3 N, Material material);
	~Triangle();

	Collision Intersect(Ray ray, bool shatterray = false) override;


private:
	vec3 v0, v1, v2;
	vec3 e0, e1, e2;
	vec3 N;
	float D; //Distance plane triangle to origin
};

