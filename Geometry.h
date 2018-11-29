#pragma once

//This is a base class used for geometry objects.
class Geometry
{
public:
	Geometry();
	~Geometry();
	//Color color;

	Material material;

	virtual Collision Intersect(Ray ray, bool isShadowray = false);

private:
};

