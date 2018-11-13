#pragma once

//This is a base class used for geometry objects.
class Geometry
{
public:
	Geometry();
	~Geometry();
	Color color;

	virtual Collision Intersect(Ray ray);

private:
};

