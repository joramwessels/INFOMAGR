#pragma once

//This is a base class used for geometry objects.
class Geometry
{
public:
	Geometry();
	~Geometry();
	Color color;

	virtual float Intersect(Ray ray);

private:
};

