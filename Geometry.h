#pragma once
class Geometry
{
public:
	Geometry();
	~Geometry();
	Color color;

	virtual float Intersect(Ray ray);

private:
};

