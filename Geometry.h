#pragma once
class Geometry
{
public:
	Geometry();
	~Geometry();
	Color color;

private:
	virtual float Intersect(Ray ray);
};

