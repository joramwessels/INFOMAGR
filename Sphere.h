#pragma once
struct Ray
{
	vec3 Origin = { 0,0,0 };
	vec3 Direction;
};

//To support intermediate results > 255, store colors as 3 uints. This is probably terrible performance wise.
struct Color
{
	uint R = 0;
	uint G = 0;
	uint B = 0;

	uint to_uint() {
		return ((R & 255) << 16) + ((G & 255) << 8) + (B & 255);
	}

	void from_uint(uint color) {
		B = color & 255;
		G = (color >> 8) & 255;
		R = (color >> 16) & 255;
	}
};


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

