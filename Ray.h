#pragma once
class Geometry; //Forward declaration

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

	Color(uint R = 0, uint G = 0, uint B = 0) {
		this->B = B;
		this->G = G;
		this->R = R;
	}

	uint to_uint() {
		return ((R & 255) << 16) + ((G & 255) << 8) + (B & 255);
	}

	void from_uint(uint color) {
		B = color & 255;
		G = (color >> 8) & 255;
		R = (color >> 16) & 255;
	}

	Color operator * (const Color& operand) const { return Color(R * operand.R, G * operand.G, B * operand.B); }
	Color operator * (const float& operand) const { return Color((float)R * operand, (float)G * operand, (float)B * operand); }
	Color operator / (const Color& operand) const { return Color(R / operand.R, G / operand.G, B / operand.B); }
	Color operator / (const float& operand) const { return Color((float)R / operand, (float)G / operand, (float)B / operand); }

};

struct Collision
{
	Geometry* other;
	vec3 N;
	vec3 Pos;
	float t;
};