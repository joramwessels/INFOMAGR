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
		return ((R & 255) << 16) | ((G & 255) << 8) | (B & 255);
	}

	uint to_uint_safe() {
		uint Rtemp = R, Gtemp = G, Btemp = B;
		if (R > 255) Rtemp = 255; //Set the last bits all to one.
		if (G > 255) Gtemp = 255; //Set the last bits all to one.
		if (B > 255) Btemp = 255; //Set the last bits all to one.
		return ((Rtemp & 255) << 16) | ((Gtemp & 255) << 8) | (Btemp & 255);
	}

	void from_uint(uint color) {
		B = color & 255;
		G = (color >> 8) & 255;
		R = (color >> 16) & 255;
	}

	Color operator * (const Color& operand) const { return Color(R * operand.R, G * operand.G, B * operand.B); }
	Color operator + (const Color& operand) const { return Color(R + operand.R, G + operand.G, B + operand.B); }
	void operator += (const Color& operand) {R += operand.R; G += operand.G; B += operand.B;}
	void operator = (const uint& operand) {from_uint(operand);}

	Color operator * (const float& operand) const { return Color((float)R * operand, (float)G * operand, (float)B * operand); }
	Color operator / (const Color& operand) const { return Color(R / operand.R, G / operand.G, B / operand.B); }
	Color operator / (const float& operand) const { return Color((float)R / operand, (float)G / operand, (float)B / operand); }
	
	Color operator >> (const int& operand) const { return Color(R >> operand, G >> operand, B >> operand); }
	Color operator << (const int& operand) const { return Color(R << operand, G << operand, B << operand); }
	
	void operator >>=( const int &operand ){ R >> operand; G >> operand; B >> operand;}
	void operator <<=( const int &operand ){ R << operand; G << operand; B << operand;}
};

struct Material
{
	enum MATERIALTYPE {
		DIFFUSE,
		MIRROR,
		GLASS
	};

	enum TEXTURETYPE {
		COLOR,
		CHECKERBOARD,
		TEXTURE
	};

	MATERIALTYPE type = DIFFUSE;
	TEXTURETYPE texturetype = COLOR;
	Color color; //Should be in range 0...255, to avoid problems with mirrors
	Color color2; //For checkerboard

	Material()
	{
		type = DIFFUSE;
		color.from_uint(0xffffff);
	}

	Material(MATERIALTYPE type, Color color)
	{
		this->texturetype = COLOR;
		this->type = type;
		this->color = color;
	}
	
	Material(MATERIALTYPE type, uint color) {
		this->texturetype = COLOR;
		this->type = type;
		this->color = color;

	}

	Material(MATERIALTYPE type, TEXTURETYPE texturetype, uint color1, uint color2) {
		this->texturetype = texturetype;
		this->type = type;
		this->color = color1;
		this->color2 = color2;

	}
};

struct Collision
{
	Geometry* other;
	vec3 N;
	vec3 Pos;
	float t;
	Color colorAt;
};

struct Light
{
	vec3 position;
	Color color;
};