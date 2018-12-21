#pragma once
class Geometry; //Forward declaration

//To support intermediate results > 255, store colors as 3 uints. This is probably terrible performance wise.
struct Color
{
	uint R = 0;
	uint G = 0;
	uint B = 0;

	Color( uint R = 0, uint G = 0, uint B = 0 )
	{
		this->B = B;
		this->G = G;
		this->R = R;
	}

	uint to_uint()
	{
		return ( ( R & 255 ) << 16 ) | ( ( G & 255 ) << 8 ) | ( B & 255 );
	}

	uint to_uint_safe()
	{
		uint Rtemp = R, Gtemp = G, Btemp = B;
		if ( R > 255 ) Rtemp = 255; //Set the last bits all to one.
		if ( G > 255 ) Gtemp = 255; //Set the last bits all to one.
		if ( B > 255 ) Btemp = 255; //Set the last bits all to one.
		return ( ( Rtemp & 255 ) << 16 ) | ( ( Gtemp & 255 ) << 8 ) | ( Btemp & 255 );
	}

	void from_uint( uint color )
	{
		B = color & 255;
		G = ( color >> 8 ) & 255;
		R = ( color >> 16 ) & 255;
	}

	Color operator*( const Color &operand ) const { return Color( R * operand.R, G * operand.G, B * operand.B ); }
	Color operator+( const Color &operand ) const { return Color( R + operand.R, G + operand.G, B + operand.B ); }
	void operator+=( const Color &operand )
	{
		R += operand.R;
		G += operand.G;
		B += operand.B;
	}
	void operator=( const uint &operand ) { from_uint( operand ); }

	Color operator*( const float &operand ) const { return Color( (float)R * operand, (float)G * operand, (float)B * operand ); }
	Color operator*(const int &operand) const { return Color(R * operand, G * operand, B * operand); }
	Color operator/( const Color &operand ) const { return Color( R / operand.R, G / operand.G, B / operand.B ); }
	Color operator/( const float &operand ) const { return Color( (float)R / operand, (float)G / operand, (float)B / operand ); }

	Color operator>>( const int &operand ) const { return Color( R >> operand, G >> operand, B >> operand ); }
	Color operator<<( const int &operand ) const { return Color( R << operand, G << operand, B << operand ); }

	void operator>>=( const int &operand )
	{
		R >> operand;
		G >> operand;
		B >> operand;
	}
	void operator<<=( const int &operand )
	{
		R << operand;
		G << operand;
		B << operand;
	}
};

struct Material
{

	enum TEXTURETYPE
	{
		COLOR,
		CHECKERBOARD,
		TEXTURE
	};

	int specularity = 0; //Range 0...256
	float refractionIndex = 0.0f;
	TEXTURETYPE texturetype = COLOR;
	Color color;  //Should be in range 0...255, to avoid problems with mirrors
	Color color2; //For checkerboard

	Surface* texture;

	Material()
	{
		specularity = 256;
		refractionIndex = 0.0f;
		color.from_uint( 0xffffff );
	}

	Material(float specularity, float refractionIndex, Color color)
	{
		this->texturetype = COLOR;
		this->specularity = specularity * 256;
		this->refractionIndex = refractionIndex;
		this->color = color;
	}
	
	Material( float specularity, float refractionIndex, uint color )
	{
		this->texturetype = COLOR;
		this->specularity = specularity * 256;
		this->refractionIndex = refractionIndex;
		this->color = color;

	}

	Material( float specularity, float refractionIndex, TEXTURETYPE texturetype, uint color1, uint color2 )
	{
		this->texturetype = texturetype;
		this->specularity = specularity * 256;
		this->refractionIndex = refractionIndex;
		this->color = color1;
		this->color2 = color2;
	}

	Material( float specularity, float refractionIndex, TEXTURETYPE texturetype, Surface *texture )
	{
		this->texturetype = texturetype;
		this->specularity = specularity * 256;
		this->refractionIndex = refractionIndex;
		this->texture = texture;
	}
};

struct Ray
{
	vec3 Origin = {0, 0, 0};
	vec3 Direction;
	bool InObject = false;
	float mediumRefractionIndex = 1.0f;
	int bvhtraversals = 0;

};

struct Collision
{
	Geometry *other;
	vec3 N;
	vec3 Pos;
	float t;
	Color colorAt;
	int bvhdepth = 0;
};

struct Light
{
	enum LIGHTTYPE {
		POINTLIGHT,
		DIRECTIONAL,
		SPOT,
		AMBIENT
	};

	LIGHTTYPE type = POINTLIGHT;
	vec3 position;
	vec3 direction;
	Color color;
	
};

class Skybox
{
public:
	Material::TEXTURETYPE type = Material::COLOR;
	Color color;
	Color color2;
	Surface* texture;

	Skybox(char* filename) {
		type = Material::TEXTURE;
		texture = new Surface(filename);
	}

	Skybox(uint color) {
		type = Material::COLOR;
		this->color = color;
	}

	Skybox(Color color1, Color color2) {
		type = Material::CHECKERBOARD;
		this->color = color1;
		this->color2 = color2;
	}

	Color ColorAt(vec3 Direction) {
		float u;
		float v;

		Color result;

		switch (type)
		{
		case Material::COLOR:
			return color;
			break;
		case Material::CHECKERBOARD:
			u = 0.5f + (atan2f(-Direction.z, -Direction.x) * INV2PI);
			v = 0.5f - (asinf(-Direction.y) * INVPI);

			if ((int)(v * 10) % 2 == (int)(u * 10) % 2)
			{
				return color2;
			}

			else
			{
				return color;
			}
			break;
		case Material::TEXTURE:
			u = 0.5f + (atan2f(-Direction.z, -Direction.x) * INV2PI);
			v = 0.5f - (asinf(-Direction.y) * INVPI);
			result = texture->GetBuffer()[(int)((texture->GetWidth() - 1) * u) + (int)((texture->GetHeight() - 1) * v) * texture->GetPitch()];
			return result;
			break;
		default:
			break;
		}
	}

};

struct AABB		// 6*4 = 24 bytes
{
	float xmin;
	float xmax;
	float ymin;
	float ymax;
	float zmin;
	float zmax;

	AABB()
	{
	}

	AABB(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
	{
		this->xmin = xmin;
		this->xmax = xmax;
		this->ymin = ymin;
		this->ymax = ymax;
		this->zmin = zmin;
		this->zmax = zmax;
	}

	vec3 Midpoint() {
		return vec3((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2);
	}

	float Area() {
		float diffx = xmax - xmin;
		float diffy = ymax - ymin;
		float diffz = zmax - zmin;

		return 2 * ((diffx * diffy) + (diffy * diffz) + (diffx * diffz));
	}

	struct AABBIntersection {
		bool intersects;
		float tEntry;
	};

	// Ray-AABB intersection algorithm found at:
	//		https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
	AABBIntersection Intersects(Ray ray)
	{
		// Intersect with the extended box edges
		float t0x = (xmin - ray.Origin.x) / ray.Direction.x;
		float t1x = (xmax - ray.Origin.x) / ray.Direction.x;
		float t0y = (ymin - ray.Origin.y) / ray.Direction.y;
		float t1y = (ymax - ray.Origin.y) / ray.Direction.y;
		// Check if the max is actually bigger than the min
		if (t0x > t1x) swap(t0x, t1x);
		if (t0y > t1y) swap(t0y, t1y);
		// If not neither the x nor y dimension intersects, there's no intersection
		if ((t0x > t1y) || (t0y > t1x)) return { false, -1};

		// Take the smallest and biggest of the x-y pairs
		float t0 = (t0y > t0x ? t0y : t0x);
		float t1 = (t1y < t1x ? t1y : t1x);
		// Intersect with the extended z box edge
		float t0z = (zmin - ray.Origin.z) / ray.Direction.z;
		float t1z = (zmax - ray.Origin.z) / ray.Direction.z;
		if (t0z > t1z) swap(t0z, t1z);
		// If it's not on the z edge, there's no intersection
		if ((t0 > t1z) || (t0z > t1) || (t1 < 0)) return {false, -1};
		
		return {true, t0};
	}

};
