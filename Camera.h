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
};


class Camera
{
public:
	Camera(vec3 position = { 0,0,0 }, vec3 direction = { 0,0,1 }, float virtualScreenDistance = 1);
	~Camera();

	Ray generateRayTroughVirtualScreen(int pixelx, int pixely);

private:
	vec3 position;
	vec3 direction;
	float virtualScreenDistance; //Change this to change the FOV

	vec3 virtualScreenCenter;

	vec3 virtualScreenCornerTL, virtualScreenCornerTR, virtualScreenCornerBL;
};

