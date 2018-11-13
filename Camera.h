#pragma once


class Camera
{
public:
	Camera(vec3 position = { 0,0,0 }, vec3 direction = { 0,0,1 }, float virtualScreenDistance = 1);
	~Camera();

	Ray generateRayTroughVirtualScreen(int pixelx, int pixely);

	void moveTo(vec3 position, vec3 direction);

private:
	vec3 position;
	vec3 direction;
	float virtualScreenDistance; //Change this to change the FOV

	vec3 virtualScreenCenter;

	vec3 virtualScreenCornerTL, virtualScreenCornerTR, virtualScreenCornerBL;
};

