#pragma once


class Camera
{
public:
	Camera(vec3 position = { 0,0,0 }, vec3 direction = { 0,0,1 }, float virtualScreenDistance = 1.2);
	~Camera();

	Ray generateRayTroughVirtualScreen(float pixelx, float pixely);

	//void setPosDir(vec3 position, vec3 direction);
	//void lookAt(vec3 direction);
	void move(vec3 direction);

	void rotate(vec3 deg);

	vec3 getDirection() {
		return direction;
	}

	vec3 getLeft() {
		return left;
	}

	vec3 getUp() {
		return up;
	}

private:
	vec3 position;
	vec3 left;
	vec3 up;

	vec3 direction;
	float virtualScreenDistance; //Change this to change the FOV

	vec3 virtualScreenCenter;

	vec3 virtualScreenCornerTL, virtualScreenCornerTR, virtualScreenCornerBL;

	float xsize = SCRWIDTH / 800.0f;
	float ysize = SCRHEIGHT / 800.0f;

};

