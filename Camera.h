#pragma once


class Camera
{
public:
	Camera(vec3 position = { 0,0,0 }, vec3 direction = { 0,0,1 }, float virtualScreenDistance = 3);
	~Camera();

	float* generateRayTroughVirtualScreen(float pixelx, float pixely);

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

	void calculateVirtualScreenCorners();

	void setFocalPoint(float f);
	void setZoom(float z);
	float zoom = 1.0f; //READ-ONLY! Use setZoom() to change!
	float focalpoint; //READ-ONLY! Use setFocalPoint() to change!

	bool DoF = false;

	vec3 position;
	vec3 virtualScreenCornerTL, virtualScreenCornerTR, virtualScreenCornerBL;
private:
	vec3 left;
	vec3 up;

	vec3 direction;

	vec3 virtualScreenCenter;


	float xsize = SCRWIDTH / 800.0f;
	float ysize = SCRHEIGHT / 800.0f;

};

