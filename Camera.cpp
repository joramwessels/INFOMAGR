#include "precomp.h"


Camera::Camera(vec3 position, vec3 direction, float virtualScreenDistance)
{
	this->position = position;
	this->direction = direction.normalized();
	this->virtualScreenDistance = virtualScreenDistance;

	virtualScreenCenter = position + virtualScreenDistance * direction;

	//Calculate the virtual screen corners
	

	virtualScreenCornerTL = virtualScreenCenter + vec3(-xsize, -ysize, 0); //top left
	virtualScreenCornerTR = virtualScreenCenter + vec3(xsize, -ysize, 0); //top right
	virtualScreenCornerBL = virtualScreenCenter + vec3(-xsize, ysize, 0); //bottom left
}


Camera::~Camera()
{
}

//Generate a ray from the camera through a pixel in the virtual screen
Ray Camera::generateRayTroughVirtualScreen(int pixelx, int pixely)
{
	vec2 pixelPosScaled;
	pixelPosScaled.x = float(pixelx) / SCRWIDTH; //Scale the pixel position to be in the range 0..1
	pixelPosScaled.y = float(pixely) / SCRHEIGHT;

	Ray ray;
	ray.Origin = position;

	vec3 positionOnVirtualScreen = virtualScreenCornerTL + pixelPosScaled.x * (virtualScreenCornerTR - virtualScreenCornerTL) + pixelPosScaled.y * (virtualScreenCornerBL - virtualScreenCornerTL);
	ray.Direction = (positionOnVirtualScreen - position).normalized();

	return ray;
}
/*
void Camera::setPosDir(vec3 position, vec3 direction)
{
	
	vec3 left = direction;
	left.rotateY(90);

	printf("Now looking in direction x: %f, y: %f, z: %f \n", direction.x, direction.y, direction.z);
	this->position = position;
	this->direction = direction.normalized();
	this->virtualScreenDistance = virtualScreenDistance;

	virtualScreenCenter = position + (virtualScreenDistance * direction);

	//Calculate the virtual screen corners
	virtualScreenCornerTL = virtualScreenCenter - xsize * left + vec3(0, -ysize, 0); //top left
	virtualScreenCornerTR = virtualScreenCenter + xsize * left + vec3(0, -ysize, 0); //top right
	virtualScreenCornerBL = virtualScreenCenter - xsize * left + vec3(0, ysize, 0); //bottom left
	
}*/
/*
void Camera::lookAt(vec3 direction) {
	setPosDir(position, direction);
}*/

void Camera::move(vec3 direction)
{
	position += direction;
	//this->direction += direction;
	virtualScreenCenter += direction;
	virtualScreenCornerBL += direction;
	virtualScreenCornerTL += direction;
	virtualScreenCornerTR += direction;
}

void Camera::rotate(vec3 deg) {
	//TODO Fix this mess
	//Rotate around world-Y axis
	direction.rotateY(deg.y);

	//virtualScreenCenter.rotateY(deg.y);
	//virtualScreenCornerTL.rotateY(deg.y);
	//virtualScreenCornerTR.rotateY(deg.y);
	//virtualScreenCornerBL.rotateY(deg.y);
	
	vec3 left = direction;
	left.rotateY(90);
	left.y = 0;
	left.normalize();

	//printf("left x: %f, y: %f, z: %f \n", left.x, left.y, left.z);

	//virtualScreenCornerTL = virtualScreenCenter - xsize * left + vec3(0, -ysize, 0); //top left
	//virtualScreenCornerTR = virtualScreenCenter + xsize * left + vec3(0, -ysize, 0); //top right
	//virtualScreenCornerBL = virtualScreenCenter - xsize * left + vec3(0, ysize, 0); //bottom left

	mat4 rotationmatrix = mat4::rotate(-left, (deg.x * PI / 180));

	
	vec4 dir = { direction, 0 };
	dir = rotationmatrix * dir;
	direction = { dir.x, dir.y, dir.z };
	direction.normalize();
	


	vec3 up = cross(direction, left);
	//up.normalize();
	virtualScreenCenter = position + (virtualScreenDistance * direction);

	virtualScreenCornerTL = virtualScreenCenter - (xsize * left) - (ysize * up); //top left
	virtualScreenCornerTR = virtualScreenCenter + (xsize * left) - (ysize * up); //top right
	virtualScreenCornerBL = virtualScreenCenter - (xsize * left) + (ysize * up); //bottom left

	printf("TL y: %f, TR y: %f \n", virtualScreenCornerTL.y, virtualScreenCornerTR.y);

	/*

	vec3 down = { 0, 1, 0 };
	float angle = dot(direction, down);

	//Prevent looking all the way up or down because something weird happens with the calculation
	if (!((angle > 0.8 && deg.x > 0) || (angle < -0.8 && deg.x < 0)))
	{
		//Rotate around local x axis
		mat4 rotationmatrix = mat4::rotate(-left, (deg.x * PI / 180));

		vec4 sc = { virtualScreenCenter, 0 };
		sc = rotationmatrix * sc;
		virtualScreenCenter = { sc.x, sc.y, sc.z };
		virtualScreenCenter.normalize();

		vec4 dir = { direction, 0 };
		dir = rotationmatrix * dir;
		direction = { dir.x, dir.y, dir.z };
		direction.normalize();

		vec4 tl = { virtualScreenCornerTL, 0 };
		tl = rotationmatrix * tl;
		virtualScreenCornerTL = { tl.x, tl.y, tl.z };
		virtualScreenCornerTL;

		vec4 tr = { virtualScreenCornerTR, 0 };
		tr = rotationmatrix * tr;
		virtualScreenCornerTR = { tr.x, tr.y, tr.z };
		virtualScreenCornerTR;

		vec4 bl = { virtualScreenCornerBL, 0 };
		bl = rotationmatrix * bl;
		virtualScreenCornerBL = { bl.x, bl.y, bl.z };
		virtualScreenCornerBL;

		printf("looking at x: %f y: %f z: %f \n", direction.x, direction.y, direction.z);
	}*/
}
