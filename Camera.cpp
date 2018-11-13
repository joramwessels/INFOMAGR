#include "precomp.h"


Camera::Camera(vec3 position, vec3 direction, float virtualScreenDistance)
{
	this->position = position;
	this->direction = direction.normalized();
	this->virtualScreenDistance = virtualScreenDistance;

	virtualScreenCenter = position + virtualScreenDistance * direction;

	//Calculate the virtual screen corners
	virtualScreenCornerTL = virtualScreenCenter + vec3(-1, -1, 0); //top left
	virtualScreenCornerTR = virtualScreenCenter + vec3(1, -1, 0); //top right
	virtualScreenCornerBL = virtualScreenCenter + vec3(-1, 1, 0); //bottom left
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

void Camera::moveTo(vec3 position, vec3 direction)
{
	this->position = position;
	this->direction = direction.normalized();
	this->virtualScreenDistance = virtualScreenDistance;

	virtualScreenCenter = position + virtualScreenDistance * direction;

	//Calculate the virtual screen corners
	virtualScreenCornerTL = virtualScreenCenter + vec3(-1, -1, 0); //top left
	virtualScreenCornerTR = virtualScreenCenter + vec3(1, -1, 0); //top right
	virtualScreenCornerBL = virtualScreenCenter + vec3(-1, 1, 0); //bottom left

}

