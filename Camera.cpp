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

void Camera::move(vec3 direction)
{
	position += direction;

	virtualScreenCenter += direction;
	virtualScreenCornerBL += direction;
	virtualScreenCornerTL += direction;
	virtualScreenCornerTR += direction;
}

void Camera::rotate(vec3 deg) {
	//Rotate around world-Y axis
	direction.rotateY(deg.y);

	left = direction;
	left.y = 0;
	left.rotateY(90);
	left.normalize();

	//Rotate around 'left' vector, aka local X axis
	mat4 rotationmatrix = mat4::rotate(-left, (deg.x * PI / 180));
	
	vec4 dir = { direction, 0 };
	dir = rotationmatrix * dir;
	direction = { dir.x, dir.y, dir.z };
	direction.normalize();

	up = cross(direction, left);
	virtualScreenCenter = position + (virtualScreenDistance * direction);

	virtualScreenCornerTL = virtualScreenCenter - (xsize * left) - (ysize * up); //top left
	virtualScreenCornerTR = virtualScreenCenter + (xsize * left) - (ysize * up); //top right
	virtualScreenCornerBL = virtualScreenCenter - (xsize * left) + (ysize * up); //bottom left
}
