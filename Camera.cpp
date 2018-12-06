#include "precomp.h"


Camera::Camera(vec3 position, vec3 direction, float virtualScreenDistance)
{
	this->position = position;
	this->direction = direction.normalized();
	this->focalpoint = virtualScreenDistance;

	calculateVirtualScreenCorners();
}


Camera::~Camera()
{
}

//Generate a ray from the camera through a pixel in the virtual screen
Ray Camera::generateRayTroughVirtualScreen(float pixelx, float pixely)
{
	vec2 pixelPosScaled;
	pixelPosScaled.x = pixelx / SCRWIDTH; //Scale the pixel position to be in the range 0..1
	pixelPosScaled.y = pixely / SCRHEIGHT;

	vec3 DofRandomness = { 0, 0, 0 };
	if (DoF) DofRandomness = vec3((RandomFloat() * 0.1 - 0.05), (RandomFloat() * 0.1 - 0.05), 0); //TODO: make random and maybe 7-gon instead of square?

	Ray ray;
	ray.Origin = position + DofRandomness;

	vec3 positionOnVirtualScreen = virtualScreenCornerTL + pixelPosScaled.x * (virtualScreenCornerTR - virtualScreenCornerTL) + pixelPosScaled.y * (virtualScreenCornerBL - virtualScreenCornerTL);
	ray.Direction = (positionOnVirtualScreen - ray.Origin).normalized();

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

	//Rotate around 'left' vector, aka local X axis
	mat4 rotationmatrix = mat4::rotate(-left, (deg.x * PI / 180));
	
	vec4 dir = { direction, 0 };
	dir = rotationmatrix * dir;
	direction = { dir.x, dir.y, dir.z };
	direction.normalize();

	calculateVirtualScreenCorners();
}

void Camera::calculateVirtualScreenCorners()
{
	left = direction;
	left.y = 0;
	left.rotateY(90);
	left.normalize();

	up = cross(direction, left);

	float zoomscale = focalpoint / zoom;
	
	virtualScreenCenter = position + (focalpoint * direction);

	virtualScreenCornerTL = virtualScreenCenter - (xsize * left * zoomscale) - (ysize * up * zoomscale); //top left
	virtualScreenCornerTR = virtualScreenCenter + (xsize * left * zoomscale) - (ysize * up * zoomscale); //top right
	virtualScreenCornerBL = virtualScreenCenter - (xsize * left * zoomscale) + (ysize * up * zoomscale); //bottom left
}

void Camera::setFocalPoint(float f)
{
	focalpoint = f;
	//printf("Set camera focal point to %f \n", focalpoint);

	calculateVirtualScreenCorners();
}

void Camera::setZoom(float z)
{
	zoom = z;
	//printf("Set camera zoom to %f \n", zoom);

	calculateVirtualScreenCorners();
}
