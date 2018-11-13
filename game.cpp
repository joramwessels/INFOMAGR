#include "precomp.h" // include (only) this in every .cpp file

bool animatecamera = true;

// -----------------------------------------------------------
// Initialize the application
// -----------------------------------------------------------
void Game::Init()
{
	//Set up the scene
	numGeometries = 6;
	geometry = new Geometry*[numGeometries];
	geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, 0x00ff00);
	geometry[1] = new Sphere(vec3(-4.2, 0, 8), 1, 0xff0000);
	geometry[2] = new Sphere(vec3(-2.1, 0.5, 8), 1, 0xff22222);
	geometry[3] = new Sphere(vec3(0, 1.1, 8), 1, 0xff4444);
	geometry[4] = new Sphere(vec3(2.1, 1.5, 8), 1, 0xff6666);
	geometry[5] = new Sphere(vec3(4.2, 2, 8), 1, 0xff8888);

}

// -----------------------------------------------------------
// Close down application
// -----------------------------------------------------------
void Game::Shutdown()
{
}

// -----------------------------------------------------------
// Main application tick function
// -----------------------------------------------------------
void Game::Tick( float deltaTime )
{
	//Shoot a ray for every pixel
	for (int pixelx = 0; pixelx < SCRWIDTH; pixelx++)
	{
		for (int pixely = 0; pixely < SCRHEIGHT; pixely++)
		{
			//Generate the ray
			Ray ray = camera.generateRayTroughVirtualScreen(pixelx, pixely);
			
			//Trace the ray, and plot the result to the screen
			uint result = TraceRay(ray).to_uint();
			screen->Plot(pixelx, pixely, result);
			
		}
	}

	//Just for fun ;)
	if (animatecamera)
	{
		camerapos.z += 0.01;
		camera.moveTo(camerapos, { 0,0,1 });
	}
}

//Find the nearest collision along the ray
Geometry * Tmpl8::Game::nearestCollision(Ray ray)
{
	float closestdist = 0xffffff;
	Geometry* closest = nullptr;

	for (int i = 0; i < numGeometries; i++)
	{
		float dist = geometry[i]->Intersect(ray);
		if (dist != -1 && dist < closestdist) {
			//Collision
			closest = geometry[i];
			closestdist = dist;
		}
	}
	return closest;
}

//Trace the ray. TODO: add light sources and stuff
Color Tmpl8::Game::TraceRay(Ray ray)
{
	Color color;
	color.R = 0;
	color.G = 0;
	color.B = 0;

	Geometry* collision = nearestCollision(ray);

	if (collision != nullptr)
	{
		color = collision->color;
	}

	return color;
}
