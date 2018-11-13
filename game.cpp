#include "precomp.h" // include (only) this in every .cpp file

bool animatecamera = false;

// -----------------------------------------------------------
// Initialize the application
// -----------------------------------------------------------
void Game::Init()
{
	//Set up the scene
	numGeometries = 6;
	//geometry = new Geometry*[numGeometries];
	geometry = new Geometry*[6];
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
			uint result = (TraceRay(ray) / 255).to_uint();
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
Collision Tmpl8::Game::nearestCollision(Ray ray)
{
	float closestdist = 0xffffff;
	Collision closest;
	closest.t = -1;

	for (int i = 0; i < numGeometries; i++)
	{
		Collision collision = geometry[i]->Intersect(ray);
		float dist = collision.t;
		if (dist != -1 && dist < closestdist) {
			//Collision
			closest = collision;
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

	Collision collision = nearestCollision(ray);

	if (collision.t != -1)
	{
		color = collision.other->color * DirectIllumination(collision); // * directillumination
		int d = 7;
	}

	return color;
}

//Cast a ray from the collision point towards the light, to check if light can reach the point
Color Tmpl8::Game::DirectIllumination(Collision collision)
{
	vec3 lightposition = { -5, -5, 0 };
	//vec3 lightposition = { 0, -5, 0 };
	Color lightcolor;
	lightcolor.from_uint(0x000000);

	vec3 L = (lightposition - collision.Pos).normalized();

	Ray scatterray;
	scatterray.Direction = L;
	scatterray.Origin = collision.Pos - (0.0001 * collision.N); //move away a little bit from the surface

	bool collided = false;
	for (int i = 0; i < numGeometries; i++)
	{
		//Check if position is reachable by lightsource
		Collision scattercollision = geometry[i]->Intersect(scatterray);
		if (scattercollision.t != -1)
		{
			//This ray does not reach the light source
			collided = true;
			break;
		}
	}

	if (collided) {
		return lightcolor;
	}
	else {
		lightcolor.from_uint(0xffffff);
		return lightcolor;
	}

}
