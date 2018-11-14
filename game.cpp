#include "precomp.h" // include (only) this in every .cpp file

bool animatecamera = true;
int frame = 0;

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
			int hdrscale = 255;
			uint result = (TraceRay(ray) / hdrscale).to_uint_safe();
			screen->Plot(pixelx, pixely, result);
			
		}
	}

	//Just for fun ;)
	if (animatecamera)
	{
		camerapos.z += 0.01;
		camera.moveTo(camerapos, { 0,0,1 });
	}
	printf("Frame %i done. \n", frame++);
}

//Find the nearest collision along the ray
Collision Tmpl8::Game::nearestCollision(Ray ray)
{
	float closestdist = 0xffffff;
	Collision closest;
	closest.t = -1;

	//Loop over all geometries to find the closest collision
	for (int i = 0; i < numGeometries; i++)
	{
		Collision collision = geometry[i]->Intersect(ray);
		float dist = collision.t;
		if (dist != -1 && dist < closestdist) {
			//Collision. Check if closest
			closest = collision;
			closestdist = dist;
		}
	}
	return closest;
}

//Trace the ray. 
Color Tmpl8::Game::TraceRay(Ray ray)
{
	Color color;
	color.R = 0;
	color.G = 0;
	color.B = 0;

	//check if the ray collides
	Collision collision = nearestCollision(ray);

	if (collision.t != -1)
	{
		//The ray collides.
		color = collision.other->color * DirectIllumination(collision); 
	}

	//Ray out of scene
	return color;
}

//Cast a ray from the collision point towards the light, to check if light can reach the point
//TODO: make light sources dynamic. (aka create a class for them and loop over them)
//TODO: consider distance to light source.

Color Tmpl8::Game::DirectIllumination(Collision collision)
{
	vec3 lightposition = { -5, -5, 0 };
	Color lightcolor;
	lightcolor.from_uint(0x000000);

	vec3 L = (lightposition - collision.Pos).normalized();

	Ray scatterray;
	scatterray.Direction = L;
	scatterray.Origin = collision.Pos + (0.0000025f * collision.N); //move away a little bit from the surface, to avoid self-collision in the outward direction. TODO: what is the best value here?

	bool collided = false;
	for (int i = 0; i < numGeometries; i++)
	{
		//Check if position is reachable by lightsource
		Collision scattercollision = geometry[i]->Intersect(scatterray);
		if (scattercollision.t != -1)
		{
			//Collision, so this ray does not reach the light source
			collided = true;
			break;
		}
	}

	if (collided) {
		return lightcolor;
	}
	else {
		lightcolor.from_uint(0xffffff);
		lightcolor = lightcolor * 1000;
		//printf("N dot L: %f", dot(-collision.N, L));
		float r = (lightposition - collision.Pos).length();
		return lightcolor * (dot(collision.N, L) / (4 * PI * r * r));
	}

}
