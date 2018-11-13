#include "precomp.h" // include (only) this in every .cpp file

// -----------------------------------------------------------
// Initialize the application
// -----------------------------------------------------------
void Game::Init()
{
	sphere = new Sphere(vec3(0, 0, 8), 1, 0xff0000);
	plane = new Plane(vec3(0, 1, 0), -3.0f, 0x00ff00);
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
			Ray ray = camera.generateRayTroughVirtualScreen(pixelx, pixely);
			
			uint result = TraceRay(ray).to_uint();
			screen->Plot(pixelx, pixely, result);
			
		}
	}
}

Color Tmpl8::Game::TraceRay(Ray ray)
{
	Color color;
	color.R = 0;
	color.G = 0;
	color.B = 0;

	if (plane->Intersect(ray) != -1) {
		color = plane->color;
	}

	
	if (sphere->Intersect(ray) != -1) {
	color = sphere->color;
	}
	

	return color;
}
