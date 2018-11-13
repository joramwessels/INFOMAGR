#include "precomp.h" // include (only) this in every .cpp file

// -----------------------------------------------------------
// Initialize the application
// -----------------------------------------------------------
void Game::Init()
{
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
			
			Color result = TraceRay(ray);
			uint color = ((result.R & 255) << 16) + ((result.G & 255) << 8) + (result.B & 255);
			screen->Plot(pixelx, pixely, color);
			
		}
	}
}

Color Tmpl8::Game::TraceRay(Ray ray)
{
	Color color;
	color.R = 0;
	color.G = 0;
	color.B = 0;

	return color;
}
