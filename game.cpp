#include "precomp.h" // include (only) this in every .cpp file

bool animatecamera = false;
int frame = 0;

// -----------------------------------------------------------
// Initialize the application
// -----------------------------------------------------------
void Game::Init()
{

	//Set up the scene
	numGeometries = 11;
	geometry = new Geometry*[numGeometries];
	//geometry = new Geometry*[6];
	geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, Material(Material(Material::DIFFUSE, Material::TEXTURE, new Surface("assets\\tiles.jpg"))));
	//geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, Material(Material(Material::DIFFUSE, Material::CHECKERBOARD, 0xffffff, 0x000000)));
	//geometry[1] = new Sphere(vec3(-4.2, 0, 8), 1, Material(Material::DIFFUSE, Material::CHECKERBOARD, 0x000000, 0xffffff));
	//geometry[1] = new Sphere(vec3(-4.2, 0, 8), 1, Material(Material::GLASS, Material::TEXTURE, new Surface("assets\\earthmap1k.jpg")));
	geometry[1] = new Sphere(vec3(-4.2, 0, 8), 1, Material(Material::GLASS, 0xffffff));
	geometry[2] = new Sphere(vec3(-2.1, 0.5, 8), 1, Material(Material::DIFFUSE, 0xff000f));
	geometry[3] = new Sphere(vec3(0, 1.1, 8), 1, Material(Material::DIFFUSE, Material::TEXTURE, new Surface("assets\\earthmap1k.jpg")));
	geometry[4] = new Sphere(vec3(0, -1.5, 12), 1, Material(Material::MIRROR, 0xffffff));
	geometry[5] = new Sphere(vec3(2.1, 1.5, 8), 1, Material(Material::DIFFUSE, 0xffffff));
	geometry[6] = new Sphere(vec3(4.2, 0, 8), 1, Material(Material::DIFFUSE, Material::CHECKERBOARD, 0x000000, 0xffffff));
	
	geometry[7] = new Sphere(vec3(4.2, 0, 0), 1, Material(Material::DIFFUSE, Material::CHECKERBOARD, 0x000000, 0xff0000));
	geometry[8] = new Sphere(vec3(0, 0, -8), 1, Material(Material::DIFFUSE, Material::CHECKERBOARD, 0x000000, 0x00ff00));
	geometry[9] = new Sphere(vec3(-4.2, 0, 0), 1, Material(Material::DIFFUSE, Material::CHECKERBOARD, 0x000000, 0x0000ff));
	
	geometry[10] = new Triangle({ -3, -1.4, 0 }, { -1, -1.4, -1 }, { -0.5, -1.4, 1 }, Material(Material::DIFFUSE, 0xff1111));

	numLights = 3;
	lights = new Light[numLights];
	lights[0].position = { -5, -5, 20 };
	lights[0].color = 0xffffff;
	//lights[0].color = 0xff1111;
	lights[0].color = lights[0].color * 700;
	
	lights[1].position = {5, -5, 0 };
	lights[1].color = 0xffffff;
	//lights[1].color = 0x1111ff;
	lights[1].color = lights[1].color * 700;

	lights[2].position = { -5, -5, 0 };
	lights[2].color = 0xffffff;
	//lights[2].color = 0x11ff11;
	lights[2].color = lights[2].color * 700;
	
	skybox = new Surface("assets\\skybox4.jpg");

	//camera.lookAt({ 1, 0, 0 });

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
#pragma omp parallel for
	for (int pixelx = 0; pixelx < SCRWIDTH; pixelx++)
	{
		for (int pixely = 0; pixely < SCRHEIGHT; pixely++)
		{
			//Generate the ray
			Ray ray = camera.generateRayTroughVirtualScreen(pixelx, pixely);
			
			//Trace the ray, and plot the result to the screen
			int hdrscale = 255;
			uint result = (TraceRay(ray) >> 8).to_uint_safe();
			screen->Plot(pixelx, pixely, result);
			
		}
	}

	if (keyW) {
		camera.move(camera.getDirection() * 0.1f);
	}
	
	if (keyS) {
		camera.move(camera.getDirection() * -0.1f);
	}

	if (keyD) {
		camera.move(camera.getLeft() * 0.1f);
	}

	if (keyA) {
		camera.move(camera.getLeft() * -0.1f);
	}
	
	if (keyQ) {
		camera.move(camera.getUp() * 0.1f);
	}

	if (keyE) {
		camera.move(camera.getUp() * -0.1f);
	}

	//Just for fun ;)
	if ( animatecamera )
	{
		camera.move({ 0,0,0.01 });
	}
	//printf("Frame %i done. \n", frame++);
}

void Tmpl8::Game::MouseMove( int x, int y )
{
	camera.rotate({ (float)y, (float)x, 0 });
	
}

//Find the nearest collision along the ray
Collision Tmpl8::Game::nearestCollision(Ray ray)
{
	float closestdist = 0xffffff;
	Collision closest;
	closest.t = -1;

	//Loop over all geometries to find the closest collision
	for ( int i = 0; i < numGeometries; i++ )
	{
		Collision collision = geometry[i]->Intersect( ray );
		float dist = collision.t;
		if ( dist != -1 && dist < closestdist )
		{
			//Collision. Check if closest
			closest = collision;
			closestdist = dist;
		}
	}
	return closest;
}

//Trace the ray.
Color Tmpl8::Game::TraceRay( Ray ray, int recursiondepth )
{
	if ( recursiondepth > MAX_RECURSION_DEPTH )
	{
		return 0x000000;
	}

	Color color; //sky color

	/*color.R = 25500;
	color.G = 25500;
	color.B = 25500;*/

	//check if the ray collides
	Collision collision = nearestCollision( ray );

	if ( collision.t != -1 )
	{
		//The ray collides.
		if ( collision.other->material.type == Material::DIFFUSE )
		{
			color = collision.colorAt * DirectIllumination(collision);
		}
		if ( collision.other->material.type == Material::MIRROR )
		{
			Ray reflectedray;
			reflectedray.Direction = reflect( ray.Direction, collision.N );
			reflectedray.Origin = collision.Pos + 0.00001f * reflectedray.Direction;
			return ( collision.colorAt * TraceRay( reflectedray, recursiondepth + 1 ) ) >> 8; //Devide by 255 to scale back into the same range, after multiplying by material color.
		}
		if ( collision.other->material.type == Material::GLASS )
		{
			float n1, n2;
			if ( ray.InObject ) n1 = refractionIndex( ray.Medium ), n2 = refractionIndex( Material::AIR );
			else				n1 = refractionIndex( ray.Medium ), n2 = refractionIndex( collision.other->material.type );
			float transition = n1 / n2;
			float costheta = dot( collision.N, -ray.Direction );
			float k = 1 - ( transition * transition ) * ( 1.0f - ( costheta * costheta ) );
			if ( k < 0 )
			{
				// total internal reflection
				return Color( 0, 0, 0 );
			}

			// Fresnel reflection (Schlick's approximation)
			float ndiff = n1 - n2;
			float nsumm = n1 + n2;
			float temp = ndiff / nsumm;

			float R0 = temp * temp;
			float Fr = R0 + ( 1.0f - R0 ) * powf( 1.0f - costheta, 5.0f );
			Color reflection;
			if ( Fr > 0.0f )
			{
				Ray reflectedray;
				reflectedray.Direction = reflect( ray.Direction, collision.N );
				reflectedray.Origin = collision.Pos + 0.00001f * reflectedray.Direction;
				reflection = TraceRay( reflectedray, recursiondepth + 1 );
			}
			else
			{
				reflection = Color( 0, 0, 0 );
			}

			// Snell refraction
			Color refraction;
			if ( Fr < 1.0f )
			{
				Ray refractedray;
				refractedray.Direction = transition * ray.Direction + collision.N * ( transition * costheta - sqrt( k ) );
				refractedray.Origin = collision.Pos + 0.00001f * refractedray.Direction;
				refractedray.InObject = !ray.InObject;
				refractedray.Medium = ( ray.InObject ? Material::AIR : collision.other->material.type ); // Exiting an object defaults material to air
				refraction = TraceRay( refractedray, recursiondepth + 1 );
			}
			else
			{
				refraction = Color( 0, 0, 0 );
			}

			return ( collision.colorAt * ( refraction * ( 1 - Fr ) + reflection * Fr ) ) >> 8;
		}
	}
	else {
		//There was no collision.
		//--> skybox.
		vec3 skyBoxN = ray.Direction;

		float u = 0.5 + (atan2f(-skyBoxN.z, -skyBoxN.x) / (2 * PI));
		float v = 0.5 - (asinf(-skyBoxN.y) / PI);
		color = skybox->GetBuffer()[(int)((skybox->GetWidth() - 1) * u) + (int)((skybox->GetHeight() - 1) * v) * skybox->GetPitch()];
		//color <<= 8;

		//color.R = 255;
		//color.G = 0;
		//color.B = 0;

		return color << 8;
	}

	//Ray out of scene
	return color;
}

//Cast a ray from the collision point towards the light, to check if light can reach the point
//TODO: make light sources dynamic. (aka create a class for them and loop over them)
//TODO: consider distance to light source.

Color Tmpl8::Game::DirectIllumination( Collision collision )
{
	Color result = 0x000000;

	for ( int i = 0; i < numLights; i++ )
	{

		vec3 L = ( lights[i].position - collision.Pos ).normalized();

		Ray scatterray;
		scatterray.Direction = L;
		scatterray.Origin = collision.Pos + ( 0.00025f * collision.N ); //move away a little bit from the surface, to avoid self-collision in the outward direction. TODO: what is the best value here?

		bool collided = false;
		for ( int i = 0; i < numGeometries; i++ )
		{
			//Check if position is reachable by lightsource
			Collision scattercollision = geometry[i]->Intersect( scatterray, true );
			if ( scattercollision.t != -1 )
			{
				//Collision, so this ray does not reach the light source
				collided = true;
				break;
			}
		}

		if ( collided )
		{
			continue;
		}
		else
		{
			//lightcolor = lightcolor * 1000;
			//printf("N dot L: %f", dot(-collision.N, L));
			float r = ( lights[i].position - collision.Pos ).length();
			result += lights[i].color * ( max( 0.0f, dot( collision.N, L ) ) / ( 4 * PI * r * r ) );
		}
	}
	return result;
}

vec3 Tmpl8::Game::reflect( vec3 D, vec3 N )
{
	return D - 2 * ( dot( D, N ) ) * N;
}

float Tmpl8::Game::refractionIndex( int medium )
{
	if ( medium == Material::AIR ) return 1.0f;
	if ( medium == Material::GLASS ) return 1.52f;
}
