#include "precomp.h" // include (only) this in every .cpp file
#define TINYOBJLOADER_IMPLEMENTATION
#include "lib\tinyobjloader\tiny_obj_loader.h"


bool animatecamera = false;
int frame = 0;

// -----------------------------------------------------------
// Initialize the application
// -----------------------------------------------------------
void Game::Init()
{
	loadscene(BEERS_LAW);
	SSAA = true;

	mytimer.reset();
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
	frames++;

	//Shoot a ray for every pixel
#pragma omp parallel for
	for (int pixely = 0; pixely < SCRHEIGHT; pixely++)
	{
		for (int pixelx = 0; pixelx < SCRWIDTH; pixelx++)
		{
			Color result;

			if (SSAA) {
				Ray ray = camera.generateRayTroughVirtualScreen(pixelx, pixely);
				Ray ray2 = camera.generateRayTroughVirtualScreen(pixelx + 0.5f, pixely);
				Ray ray3 = camera.generateRayTroughVirtualScreen(pixelx + 0.5f, pixely + 0.5f);
				Ray ray4 = camera.generateRayTroughVirtualScreen(pixelx, pixely + 0.5f);

				result = (TraceRay(ray) + TraceRay(ray2) + TraceRay(ray3) + TraceRay(ray4)) >> 2;
			}
			else {
				//Generate the ray
				Ray ray = camera.generateRayTroughVirtualScreen(pixelx, pixely);

				//Trace the ray, and plot the result to the screen
				int hdrscale = 255;
				result = TraceRay(ray);

			}
			
			
			
			screen->Plot(pixelx, pixely, (result >> 8).to_uint_safe());
			
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

	if (mytimer.elapsed() > 1000) {
		prevsecframes = frames;
		frames = 0;
		mytimer.reset();
	}

	char buffer[64];
	sprintf(buffer, "FPS: %i", prevsecframes);

	screen->Bar(0, 0, 45, 8, 0x000000);
	screen->Print(buffer, 1, 2, 0xffffff);
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
		if (collision.other->material.refractionIndex == 0.0f) {
			// Non-transparant objects
			Color albedo, reflection;
			int specularity = collision.other->material.specularity;
			if ( specularity < 256 )
			{
				// Diffuse aspect
				albedo = collision.colorAt * DirectIllumination(collision);
			}
			if ( specularity > 0 )
			{
				// Reflective aspect
				Ray reflectedray;
				reflectedray.Direction = reflect( ray.Direction, collision.N );
				reflectedray.Origin = collision.Pos + 0.00001f * reflectedray.Direction;
				reflection = TraceRay( reflectedray, recursiondepth + 1 );
			}
			return ( albedo * ( 256 - specularity ) + reflection * specularity ) >> 8; //Fixed point-math
		}
		else
		{
			// Transparant objects
			float n1, n2;
			if ( ray.InObject ) n1 = ray.mediumRefractionIndex, n2 = 1.0f;
			else				n1 = ray.mediumRefractionIndex, n2 = collision.other->material.refractionIndex;
			float transition = n1 / n2;
			float costheta = dot( collision.N, -ray.Direction );
			float k = 1 - ( transition * transition ) * ( 1.0f - ( costheta * costheta ) );

			float Fr;
			if ( k < 0 )
			{
				// total internal reflection
				//return Color( 0, 0, 0 );
				Fr = 1;
			}
			else {

				float ndiff = n1 - n2;
				float nsumm = n1 + n2;
				float temp = ndiff / nsumm;

				float R0 = temp * temp;
				Fr = R0 + ( 1.0f - R0 ) * powf( 1.0f - costheta, 5.0f );
			}

			// Fresnel reflection (Schlick's approximation)
			Color reflection, refraction;
			if ( Fr > 0.0f )
			{
				Ray reflectedray;
				reflectedray.Direction = reflect( ray.Direction, collision.N );
				reflectedray.Origin = collision.Pos + 0.001f * reflectedray.Direction;
				reflection = TraceRay( reflectedray, recursiondepth + 1 );
			}

			// Snell refraction
			if ( Fr < 1.0f )
			{
				Ray refractedray;
				refractedray.Direction = transition * ray.Direction + collision.N * ( transition * costheta - sqrt( k ) );
				refractedray.Origin = collision.Pos + 0.001f * refractedray.Direction;
				refractedray.InObject = !ray.InObject;
				refractedray.mediumRefractionIndex = ( ray.InObject ? 1.0f : collision.other->material.refractionIndex ); // Exiting an object defaults material to air
				refraction = TraceRay( refractedray, recursiondepth + 1 );

				// Beer's law
				if (ray.mediumRefractionIndex != 1.0f)
				{
					float distance = collision.t;// (collision.Pos - ray.Origin).length();
					float x = 1000000.0f;
					//Color a = Color(log(collision.colorAt.R/255.0f), log(collision.colorAt.G / 255.0f), log(collision.colorAt.B / 255.0f));
					Color a = Color(0.2f, 1.8, 1.8);// Color(log(0), log(x), log(x));
					refraction.R *= exp(-a.R * distance);
					refraction.G *= exp(-a.G * distance);
					refraction.B *= exp(-a.B * distance);
					//refraction = refraction >> 8;
				}
			}

			return ( ( refraction * ( 1 - Fr ) + reflection * Fr ) );
		}
	}
	else {
		//There was no collision.
		//--> skybox.
		/*vec3 skyBoxN = ray.Direction;

		float u = 0.5f + (atan2f(-skyBoxN.z, -skyBoxN.x) * INV2PI);
		float v = 0.5f - (asinf(-skyBoxN.y) * INVPI);
		color = skybox->GetBuffer()[(int)((skybox->GetWidth() - 1) * u) + (int)((skybox->GetHeight() - 1) * v) * skybox->GetPitch()];
		//color <<= 8;
		*/
		color.R = 255;
		color.G = 0;
		color.B = 0;

		return skybox->ColorAt(ray.Direction) << 8;
	}

	//Ray out of scene
	return color;
}

//Cast a ray from the collision point towards the light, to check if light can reach the point

Color Tmpl8::Game::DirectIllumination( Collision collision )
{
	Color result = 0x000000;

	for ( int i = 0; i < numLights; i++ )
	{
		if (lights[i].type != Light::AMBIENT)
		{
			vec3 L = (lights[i].position - collision.Pos).normalized();


			Ray shadowray;
			shadowray.Direction = L;
			shadowray.Origin = collision.Pos + (0.00025f * L); //move away a little bit from the surface, to avoid self-collision in the outward direction. TODO: what is the best value here?
			float maxt = (lights[i].position.x - collision.Pos.x) / L.x; //calculate t where the shadowray hit the light source. Because we don't want to count collisions that are behind the light source.

			
			if (lights[i].type == Light::DIRECTIONAL) {
				shadowray.Direction = lights[i].direction;
				maxt = 5000;
			}

			if (lights[i].type == Light::SPOT) {
				if (dot(-L, lights[i].direction) < 0.9f) {
					continue;
				}
			}

			bool collided = false;

			for (int i = 0; i < numGeometries; i++)
			{
				//Check if position is reachable by lightsource
				Collision scattercollision = geometry[i]->Intersect(shadowray, true);
				if (scattercollision.t != -1 && scattercollision.t < maxt)
				{
					//Collision, so this ray does not reach the light source
					collided = true;
					break;
				}
			}


			if (collided)
			{
				continue;
			}
			else
			{
				if (lights[i].type == Light::SPOT || lights[i].type == Light::POINTLIGHT)
				{
					float r = (lights[i].position - collision.Pos).length();
					result += lights[i].color * (max(0.0f, dot(collision.N, L)) * INV4PI / (r * r));
				}
				else {
					//DIRECTIONAL, don't use quadratic falloff
					result += lights[i].color * (max(0.0f, dot(collision.N, lights[i].direction)));
				}
			}
		}
		else {
			//AMBIENT LIGHT: just return the light color
			result += lights[i].color;
		}
	}
	return result;
}

vec3 Tmpl8::Game::reflect( vec3 D, vec3 N )
{
	return D - 2 * ( dot( D, N ) ) * N;
}

void Tmpl8::Game::loadscene(SCENES scene)
{
	geometry = new Geometry*[5000];

	switch (scene)
	{
	case SCENE_SIMPLE:
	{
		//Set up the scene
		numGeometries = 9;

		//geometry = new Geometry*[6];
		geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, Material(Material(0.0f, 0.0f, Material::TEXTURE, new Surface("assets\\tiles.jpg"))));

		geometry[1] = new Sphere(vec3(-4.2, 0, 8), 1, Material(0.0f, 1.1f, 0xffffff));
		geometry[2] = new Sphere(vec3(-2.1, 0.5, 8), 1, Material(0.0f, 0.0f, 0xff000f));
		geometry[3] = new Sphere( vec3( 0, 1.1, 8 ), 1, Material( 0.0f, 0.0f, Material::TEXTURE, new Surface( "assets\\earthmap1k.jpg" ) ) );
		geometry[4] = new Sphere(vec3(0, -1.5, 12), 1, Material(1.0f, 0.0f, 0xffffff));
		geometry[5] = new Sphere( vec3( 2.1, 1.5, 8 ), 1, Material( 0.3f, 0.0f, 0xffffff ) );
		geometry[6] = new Sphere( vec3( 4.2, 0, 8 ), 1, Material( 0.0f, 0.0f, Material::CHECKERBOARD, 0x000000, 0xffffff ) );

		geometry[7] = new Sphere( vec3( 4.2, 0, 0 ), 1, Material( 0.0f, 0.0f, Material::CHECKERBOARD, 0x000000, 0xff0000 ) );
		geometry[8] = new Sphere( vec3( 3, 0, -8 ), 1, Material( 0.0f, 0.0f, Material::CHECKERBOARD, 0x000000, 0x00ff00 ) );
		//geometry[9] = new Sphere(vec3(-4.2, 0, 0), 1, Material(Material::DIFFUSE, Material::CHECKERBOARD, 0x000000, 0x0000ff));

		//geometry[10] = new Triangle({ -3, -1.4, 0 }, { -1, -1.4, -1 }, { -0.5, -1.4, 1 }, Material(Material::DIFFUSE, 0xff1111));

		numLights = 3;
		lights = new Light[numLights];
		lights[0].position = { -5, -5, 20 };
		lights[0].color = 0xffffff;
		//lights[0].color = 0xff1111;
		lights[0].color = lights[0].color * 700;

		lights[1].position = { 5, -5, 0 };
		lights[1].color = 0xffffff;
		//lights[1].color = 0x1111ff;
		lights[1].color = lights[1].color * 700;

		lights[2].position = { -5, -5, 0 };
		lights[2].color = 0xffffff;
		//lights[2].color = 0x11ff11;
		lights[2].color = lights[2].color * 700;

		skybox = new Skybox("assets\\skybox4.jpg");


		break;
	}
	case SCENE_LIGHTING_AMBIENT:
	{
		//Set up the scene
		numGeometries = 9;

		//geometry = new Geometry*[6];
		geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, Material(Material(0.0f, 0.0f, Material::TEXTURE, new Surface("assets\\tiles.jpg"))));

		geometry[1] = new Sphere(vec3(-4.2, 0, 8), 1, Material(0.0f, 1.52f, 0xffffff));
		geometry[2] = new Sphere(vec3(-2.1, 0.5, 8), 1, Material(0.0f, 0.0f, 0xff000f));
		geometry[3] = new Sphere(vec3(0, 1.1, 8), 1, Material(0.0f, 0.0f, Material::TEXTURE, new Surface("assets\\earthmap1k.jpg")));
		geometry[4] = new Sphere(vec3(0, -1.5, 12), 1, Material(1.0f, 0.0f, 0xffffff));
		geometry[5] = new Sphere(vec3(2.1, 1.5, 8), 1, Material(0.3f, 0.0f, 0xffffff));
		geometry[6] = new Sphere(vec3(4.2, 0, 8), 1, Material(0.0f, 0.0f, Material::CHECKERBOARD, 0x000000, 0xffffff));

		geometry[7] = new Sphere(vec3(4.2, 0, 0), 1, Material(0.0f, 0.0f, Material::CHECKERBOARD, 0x000000, 0xff0000));
		geometry[8] = new Sphere(vec3(3, 0, -8), 1, Material(0.0f, 0.0f, Material::CHECKERBOARD, 0x000000, 0x00ff00));
		//geometry[9] = new Sphere(vec3(-4.2, 0, 0), 1, Material(Material::DIFFUSE, Material::CHECKERBOARD, 0x000000, 0x0000ff));

		//geometry[10] = new Triangle({ -3, -1.4, 0 }, { -1, -1.4, -1 }, { -0.5, -1.4, 1 }, Material(Material::DIFFUSE, 0xff1111));


		//geometry[10] = new Triangle({ -3, -1.4, 0 }, { -1, -1.4, -1 }, { -0.5, -1.4, 1 }, Material(Material::DIFFUSE, 0xff1111));

		numLights = 1;
		lights = new Light[numLights];
		lights[0].position = { -5, -5, 20 };
		lights[0].color = 0xffffff;
		//lights[0].color = 0xff1111;
		lights[0].color = lights[0].color;
		lights[0].type = Light::AMBIENT;

		skybox = new Skybox("assets\\skybox4.jpg");


		break;
	}
	case SCENE_LIGHTING_SPOT:
	{
		//Set up the scene
		numGeometries = 9;

		//geometry = new Geometry*[6];
		geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, Material(Material(0.0f, 0.0f, Material::TEXTURE, new Surface("assets\\tiles.jpg"))));

		geometry[1] = new Sphere(vec3(-4.2, 0, 8), 1, Material(0.0f, 1.52f, 0xffffff));
		geometry[2] = new Sphere(vec3(-2.1, 0.5, 8), 1, Material(0.0f, 0.0f, 0xff000f));
		geometry[3] = new Sphere(vec3(0, 1.1, 8), 1, Material(0.0f, 0.0f, Material::TEXTURE, new Surface("assets\\earthmap1k.jpg")));
		geometry[4] = new Sphere(vec3(0, -1.5, 12), 1, Material(1.0f, 0.0f, 0xffffff));
		geometry[5] = new Sphere(vec3(2.1, 1.5, 8), 1, Material(0.3f, 0.0f, 0xffffff));
		geometry[6] = new Sphere(vec3(4.2, 0, 8), 1, Material(0.0f, 0.0f, Material::CHECKERBOARD, 0x000000, 0xffffff));

		geometry[7] = new Sphere(vec3(4.2, 0, 0), 1, Material(0.0f, 0.0f, Material::CHECKERBOARD, 0x000000, 0xff0000));
		geometry[8] = new Sphere(vec3(3, 0, -8), 1, Material(0.0f, 0.0f, Material::CHECKERBOARD, 0x000000, 0x00ff00));
		//geometry[9] = new Sphere(vec3(-4.2, 0, 0), 1, Material(Material::DIFFUSE, Material::CHECKERBOARD, 0x000000, 0x0000ff));

		//geometry[10] = new Triangle({ -3, -1.4, 0 }, { -1, -1.4, -1 }, { -0.5, -1.4, 1 }, Material(Material::DIFFUSE, 0xff1111));


		//geometry[10] = new Triangle({ -3, -1.4, 0 }, { -1, -1.4, -1 }, { -0.5, -1.4, 1 }, Material(Material::DIFFUSE, 0xff1111));

		numLights = 1;
		lights = new Light[numLights];
		lights[0].position = { 0, -2, -7 };
		lights[0].color = 0xffffff;
		//lights[0].color = 0xff1111;
		lights[0].color = lights[0].color * 7000;
		lights[0].type = Light::SPOT;
		lights[0].direction = vec3(0, 0.5f, 1).normalized();

		skybox = new Skybox("assets\\skybox4.jpg");


		break;
	}
	case SCENE_OBJ_GLASS:
	{
		camera.rotate({ -20, 180, 0 });
		geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, Material(Material(0.0f, 0.0f, Material::TEXTURE, new Surface("assets\\tiles.jpg"))));

		numGeometries = 1;
		loadobj("assets\\MaleLow.obj", { 0.5f, -0.5f, 0.5f }, { 0, 1.5f, -9 }, Material(0.0f, 1.52f, 0xffffff));

		numLights = 3;
		lights = new Light[numLights];
		lights[0].position = { -5, -5, 20 };
		lights[0].color = 0xffffff;
		//lights[0].color = 0xff1111;
		lights[0].color = lights[0].color * 700;

		lights[1].position = { 5, -5, 0 };
		lights[1].color = 0xffffff;
		//lights[1].color = 0x1111ff;
		lights[1].color = lights[1].color * 700;

		lights[2].position = { -5, -5, 0 };
		lights[2].color = 0xffffff;
		//lights[2].color = 0x11ff11;
		lights[2].color = lights[2].color * 700;

		skybox = new Skybox("assets\\skybox4.jpg");

		break;
	}
	case SCENE_OBJ_HALFREFLECT:
	{
		//camera.rotate({ -40, 0, 0 });
		geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, Material(Material(0.0f, 0.0f, Material::TEXTURE, new Surface("assets\\tiles.jpg"))));

		numGeometries = 1;
		loadobj("assets\\Banana.obj", { 0.02f, -0.02f, 0.02f }, { -2.5, 1.5f, 10 }, Material(0.5f, 0.0f, 0xffff00));

		numLights = 3;
		lights = new Light[numLights];
		lights[0].position = { -5, -5, 20 };
		lights[0].color = 0xffffff;
		//lights[0].color = 0xff1111;
		lights[0].color = lights[0].color * 700;

		lights[1].position = { 5, -5, 0 };
		lights[1].color = 0xffffff;
		//lights[1].color = 0x1111ff;
		lights[1].color = lights[1].color * 700;

		lights[2].position = { -5, -5, 0 };
		lights[2].color = 0xffffff;
		//lights[2].color = 0x11ff11;
		lights[2].color = lights[2].color * 700;

		skybox = new Skybox("assets\\skybox4.jpg");
		break;
	}
	case SCENE_LIGHTING_DIRECTIONAL:
	{
		//Set up the scene
		numGeometries = 9;

		//geometry = new Geometry*[6];
		geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, Material(Material(0.0f, 0.0f, Material::TEXTURE, new Surface("assets\\tiles.jpg"))));

		geometry[1] = new Sphere(vec3(-4.2, 0, 8), 1, Material(0.0f, 1.52f, 0xffffff));
		geometry[2] = new Sphere(vec3(-2.1, 0.5, 8), 1, Material(0.0f, 0.0f, 0xff000f));
		geometry[3] = new Sphere(vec3(0, 1.1, 8), 1, Material(0.0f, 0.0f, Material::TEXTURE, new Surface("assets\\earthmap1k.jpg")));
		geometry[4] = new Sphere(vec3(0, -1.5, 12), 1, Material(1.0f, 0.0f, 0xffffff));
		geometry[5] = new Sphere(vec3(2.1, 1.5, 8), 1, Material(0.3f, 0.0f, 0xffffff));
		geometry[6] = new Sphere(vec3(4.2, 0, 8), 1, Material(0.0f, 0.0f, Material::CHECKERBOARD, 0x000000, 0xffffff));

		geometry[7] = new Sphere(vec3(4.2, 0, 0), 1, Material(0.0f, 0.0f, Material::CHECKERBOARD, 0x000000, 0xff0000));
		geometry[8] = new Sphere(vec3(3, 0, -8), 1, Material(0.0f, 0.0f, Material::CHECKERBOARD, 0x000000, 0x00ff00));
		//geometry[9] = new Sphere(vec3(-4.2, 0, 0), 1, Material(Material::DIFFUSE, Material::CHECKERBOARD, 0x000000, 0x0000ff));

		//geometry[10] = new Triangle({ -3, -1.4, 0 }, { -1, -1.4, -1 }, { -0.5, -1.4, 1 }, Material(Material::DIFFUSE, 0xff1111));


		//geometry[10] = new Triangle({ -3, -1.4, 0 }, { -1, -1.4, -1 }, { -0.5, -1.4, 1 }, Material(Material::DIFFUSE, 0xff1111));

		numLights = 1;
		lights = new Light[numLights];
		lights[0].color = 0x888888;
		//lights[0].color = 0xff1111;
		lights[0].type = Light::DIRECTIONAL;
		lights[0].direction = vec3(-1, -0.5f, -1).normalized();

		skybox = new Skybox("assets\\skybox4.jpg");
		break;
	}
	case SCENE_PERFORMANCE:
	{
		//Set up the scene
		numGeometries = 3;

		//geometry = new Geometry*[6];
		geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, Material(0.0f, 0.0f, Material::CHECKERBOARD, 0xff0000, 0xffff00));

		geometry[1] = new Sphere(vec3(-4.2, 0, 8), 1, Material(0.0f, 0.95f, 0xffffff));
		geometry[2] = new Sphere(vec3(-2.1, 0.5, 12), 1, Material(0.3f, 0.0f, 0xffffff));

		numLights = 2;
		lights = new Light[numLights];
		lights[0].position = { -5, -5, 0 };
		lights[0].color = 0xffffff;
		//lights[0].color = 0xff1111;
		lights[0].color = lights[0].color * 700;

		lights[1].position = { 5, -5, 0 };
		lights[1].color = 0x333333;
		lights[1].type = Light::AMBIENT;
		//lights[1].color = 0x1111ff;
		//lights[1].color = lights[1].color * 700;


		skybox = new Skybox(0x58caeb);
		camera.move({ -4, 0.3, 3 });
		camera.rotate({ 0, 0.001, 0 }); //Otherwise very very ugly aliasing because of the checkerboard. Now only very ugly aliasing.

		break;
	}
	case BEERS_LAW:
	{
		//Set up the scene
		numGeometries = 4;

		//geometry = new Geometry*[6];
		geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, Material(Material(0.0f, 0.0f, Material::TEXTURE, new Surface("assets\\tiles.jpg"))));

		geometry[1] = new Sphere(vec3(-3, 1, 8), 0.25, Material(0.0f, 1.52f, 0xff0f0f));
		geometry[2] = new Sphere(vec3(-1, 0.5, 8), 1, Material(0.0f, 1.52f, 0xff0f0f));
		geometry[3] = new Sphere(vec3(3, -1.5, 8), 3, Material(0.0f, 1.52f, 0xff0f0f));

		numLights = 3;
		lights = new Light[numLights];
		lights[0].position = { -5, -5, 20 };
		lights[0].color = 0xffffff;
		//lights[0].color = 0xff1111;
		lights[0].color = lights[0].color * 700;

		lights[1].position = { 5, -5, 0 };
		lights[1].color = 0xffffff;
		//lights[1].color = 0x1111ff;
		lights[1].color = lights[1].color * 700;

		lights[2].position = { -5, -5, 0 };
		lights[2].color = 0xffffff;
		//lights[2].color = 0x11ff11;
		lights[2].color = lights[2].color * 700;

		skybox = new Skybox("assets\\skybox4.jpg");


		break;
	}

	default:
		break;
	}
}

void Game::loadobj(string filename, vec3 scale, vec3 translate, Material material)
{
	int startpos = numGeometries;

	tinyobj::attrib_t attrib;
	vector<tinyobj::shape_t> shapes;
	vector<tinyobj::material_t> materials;

	string error;
	string warn;
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &error, filename.c_str());

	if (!error.empty()) { // `err` may contain warning message.
		printf("Tinyobjloader error: %s", error);
	}

	printf("loaded %i shapes \n", shapes.size());
	for (size_t shape = 0; shape < shapes.size(); shape++)
	{
		printf("loaded %i faces \n", shapes[shape].mesh.num_face_vertices.size());

	}

	//From https://github.com/syoyo/tinyobjloader
	// Loop over shapes

	for (size_t s = 0; s < shapes.size(); s++) {
		// Loop over faces(polygon)
		size_t index_offset = 0;
		//for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {

		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) { //Only do 12 triangles for now
			int fv = shapes[s].mesh.num_face_vertices[f];

			vec3 vertices[3];
			//vec3 normals[3];

			// Loop over vertices in the face.
			for (size_t v = 0; v < fv; v++) {
				// access to vertex

				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
				tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
				tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
				tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
				//tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
				//tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
				//tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];
				tinyobj::real_t tx = attrib.texcoords[2 * idx.texcoord_index + 0];
				tinyobj::real_t ty = attrib.texcoords[2 * idx.texcoord_index + 1];

				//vec3 scale = { 0.000000005f, -0.000000005f, 0.000000005f };
				vertices[v] = vec3(vx, vy, vz);
				vertices[v] *= scale;
				//normals[v] = { nx, ny, nz };
				//printf("Vertice %i: %f, %f, %f, fv: %i \n", v, vx, vy, vz, fv);

				// Optional: vertex colors
				// tinyobj::real_t red = attrib.colors[3*idx.vertex_index+0];
				// tinyobj::real_t green = attrib.colors[3*idx.vertex_index+1];
				// tinyobj::real_t blue = attrib.colors[3*idx.vertex_index+2];
			}
			geometry[startpos] = new Triangle(vertices[0] + translate, vertices[2] + translate, vertices[1] + translate, material);
			startpos++;
			numGeometries++;

			index_offset += fv;

			// per-face material
			shapes[s].mesh.material_ids[f];
		}
	}
	printf("Loadobj done.");
}
