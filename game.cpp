#include "precomp.h" // include (only) this in every .cpp file
#define TINYOBJLOADER_IMPLEMENTATION
#include "lib\tinyobjloader\tiny_obj_loader.h"

float frame = 0;


// -----------------------------------------------------------
// Initialize the application
// -----------------------------------------------------------
void Game::Init()
{
	/*
	AVAILABLE SCENES:
		SCENE_SIMPLE,
		SCENE_OBJ_GLASS,
		SCENE_OBJ_HALFREFLECT,
		SCENE_LIGHTING_AMBIENT,
		SCENE_LIGHTING_SPOT,
		SCENE_LIGHTING_DIRECTIONAL,
		SCENE_PERFORMANCE,
		SCENE_BEERS_LAW,
		SCENE_TEST_BVH,
		SCENE_STRESSTEST,
		SCENE_ANIMATION
	*/

	loadscene(SCENES::SCENE_OBJ_HALFREFLECT);
	
	/*
	//GPU TEST STUFF START
	float *x;
	float *xgpu;

	cudaMalloc(&xgpu, 100 * sizeof(float));
	x = new float[100];
	
	for (size_t i = 0; i < 100; i++)
	{
		x[i] = i;
	}

	cudaMemcpy(xgpu, x, 100 * sizeof(float), cudaMemcpyHostToDevice);

	testkernel << <1, 100 >> > (xgpu);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

	cudaMemcpy(x, xgpu, 100 * sizeof(float), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < 100; i++)
	{
		printf("x[%i]: %f \n", i, x[i]);
	}
	//GPU TEST STUFF END
	*/

	SSAA = false;
	camera.DoF = false;
	use_bvh = true;
	bvhdebug = false;

	mytimer.reset();
}

// -----------------------------------------------------------
// Close down application
// -----------------------------------------------------------
void Game::Shutdown()
{

}

//Random positions for the SSAA
float random1 = RandomFloat();
float random2 = RandomFloat();
float random3 = RandomFloat();
float random4 = RandomFloat();
float random5 = RandomFloat();
float random6 = RandomFloat();
float random7 = RandomFloat();
float random8 = RandomFloat();


// -----------------------------------------------------------
// Main application tick function
// -----------------------------------------------------------
void Game::Tick( float deltaTime )
{
	((int*)rayQueue)[0] = SCRHEIGHT * SCRWIDTH * 4 * Ray::SIZE; //Size of array
	((int*)rayQueue)[1] = 0; //num rays in queue

	frames++;
	raysPerFrame = 0;

	if (positionInRaysQueue != endOfRaysQueue)
	{
		printf("ERROR: positionInRaysQueue and endOfRaysQueue did not start tick at the same index.\n");
	}

	// Generate initial rays
//#pragma omp parallel for
	for (int pixely = 0; pixely < SCRHEIGHT; pixely++) for (int pixelx = 0; pixelx < SCRWIDTH; pixelx++)
	{
		Color result;

		if (SSAA) {

			//Generate 4 rays
			float* ray1 = camera.generateRayTroughVirtualScreen(pixelx + random5, pixely + random6);
			addRayToQueue(
				vec3 { ray1[0], ray1[1], ray1[2] }, vec3 { ray1[3], ray1[4], ray1[5] },
				false, 1.0f, 0, 0, pixelx, pixely, 0.25f, rayQueue
			);

			float* ray2 = camera.generateRayTroughVirtualScreen(pixelx + random1, pixely + random7);
			addRayToQueue(
				vec3{ ray2[0], ray2[1], ray2[2] }, vec3{ ray2[3], ray2[4], ray2[5] },
				false, 1.0f, 0, 0, pixelx, pixely, 0.25f, rayQueue
			);

			float* ray3 = camera.generateRayTroughVirtualScreen(pixelx + random2, pixely + random3);
			addRayToQueue(
				vec3{ ray3[0], ray3[1], ray3[2] }, vec3{ ray3[3], ray3[4], ray3[5] },
				false, 1.0f, 0, 0, pixelx, pixely, 0.25f, rayQueue
			);

			float* ray4 = camera.generateRayTroughVirtualScreen(pixelx + random8, pixely + random4);
			addRayToQueue(
				vec3{ ray4[0], ray4[1], ray4[2] }, vec3{ ray4[3], ray4[4], ray4[5] },
				false, 1.0f, 0, 0, pixelx, pixely, 0.25f, rayQueue
			);
		}
		else {

			//Generate the ray
			float* ray = camera.generateRayTroughVirtualScreen(pixelx, pixely);
			addRayToQueue(
				vec3{ ray[0], ray[1], ray[2] }, vec3{ ray[3], ray[4], ray[5] },
				false, 1.0f, 0, 0, pixelx, pixely, 1.0f, rayQueue
			);
		}
	}

	// Tracing queued rays
//#pragma omp parallel for
	//for (int i = 0; i < num_rays; i ++)
	int numRays = ((int*)rayQueue)[1];
	Collision* collisions = new Collision[numRays + 1];
	findCollisions(rayQueue, numRays, collisions); //Find all collisions

	
	float* newRays = new float[SCRWIDTH * SCRHEIGHT * 20 * Ray::SIZE];
	((int*)newRays)[0] = SCRWIDTH * SCRHEIGHT * 20 * Ray::SIZE; //queue size
	((int*)newRays)[1] = 0; //current count
	
	float* shadowRays = new float[SCRWIDTH * SCRHEIGHT * 20 * Ray::SIZE];
	((int*)shadowRays)[0] = SCRWIDTH * SCRHEIGHT * 20 * Ray::SIZE; //queue size
	((int*)shadowRays)[1] = 0; //current count
	
	
	for (size_t i = 1; i <= numRays; i++)
	{
		TraceRay(rayQueue, i, no_rays, collisions, newRays, shadowRays); //Trace all rays
	}
	printf("New rays generated: %i \n", ((int*)newRays)[1]);
	
	//TODO: Clean everything up. This is a mess..
	//TODO: Generate the shadowrays in the buffer
	//TODO: Repeat traceray for newly generated rays

	delete collisions;
	delete shadowRays;
	
	delete newRays;

	// Plotting intermediate screen buffer to screen
	for (size_t pixelx = 0; pixelx < SCRWIDTH; pixelx++) for (size_t pixely = 0; pixely < SCRHEIGHT; pixely++)
	{
		screen->Plot(pixelx, pixely, (intermediate[(int)pixelx + ((int)pixely * SCRWIDTH)] >> 8).to_uint_safe());
	}

	memset(intermediate, 0, sizeof(Color) * SCRWIDTH * SCRHEIGHT);

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
	if (keyplus) {
		camera.setZoom(camera.zoom * 1.1);
	}
	if (keymin) {
		camera.setZoom(camera.zoom / 1.1);
	}
	if (keyComma) {
		camera.setFocalPoint(camera.focalpoint / 1.1);
	}
	if (keyPeriod) {
		camera.setFocalPoint(camera.focalpoint * 1.1);
	}

	if ( animate )
	{
		//frame += deltaTime;
		frame ++;
		//Animate earth around mars
		((ParentBVH*)bvh)->translateRight = {sinf(frame * 0.1f) * 4.0f, 0, cosf(frame * 0.1f) * 4.0f};

		//Animate moon around earth
		ParentBVH* moonbvh = (ParentBVH*)((ParentBVH*)bvh)->right;
		moonbvh->translateRight = { sinf(frame * 0.5f) * 1.5f, sinf(frame * 0.5f) * 1.5f, cosf(frame * 0.5f) * 1.5f };
	}

	if (mytimer.elapsed() > 1000) {
		prevsecframes = frames;
		raysPerSecond = no_rays;
		avgFrameTime = mytimer.elapsed() / (float)frames;

		frames = 0;
		no_rays = 0;
		mytimer.reset();
	}
	float raysPerPixel = raysPerFrame / (SCRWIDTH * SCRHEIGHT);

	screen->Bar(0, 0, 150, 40, 0x000000);
	char buffer[64];
	sprintf(buffer, "No. primitives: %i", numGeometries);
	screen->Print(buffer, 1, 2, 0xffffff);
	sprintf(buffer, "FPS: %i", prevsecframes);
	screen->Print(buffer, 1, 10, 0xffffff);
	sprintf(buffer, "Avg time (ms): %.0f", avgFrameTime);
	screen->Print(buffer, 1, 18, 0xffffff);
	sprintf(buffer, "Rays/pixel: %.1f", raysPerPixel);
	screen->Print(buffer, 1, 26, 0xffffff);
	sprintf(buffer, "Rays/second: %i", raysPerSecond);
	screen->Print(buffer, 1, 34, 0xffffff);
	no_rays = 0;
	positionInRaysQueue = 0;
	numNewRays = 0;
	numShadowrays = 0;
}

void Tmpl8::Game::MouseMove( int x, int y )
{
	camera.rotate({ (float)y, (float)x, 0 });
}

//Find the nearest collision along the ray
Collision Tmpl8::Game::nearestCollision(float* ray_ptr)
{
	if (use_bvh)
	{
		//printf("BVH TRAVERSAL ");
		return bvh->Traverse(ray_ptr, bvh->root);
	}
	else
	{
		float closestdist = 0xffffff;
		Collision closest;
		closest.t = -1;

		//Loop over all primitives to find the closest collision
		for (int i = 0; i < numGeometries; i++)
		{
			//Collision collision = geometry[i]->Intersect(*ray);
			vec3 ray_origin =    { ray_ptr[Ray::OX], ray_ptr[Ray::OY], ray_ptr[Ray::OZ] };
			vec3 ray_direction = { ray_ptr[Ray::DX], ray_ptr[Ray::DY], ray_ptr[Ray::DZ] };
			Collision collision = intersectTriangle(i, ray_origin, ray_direction, triangles);
			float dist = collision.t;
			if (dist != -1 && dist < closestdist)
			{
				//Collision. Check if closest
				closest = collision;
				closestdist = dist;
			}
		}
		return closest;
	}
}

void Game::findCollisions(float* rays, int numrays, Collision* collisions)
{
	for (size_t i = 1; i <= numrays; i++)
	{
		float* rayptr = rays + (i * Ray::SIZE);
		Collision collision = nearestCollision(rayptr);
		collisions[i] = collision;
		int c = 3;
	}
}

//Trace the ray.
void Game::TraceRay( float* rays, int ray, int numrays, Collision* collisions, float* newRays, float* shadowRays )
{
	//printf("traceray");

	float* ray_ptr = rays + (ray * Ray::SIZE);
	// unpacking ray pointer
	vec3 direction = { ray_ptr[Ray::DX], ray_ptr[Ray::DY], ray_ptr[Ray::DZ] };
	bool inobj =   ray_ptr[Ray::INOBJ];
	float refind = ray_ptr[Ray::REFRIND];
	float rdepth = ray_ptr[Ray::DEPTH];
	float pixelx = ray_ptr[Ray::PIXX];
	float pixely = ray_ptr[Ray::PIXY];
	float energy = ray_ptr[Ray::ENERGY];

	// Basecase
	if ( ray_ptr[Ray::DEPTH] > MAX_RECURSION_DEPTH )
	{
		//return 0x000000;
		return;
	}

	// Collision detection
	Collision collision = collisions[ray];
	if (bvhdebug) {
		addToIntermediate(pixelx, pixely, (Color(255, 0, 0) * ray_ptr[Ray::BVHTRA]) << 3);
		return;
	}

	// if ray collides
	if ( collision.t > 0 )
	{
		// if opaque
		if (collision.other[T_REFRACTION] == 0.0f)
		{
			Color albedo, reflection;
			float specularity = collision.other[T_SPECULARITY];

			// diffuse aspect
			if ( specularity < 1.0f )
			{
				albedo = collision.colorAt * DirectIllumination(collision);
				addToIntermediate(pixelx, pixely, albedo * energy * (1 - specularity));
				//TODO: write albedo * (1 - ray.energy) to screen
			}

			// specular aspect
			if ( specularity > 0 )
			{
				vec3 newdirection = reflect(direction, collision.N);
				addRayToQueue(
					collision.Pos + 0.00001f * newdirection, // collision.Pos + 0.00001f * -collision.N
					newdirection,
					inobj,
					refind,
					0,
					rdepth + 1,
					pixelx,
					pixely,
					energy * specularity,
					newRays
				);
			}
		}
		// if transparent
		else
		{
			float n1, n2;
			if ( inobj ) n1 = refind, n2 = 1.0f;
			else				n1 = refind, n2 = collision.other[T_REFRACTION];
			float transition = n1 / n2;
			float costheta = dot( collision.N, -direction );
			float k = 1 - ( transition * transition ) * ( 1.0f - ( costheta * costheta ) );

			float Fr;
			if ( k < 0 )
			{
				// total internal reflection
				Fr = 1;
			}
			else
			{
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
				vec3 newdirection = reflect(direction, collision.N);
				//printf("a");

				addRayToQueue(
					collision.Pos + 0.00001f * newdirection, // collision.Pos + 0.00001f * -collision.N
					newdirection,
					inobj,
					refind,
					0,
					rdepth + 1,
					pixelx,
					pixely,
					energy * Fr,
					newRays
				);
			}

			// Snell refraction
			if ( Fr < 1.0f )
			{
				vec3 newdirection = transition * direction + collision.N * (transition * costheta - sqrt(k));

				addRayToQueue(
					collision.Pos + 0.00001f * newdirection,
					newdirection,
					!inobj,
					(inobj ? 1.0f : collision.other[T_REFRACTION]),
					0,
					rdepth + 1,
					pixelx,
					pixely,
					energy * (1 - Fr),
					newRays
				);

				/* // TODO: Beer's law (and mirror albedo) requires ray.energy to be a Color rather than a float
				// Beer's law
				if (ray.mediumRefractionIndex != 1.0f && collision.colorAt.to_uint() != 0xffffff)
				{
					float distance = collision.t;

					vec3 a = vec3((float)(256 - collision.colorAt.R) / 256.0f, (float)(256 - collision.colorAt.G) / 256.0f, (float)(256 - collision.colorAt.B) / 256.0f);

					refraction.R *= exp(-a.x * distance);
					refraction.G *= exp(-a.y * distance);
					refraction.B *= exp(-a.z * distance);
				}
				*/
			}
		}
	}
	// if no collision
	else
	{
		addToIntermediate(pixelx, pixely, (skybox->ColorAt(direction) << 8) * energy);
	}
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
			vec3 origin = collision.Pos + (0.00025f * L); //move away a little bit from the surface, to avoid self-collision in the outward direction.
			vec3 direction = L;
			
			float maxt = (lights[i].position.x - collision.Pos.x) / L.x; //calculate t where the shadowray hits the light source. Because we don't want to count collisions that are behind the light source.
			
			if (lights[i].type == Light::DIRECTIONAL) {
				direction = lights[i].direction;
				maxt = 5000;
			}

			if (lights[i].type == Light::SPOT) {
				if (dot(-L, lights[i].direction) < 0.9f) {
					continue;
				}
			}

			bool collided = false;


			if (use_bvh)
			{
				float ray_ptr[Ray::SIZE] = { origin.x, origin.y, origin.z, L.x, L.y, L.z};
				//Collision shadowcollision = bvh.Traverse(&shadowray, bvh.root);
				Collision shadowcollision = bvh->Traverse(ray_ptr, bvh->root);
				//Collision shadowcollision = bvh.left->Traverse(&shadowray, bvh.left->root);

				if (shadowcollision.t < maxt && shadowcollision.t != -1) collided = true;
			}
			else {
				for (int i = 0; i < numGeometries; i++)
				{
					//Check if position is reachable by lightsource
					//Collision scattercollision = geometry[i]->Intersect(shadowray, true);
					Collision scattercollision = intersectTriangle(i, origin, direction, triangles, true);
					if (scattercollision.t != -1 && scattercollision.t < maxt)
					{
						//Collision, so this ray does not reach the light source
						collided = true;
						break;
					}
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
	triangles = new float[5000 * FLOATS_PER_TRIANGLE];

	switch (scene)
	{
	case SCENE_OBJ_GLASS:
	{
		camera.rotate({ -20, 180, 0 });
		//geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, Material(Material(0.0f, 0.0f, Material::TEXTURE, new Surface("assets\\tiles.jpg"))));

		numGeometries = 0;
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
		generateBVH();
		break;
	}
	case SCENE_TRIANGLETEST:
	{
		//camera.rotate({ -20, 180, 0 });
		//geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, Material(Material(0.0f, 0.0f, Material::TEXTURE, new Surface("assets\\tiles.jpg"))));

		numGeometries = 0;
		loadobj("assets\\cube.obj", { 1.0f, 1.0f, 1.0f }, { 0, 0, 0 }, Material(1.0f, 0.0f, 0xffffff));
		loadobj("assets\\cube.obj", { 1.0f, 1.0f, 1.0f }, { -2, 0, 0 }, Material(1.0f, 0.0f, 0xffffff));
		loadobj("assets\\cube.obj", { 1.0f, 1.0f, 1.0f }, { 2, 0, 0 }, Material(1.0f, 0.0f, 0xffffff));

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
		generateBVH();
		break;
	}

	case SCENE_OBJ_HALFREFLECT:
	{
		//camera.rotate({ -40, 0, 0 });
		//geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, Material(Material(0.0f, 0.0f, Material::TEXTURE, new Surface("assets\\tiles.jpg"))));

		numGeometries = 0;
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

		generateBVH();
		break;
	}
	case SCENE_STRESSTEST:
	{
		delete triangles;
		triangles = new float[900002 * FLOATS_PER_TRIANGLE];

		//Set up the scene
		numGeometries = 0;
		//geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, Material(Material(0.0f, 0.0f, Material::TEXTURE, new Surface("assets\\tiles.jpg"))));

		for (size_t i = 0; i < 200; i++)
		{
			float ix = i % 14;
			float iy = i / 14;
			

			loadobj("assets\\MaleLow.obj", { 0.2f, -0.2f, 0.2f }, { ix * 3 - 10, 1.5f, -5 - (2 * iy) }, Material(0.0f, 0.0f, 0xffffff));
			loadobj("assets\\MaleLow.obj", { 0.2f, -0.2f, 0.2f }, { ix * 3 - 10, -2.0f, -5 - (2 * iy) }, Material(0.0f, 0.0f, 0xffffff));
			loadobj("assets\\MaleLow.obj", { 0.2f, -0.2f, 0.2f }, { ix * 3 - 10, -5.5f, -5 - (2 * iy) }, Material(0.0f, 0.0f, 0xffffff));

		}

		camera.rotate({ 30, 150, 0 });
		camera.move({ 0.0f, -5.0f, 0.0f });

		numLights = 3;
		lights = new Light[numLights];
		lights[0].position = { -5, -10, -20 };
		lights[0].color = 0xffffff;
		//lights[0].color = 0xff1111;
		lights[0].color = lights[0].color * 700;

		lights[1].position = { 5, -10, 0 };
		lights[1].color = 0xffffff;
		//lights[1].color = 0x1111ff;
		lights[1].color = lights[1].color * 700;

		lights[2].position = { -5, -10, 0 };
		lights[2].color = 0xffffff;
		//lights[2].color = 0x11ff11;
		lights[2].color = lights[2].color * 700;


		skybox = new Skybox("assets\\skybox4.jpg");
		generateBVH();
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
			//geometry[startpos] = new Triangle(vertices[0] + translate, vertices[2] + translate, vertices[1] + translate, material);

			//The first float belonging to this triangle
			int baseindex = startpos * FLOATS_PER_TRIANGLE;

			//Vertex positions
			triangles[baseindex + T_V0X] = vertices[0].x + translate.x; //v0.x
			triangles[baseindex + T_V0Y] = vertices[0].y + translate.y; //v0.y
			triangles[baseindex + T_V0Z] = vertices[0].z + translate.z; //v0.z
			triangles[baseindex + T_V1X] = vertices[1].x + translate.x; //v1.x
			triangles[baseindex + T_V1Y] = vertices[1].y + translate.y; //v1.y
			triangles[baseindex + T_V1Z] = vertices[1].z + translate.z; //v1.z
			triangles[baseindex + T_V2X] = vertices[2].x + translate.x; //v2.x
			triangles[baseindex + T_V2Y] = vertices[2].y + translate.y; //v2.y
			triangles[baseindex + T_V2Z] = vertices[2].z + translate.z; //v2.z

			//TODO: Completely remove material class?
			//Color
			triangles[baseindex + T_COLORR] = material.color.R; //R
			triangles[baseindex + T_COLORG] = material.color.G; //G
			triangles[baseindex + T_COLORB] = material.color.B; //B
			
			//Material properties
			triangles[baseindex + T_SPECULARITY] = material.specularity; //Specularity
			triangles[baseindex + T_REFRACTION] = material.refractionIndex; //Refractionindex

			//Calculate the edges, normal and D
			initializeTriangle(startpos, triangles);

			startpos++;
			numGeometries++;

			index_offset += fv;

			// per-face material
			shapes[s].mesh.material_ids[f];
		}
	}
	printf("Loadobj done. \n\n");
}

// Adds new ray to the queue of rays to be traced
void Tmpl8::Game::addRayToQueue(Ray ray)
{
	printf("hoi");
	addRayToQueue(
		ray.Origin,
		ray.Direction,
		ray.InObject,
		ray.mediumRefractionIndex,
		ray.bvhtraversals,
		ray.recursiondepth,
		ray.pixelx,
		ray.pixely,
		ray.energy,
		rayQueue
	);
}

// Adds new ray to the ray queue
void Tmpl8::Game::addRayToQueue(vec3 ori, vec3 dir, bool inObj, float refrInd, int bvhTr, int depth, int x, int y, float energy, float* queue)
{
	int queuesize = ((int*)queue)[0];
	int currentCount = ((int*)queue)[1];
	//printf("Currentcount i: %i", currentCount);
	//printf(" queue size %i \n", queuesize);

	/*if (currentCount == 0) {
		((int*)queue)[1] = 1;
		currentCount = 1; //Keep the first entry in the queue free, to save some metadata there (queuesize, currentCount)
	}*/

	// array if full
	if (currentCount + 1 > queuesize / Ray::SIZE)
	{
		printf("ERROR: Queue overflow. Rays exceeded the %d indices of queue space.\n", rayQueueSize / Ray::SIZE);
	}


	// adding ray to array
	int index = (currentCount + 1) * Ray::SIZE; //Keep the first entry in the queue free, to save some metadata there (queuesize, currentCount)
	queue[index + Ray::OX] = (float)ori.x;
	queue[index + Ray::OY] = (float)ori.y;
	queue[index + Ray::OZ] = (float)ori.z;
	queue[index + Ray::DX] = (float)dir.x;
	queue[index + Ray::DY] = (float)dir.y;
	queue[index + Ray::DZ] = (float)dir.z;
	queue[index + Ray::INOBJ] = (float)inObj;
	queue[index + Ray::REFRIND] = refrInd;
	queue[index + Ray::BVHTRA] = (float)bvhTr;
	queue[index + Ray::DEPTH] = depth;
	queue[index + Ray::PIXX] = x;
	queue[index + Ray::PIXY] = y;
	queue[index + Ray::ENERGY] = energy;

	((int*)queue)[1]++; //Current count++
	raysPerFrame++;
	//no_rays++;
}

// Returns the next ray to be traced, and updates the queue position
int Game::getRayQueuePosition()
{
	// checking for folding overflow
	if (foldedQueue && positionInRaysQueue < endOfRaysQueue)
	{
		printf("ERROR: positionInRaysQueue is somehow ahead of endOfRaysQueue.\n");
	}


	float *ray_ptr = rayQueue + positionInRaysQueue * Ray::SIZE;
	positionInRaysQueue++;
	return (positionInRaysQueue);
}

// Calculates the edges, normal, D and aabb for a triangle
void Tmpl8::Game::initializeTriangle(int i, float * triangles)
{
	int baseindex = i * FLOATS_PER_TRIANGLE;

	vec3 v0 = { triangles[baseindex + T_V0X],triangles[baseindex + T_V0Y],triangles[baseindex + T_V0Z] };
	vec3 v1 = { triangles[baseindex + T_V1X],triangles[baseindex + T_V1Y],triangles[baseindex + T_V1Z] };
	vec3 v2 = { triangles[baseindex + T_V2X],triangles[baseindex + T_V2Y],triangles[baseindex + T_V2Z] };

	vec3 e0 = v1 - v0;
	vec3 e1 = v2 - v1;
	vec3 e2 = v0 - v2;

	vec3 N = cross(e0, e1);
	N.normalize();

	float D = -dot(N, v0);

	triangles[baseindex + T_E0X] = e0.x;
	triangles[baseindex + T_E0Y] = e0.y;
	triangles[baseindex + T_E0Z] = e0.z;
	triangles[baseindex + T_E1X] = e1.x;
	triangles[baseindex + T_E1Y] = e1.y;
	triangles[baseindex + T_E1Z] = e1.z;
	triangles[baseindex + T_E2X] = e2.x;
	triangles[baseindex + T_E2Y] = e2.y;
	triangles[baseindex + T_E2Z] = e2.z;
	triangles[baseindex + T_NX] = N.x;
	triangles[baseindex + T_NY] = N.y;
	triangles[baseindex + T_NZ] = N.z;
	triangles[baseindex + T_D] = D;

	//AABB
	triangles[baseindex + T_AABBMINX] = (v0.x <= v1.x && v0.x <= v2.x ? v0.x : (v1.x <= v0.x && v1.x <= v2.x ? v1.x : v2.x));
	triangles[baseindex + T_AABBMAXX] = (v0.x >= v1.x && v0.x >= v2.x ? v0.x : (v1.x >= v0.x && v1.x >= v2.x ? v1.x : v2.x));
	triangles[baseindex + T_AABBMINY] = (v0.y <= v1.y && v0.y <= v2.y ? v0.y : (v1.y <= v0.y && v1.y <= v2.y ? v1.y : v2.y));
	triangles[baseindex + T_AABBMAXY] = (v0.y >= v1.y && v0.y >= v2.y ? v0.y : (v1.y >= v0.y && v1.y >= v2.y ? v1.y : v2.y));
	triangles[baseindex + T_AABBMINZ] = (v0.z <= v1.z && v0.z <= v2.z ? v0.z : (v1.z <= v0.z && v1.z <= v2.z ? v1.z : v2.z));
	triangles[baseindex + T_AABBMAXZ] = (v0.z >= v1.z && v0.z >= v2.z ? v0.z : (v1.z >= v0.z && v1.z >= v2.z ? v1.z : v2.z));
}

Collision intersectTriangle(int i, vec3 origin, vec3 direction, float * triangles, bool isShadowRay)
{
	int baseindex = i * FLOATS_PER_TRIANGLE;

	vec3 v0 = { 
		triangles[baseindex + T_V0X],
		triangles[baseindex + T_V0Y],
		triangles[baseindex + T_V0Z] };
	vec3 v1 = { 
		triangles[baseindex + T_V1X],
		triangles[baseindex + T_V1Y],
		triangles[baseindex + T_V1Z] };
	vec3 v2 = { 
		triangles[baseindex + T_V2X],
		triangles[baseindex + T_V2Y],
		triangles[baseindex + T_V2Z] };
	vec3 e0 = { 
		triangles[baseindex + T_E0X],
		triangles[baseindex + T_E0Y],
		triangles[baseindex + T_E0Z] };
	vec3 e1 = { 
		triangles[baseindex + T_E1X],
		triangles[baseindex + T_E1Y],
		triangles[baseindex + T_E1Z] };
	vec3 e2 = { 
		triangles[baseindex + T_E2X],
		triangles[baseindex + T_E2Y],
		triangles[baseindex + T_E2Z] };
	vec3 N = { triangles[baseindex + T_NX],
		triangles[baseindex + T_NY],
		triangles[baseindex + T_NZ] };

	float D = triangles[baseindex + T_D];


	Collision collision;
	collision.t = -1;
	float NdotR = dot(direction, N);
	if (NdotR == 0) return collision; //Ray parrallel to plane, would cause division by 0

	float t = -(dot(origin, N) + D) / (NdotR);

	//From https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
	if (t > 0)
	{
		vec3 P = origin + t * direction;
		if (dot(N, cross(e0, (P - v0))) > 0 && dot(N, cross(e1, (P - v1))) > 0 && dot(N, cross(e2, (P - v2))) > 0)
		{
			//Collision
			collision.t = t;

			if (isShadowRay) {
				return collision;
			}

			collision.colorAt.R = triangles[baseindex + T_COLORR];
			collision.colorAt.G = triangles[baseindex + T_COLORG];
			collision.colorAt.B = triangles[baseindex + T_COLORB];
			collision.other = triangles + baseindex;
			if (NdotR > 0) collision.N = -N;
			else collision.N = N;
			collision.Pos = P;
			return collision;
		}
	}
	return collision;
}

vec3 calculateTriangleAABBMidpoint(int i, float * triangles)
{
	int baseindex = i * FLOATS_PER_TRIANGLE;
	float xmin = triangles[baseindex + T_AABBMINX];
	float xmax = triangles[baseindex + T_AABBMAXX];
	float ymin = triangles[baseindex + T_AABBMINY];
	float ymax = triangles[baseindex + T_AABBMAXY];
	float zmin = triangles[baseindex + T_AABBMINZ];
	float zmax = triangles[baseindex + T_AABBMAXZ];

	return vec3((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2);
}

