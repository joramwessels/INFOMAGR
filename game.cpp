#include "precomp.h" // include (only) this in every .cpp file

float frame = 0;

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
// Initialize the application
// -----------------------------------------------------------
void Game::Init()
{
	/*
	AVAILABLE SCENES:
		SCENE_OBJ_GLASS
		SCENE_OBJ_HALFREFLECT
		SCENE_STRESSTEST
		SCENE_TRIANGLETEST
		SCENE_FLOORONLY
		SCENE_CUBE
	*/

	scene = new SceneManager();
	scene->loadScene(SceneManager::SCENE_OBJ_HALFREFLECT, camera);
	generateBVH();

	SSAA = false;
	camera.DoF = false;
	use_bvh = true;
	bvhdebug = false;

	mytimer.reset();

	((int*)rayQueue)[0] = rayQueueSize; //queue size, there can be at most just as many rays as we have pixels
	((int*)rayQueue)[1] = 0; //Num rays currently in this queue
	((int*)rayQueue)[2] = 0; //Counter used by generateprimaryray
	((int*)rayQueue)[3] = 0; //counter used by findcollisions
	((int*)rayQueue)[4] = 0; //counter used by traceray
	((int*)newRays)[0] = rayQueueSize; //queue size
	((int*)newRays)[1] = 0; //current count
	((int*)shadowRays)[0] = rayQueueSize * 5; //queue size, can be more than the number of pixels (for instance, half reflecting objects)
	((int*)shadowRays)[1] = 0; //current count

	// Moving everything to the GPU
	cudaMalloc(&g_rayQueue, rayQueueSize * sizeof(float));
	cudaMalloc(&g_newRays, rayQueueSize * sizeof(float));
	cudaMalloc(&g_collisions, rayQueueSize * sizeof(Collision));
	cudaMalloc(&g_shadowRays, rayQueueSize * sizeof(float) * 5);
	cudaMalloc(&g_intermediate, SCRWIDTH * SCRHEIGHT * sizeof(g_Color));
	cudaMalloc(&g_screen, SCRWIDTH * SCRHEIGHT * sizeof(uint));

	cudaMemcpy(g_newRays, newRays, sizeof(float) * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(g_shadowRays, shadowRays, sizeof(float) * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(g_rayQueue, rayQueue, sizeof(float) * 5, cudaMemcpyHostToDevice);

	scene->moveSceneToGPU();

	printf("rand: %f \n", random1);
	printf("rand: %f \n", random2);
	printf("rand: %f \n", random3);
	printf("rand: %f \n", random4);
	printf("rand: %f \n", random5);
	printf("rand: %f \n", random6);
	printf("rand: %f \n", random7);
	printf("rand: %f \n", random8);

}

// -----------------------------------------------------------
// Close down application
// -----------------------------------------------------------
void Game::Shutdown()
{

}



inline void CheckCudaError(int i)
{
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("%i CUDA error: %s\n", i, cudaGetErrorString(error));
		printf("exiting...");
		std::cin.get(); exit(-1);;
	}
}


// -----------------------------------------------------------
// Main application tick function
// -----------------------------------------------------------
void Game::Tick(float deltaTime)
{
	bool useCPU = false;
	if (useCPU)
	{
		//Generate primary rays (on GPU)
		cudaMemcpy(g_rayQueue, rayQueue, rayQueueSize * sizeof(float), cudaMemcpyHostToDevice);
		float3 camPos = make_float3(camera.position.x, camera.position.y, camera.position.z);
		float3 TL = make_float3(camera.virtualScreenCornerTL.x, camera.virtualScreenCornerTL.y, camera.virtualScreenCornerTL.z);
		float3 TR = make_float3(camera.virtualScreenCornerTR.x, camera.virtualScreenCornerTR.y, camera.virtualScreenCornerTR.z);
		float3 BL = make_float3(camera.virtualScreenCornerBL.x, camera.virtualScreenCornerBL.y, camera.virtualScreenCornerBL.z);

		GeneratePrimaryRay << <24, 255 >> > (g_rayQueue, camera.DoF, camPos, TL, TR, BL, SSAA);
		CheckCudaError(1);
		
		//Copy the primary rays from the gpu
		cudaMemcpy(rayQueue, g_rayQueue, rayQueueSize * sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		bool finished = false;
		while (!finished)
		{
			int numRays = ((int*)rayQueue)[1];

			//Find collisions. Put in array 'collisions'
			findCollisions(rayQueue, numRays, collisions); //Find all collisions

			//Evaluate all found collisions. Generate shadowrays and ray extensions (reflections, refractions)
			for (int i = 1; i <= numRays; i++)
			{
				TraceRay(rayQueue, i, numRays, collisions, newRays, shadowRays); //Trace all rays
			}

			//Trace the shadowrays.
			int numShadowRays = ((int*)shadowRays)[1];
			printf("Num shadowrays: %i \n", numShadowRays);
			for (int i = 1; i <= numShadowRays; i++)
			{
				TraceShadowRay(shadowRays, i);
			}

			//Flip the arrays
			float* temp = rayQueue;
			rayQueue = newRays;
			newRays = temp;

			((int*)newRays)[1] = 0; //set new ray count to 0
			((int*)shadowRays)[1] = 0; //set new shadowray count to 0

			if (((int*)rayQueue)[1] == 0) finished = true;

		}
		for (size_t pixelx = 0; pixelx < SCRWIDTH; pixelx++)
		{
			for (size_t pixely = 0; pixely < SCRHEIGHT; pixely++)
			{
				screen->Plot(pixelx, pixely, (intermediate[(int)pixelx + ((int)pixely * SCRWIDTH)] >> 8).to_uint_safe());
			}
		}
		memset(intermediate, 0, SCRWIDTH * SCRHEIGHT * sizeof(Color));
	}

	else { //use the GPU
		//Generate primary rays (on GPU)
		//cudaMemcpy(g_rayQueue, rayQueue, rayQueueSize * sizeof(float), cudaMemcpyHostToDevice);
		float3 camPos = make_float3(camera.position.x, camera.position.y, camera.position.z);
		float3 TL = make_float3(camera.virtualScreenCornerTL.x, camera.virtualScreenCornerTL.y, camera.virtualScreenCornerTL.z);
		float3 TR = make_float3(camera.virtualScreenCornerTR.x, camera.virtualScreenCornerTR.y, camera.virtualScreenCornerTR.z);
		float3 BL = make_float3(camera.virtualScreenCornerBL.x, camera.virtualScreenCornerBL.y, camera.virtualScreenCornerBL.z);
		cudaMemset(g_rayQueue + 1, 0, sizeof(uint) * 4);

		GeneratePrimaryRay << <24, 255 >> > (g_rayQueue, camera.DoF, camPos, TL, TR, BL, SSAA);
		CheckCudaError(1);

		bool finished = false;
		while (!finished)
		{
			int numRays = ((int*)rayQueue)[1];

			//Find collisions. Put in array 'collisions'
			cudaMemset(g_rayQueue + 2, 0, sizeof(uint) * 3);
			g_findCollisions << <24, 255 >> > (scene->g_triangles, scene->numGeometries, g_rayQueue, g_collisions, use_bvh, g_BVH, g_orderedIndices);
			CheckCudaError(10);

			
			//Set the ray counters for the new rays and shadowrays to 0
			cudaMemset(g_shadowRays + 1, 0, sizeof(uint) * 2);
			cudaMemset(g_newRays + 1, 0, sizeof(uint));

			g_Tracerays << <24, 255 >> > (g_rayQueue, g_collisions, g_newRays, g_shadowRays, bvhdebug, g_intermediate, scene->numLights, scene->g_lightPos, scene->g_lightColor, scene->g_skybox, scene->skybox->texture->GetWidth(), scene->skybox->texture->GetHeight(), scene->skybox->texture->GetPitch());
			CheckCudaError(15);

			cudaMemcpyAsync(rayQueue, g_rayQueue, sizeof(uint) * 2, cudaMemcpyDeviceToHost);
			g_traceShadowRays<<<24, 255>>>(g_shadowRays, scene->g_triangles, g_intermediate, g_BVH, g_orderedIndices, scene->numGeometries, use_bvh);
			CheckCudaError(17);

			//cudaMemcpy(intermediate, g_intermediate, sizeof(Color) * SCRWIDTH * SCRHEIGHT, cudaMemcpyDeviceToHost);
			CheckCudaError(15);

			//Flip the arrays
			float* temp = g_rayQueue;
			g_rayQueue = g_newRays;
			g_newRays = temp;

			//Get the new ray count from the gpu
			if (((int*)rayQueue)[1] == 0) finished = true;
		}

		// Plotting intermediate screen buffer to screen
		copyIntermediateToScreen<<<SCRWIDTH, SCRHEIGHT>>>(g_screen, g_intermediate, screen->GetPitch());
		cudaMemcpy(screen->GetBuffer(), g_screen, SCRWIDTH * SCRHEIGHT * sizeof(uint), cudaMemcpyDeviceToHost);
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

	frames++;
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
	sprintf(buffer, "No. primitives: %i", scene->numGeometries);
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
}

void Game::MouseMove(int x, int y)
{
	camera.rotate({ (float)y, (float)x, 0 });
}

//Find the nearest collision along the ray
Collision Game::nearestCollision(float* ray_ptr)
{
	if (use_bvh)
	{
		//printf("BVH TRAVERSAL ");
		//return bvh->Traverse(ray_ptr, bvh->root);
		int* stack = new int[32];
		float* aabbEntryPoints = new float[32];
		aabbEntryPoints[2] = -5000.0f;

		stack[0] = 1; //count, next one to evaluate;
		stack[1] = 0; //Root node

		Collision closest;
		closest.t = -1;

		while (stack[0] > 0)
		{
			int next = stack[0]--;
			//printf("next: stack[%i]: %i. AABB entrypoint: %f \n", next, stack[next], aabbEntryPoints[next]);

			if (closest.t != -1 && closest.t < aabbEntryPoints[next]) {
				continue;
				//printf("%f \n", aabbEntryPoints[next]);
				//delete stack;
				//delete aabbEntryPoints;
				//return closest;
			}
			
			Collision newcollision = TraverseBVHNode(ray_ptr, bvh->pool, bvh->orderedIndices, bvh->scene, stack[next], stack, aabbEntryPoints);

			if ((newcollision.t != -1 && newcollision.t < closest.t) || closest.t == -1) {
				closest = newcollision;
				//printf("closest t now %f \n", closest.t);
			}
		}
		delete stack;
		delete aabbEntryPoints;

		//printf("done.");
		return closest;


		//return TraverseBVHNode(ray_ptr, bvh->pool, bvh->orderedIndices, bvh->scene, 0);
	}
	else
	{
		float closestdist = 0xffffff;
		Collision closest;
		closest.t = -1;

		//Loop over all primitives to find the closest collision
		for (int i = 0; i < scene->numGeometries; i++)
		{
			//Collision collision = geometry[i]->Intersect(*ray);
			vec3 ray_origin = { ray_ptr[R_OX], ray_ptr[R_OY], ray_ptr[R_OZ] };
			vec3 ray_direction = { ray_ptr[R_DX], ray_ptr[R_DY], ray_ptr[R_DZ] };
			Collision collision = intersectTriangle(i, ray_origin, ray_direction, scene->triangles);
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

void Game::findCollisions(float* rayQueue, int numrays, Collision* collisions)
{
	/*for (size_t i = 1; i <= numrays; i++)
	{
		float* rayptr = rays + (i * R_SIZE);
		Collision collision = nearestCollision(rayptr);
		collisions[i] = collision;
	}*/


	int numRays = ((uint*)rayQueue)[1];
	((uint*)rayQueue)[2] = 1;
	printf("numrays: %i", numRays);

	//numRays = 10;
	//((uint*)rayQueue)[2] = 0; //current ray to be traced


	uint id = ((uint*)rayQueue)[2]++;
	//printf("id: %i \n", id);

	while (id < numRays)
	{
		//rintf("now doing ray %i from thread %i %i \n", id, pixelx, pixely);
		float* rayptr = rayQueue + (id * R_SIZE);
		Collision nearestcollision = nearestCollision(rayptr);
		collisions[id] = nearestcollision;
		id = ((uint*)rayQueue)[2]++;
	}





}

//Trace the ray.
void Game::TraceRay(float* rays, int ray, int numrays, Collision* collisions, float* newRays, float* shadowRays)
{
	//printf("traceray");

	float* ray_ptr = rays + (ray * R_SIZE);
	// unpacking ray pointer
	vec3 direction = { ray_ptr[R_DX], ray_ptr[R_DY], ray_ptr[R_DZ] };
	bool inobj = ray_ptr[R_INOBJ];
	float refind = ray_ptr[R_REFRIND];
	float rdepth = ray_ptr[R_DEPTH];
	float pixelx = ray_ptr[R_PIXX];
	float pixely = ray_ptr[R_PIXY];
	float energy = ray_ptr[R_ENERGY];

	// Basecase
	if (ray_ptr[R_DEPTH] > MAX_RECURSION_DEPTH)
	{
		//return 0x000000;
		return;
	}

	// Collision detection
	Collision collision = collisions[ray];
	if (bvhdebug) {
		addToIntermediate(pixelx, pixely, (Color(255, 0, 0) * ray_ptr[R_BVHTRA]) << 3);
		return;
	}

	// if ray collides
	if (collision.t > 0)
	{
		// if opaque
		if (collision.refraction == 0.0f)
		{
			Color albedo, reflection;
			float specularity = collision.specularity;

			// diffuse aspect
			if (specularity < 1.0f)
			{
				//Generate shadow rays
				for (int light = 0; light < scene->numLights; light++)
				{
					vec3 lightPosition = vec3(scene->lightPos[light * 3 + 0], scene->lightPos[light * 3 + 1], scene->lightPos[light * 3 + 2]);
					vec3 direction = (lightPosition - collision.Pos).normalized();
					vec3 origin = collision.Pos + (0.00025f * direction); //move away a little bit from the surface, to avoid self-collision in the outward direction.
					float maxt = (scene->lightPos[light * 3 + 0] - collision.Pos.x) / direction.x; //calculate t where the shadowray hits the light source. Because we don't want to count collisions that are behind the light source.
					Color collisioncolor = Color(collision.R, collision.G, collision.B);
					Color shadowRayEnergy = collisioncolor * energy * (1 - specularity) * scene->lightColor[light] * (max(0.0f, dot(collision.N, direction)) * INV4PI / ((lightPosition - collision.Pos).sqrLentgh()));
					addShadowRayToQueue(origin, direction, shadowRayEnergy.R, shadowRayEnergy.G, shadowRayEnergy.B, maxt, pixelx, pixely, shadowRays);
				}
			}

			// specular aspect
			if (specularity > 0)
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
			if (inobj) n1 = refind, n2 = 1.0f;
			else				n1 = refind, n2 = collision.refraction;
			float transition = n1 / n2;
			float costheta = dot(collision.N, -direction);
			float k = 1 - (transition * transition) * (1.0f - (costheta * costheta));

			float Fr;
			if (k < 0)
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
				Fr = R0 + (1.0f - R0) * powf(1.0f - costheta, 5.0f);
			}

			// Fresnel reflection (Schlick's approximation)
			Color reflection, refraction;
			if (Fr > 0.0f)
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
			if (Fr < 1.0f)
			{
				vec3 newdirection = transition * direction + collision.N * (transition * costheta - sqrt(k));

				addRayToQueue(
					collision.Pos + 0.00001f * newdirection,
					newdirection,
					!inobj,
					(inobj ? 1.0f : collision.refraction),
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
		addToIntermediate(pixelx, pixely, (scene->skybox->ColorAt(direction) << 8) * energy);
		//addToIntermediate(pixelx, pixely, (Color(255, 0, 0) << 8) * energy);
	}
}

void Game::TraceShadowRay(float* shadowrays, int rayIndex)
{
	int baseIndex = rayIndex * SR_SIZE;
	float maxt = shadowrays[baseIndex + SR_MAXT];

	bool collided = false;

	if (use_bvh)
	{
		float shadowray[R_SIZE] = { shadowrays[baseIndex + SR_OX], shadowrays[baseIndex + SR_OY], shadowrays[baseIndex + SR_OZ], shadowrays[baseIndex + SR_DX], shadowrays[baseIndex + SR_DY], shadowrays[baseIndex + SR_DZ] };
		//Collision shadowcollision;
		//shadowcollision.t = -1;
		
		int* stack = new int[32];
		float* AABBEntryPoints = new float[32];

		stack[0] = 1; //count, next one to evaluate;
		stack[1] = 0; //root node

		while (stack[0] > 0)
		{
			int next = stack[0]--;
			Collision newcollision = TraverseBVHNode(shadowray, bvh->pool, bvh->orderedIndices, bvh->scene, stack[next], stack, AABBEntryPoints);
			if (newcollision.t > 0 && newcollision.t < maxt)
			{
				collided = true;
				break;
			}
			
		}
		delete stack;
		delete AABBEntryPoints;
		//printf("done.");

		//Collision shadowcollision = TraverseBVHNode(shadowray, bvh->pool, bvh->orderedIndices, bvh->scene, 0);

		//if (shadowcollision.t < maxt && shadowcollision.t != -1) collided = true;
	}
	else {
		for (int i = 0; i < scene->numGeometries; i++)
		{
			vec3 origin = { shadowrays[baseIndex + SR_OX], shadowrays[baseIndex + SR_OY], shadowrays[baseIndex + SR_OZ] };
			vec3 direction = { shadowrays[baseIndex + SR_DX], shadowrays[baseIndex + SR_DY], shadowrays[baseIndex + SR_DZ] };

			//Check if position is reachable by lightsource
			//Collision scattercollision = geometry[i]->Intersect(shadowray, true);
			Collision shadowcollision = intersectTriangle(i, origin, direction, scene->triangles, true);
			if (shadowcollision.t != -1 && shadowcollision.t < maxt)
			{
				//Collision, so this ray does not reach the light source
				collided = true;
				break;
			}
		}

	}


	if (collided)
	{
		return;
	}

	Color toadd;
	toadd.R = shadowrays[baseIndex + SR_R];
	toadd.G = shadowrays[baseIndex + SR_G];
	toadd.B = shadowrays[baseIndex + SR_B];

	addToIntermediate(shadowrays[baseIndex + SR_PIXX], shadowrays[baseIndex + SR_PIXY], toadd);
}

//We don't need this function anymore, but I kept it here in case we want the other lighttypes back sometime
/*
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
				float shadowray[R_SIZE] = { origin.x, origin.y, origin.z, L.x, L.y, L.z};
				//Collision shadowcollision = bvh.Traverse(&shadowray, bvh.root);
				Collision shadowcollision = bvh->Traverse(shadowray, bvh->root);
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
}*/

vec3 Game::reflect(vec3 D, vec3 N)
{
	return D - 2 * (dot(D, N)) * N;
}

// Adds new ray to the queue of rays to be traced
void Game::addRayToQueue(Ray ray)
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
void Game::addRayToQueue(vec3 ori, vec3 dir, bool inObj, float refrInd, int bvhTr, int depth, int x, int y, float energy, float* queue)
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
	if (currentCount + 1 > queuesize / R_SIZE)
	{
		printf("ERROR: Queue overflow. Rays exceeded the %d indices of queue space.\n", rayQueueSize / R_SIZE);
	}


	// adding ray to array
	int index = (currentCount + 1) * R_SIZE; //Keep the first entry in the queue free, to save some metadata there (queuesize, currentCount)
	queue[index + R_OX] = (float)ori.x;
	queue[index + R_OY] = (float)ori.y;
	queue[index + R_OZ] = (float)ori.z;
	queue[index + R_DX] = (float)dir.x;
	queue[index + R_DY] = (float)dir.y;
	queue[index + R_DZ] = (float)dir.z;
	queue[index + R_INOBJ] = (float)inObj;
	queue[index + R_REFRIND] = refrInd;
	queue[index + R_BVHTRA] = (float)bvhTr;
	queue[index + R_DEPTH] = depth;
	queue[index + R_PIXX] = x;
	queue[index + R_PIXY] = y;
	queue[index + R_ENERGY] = energy;

	((int*)queue)[1]++; //Current count++
	raysPerFrame++;
	//no_rays++;
}

void Game::addShadowRayToQueue(vec3 ori, vec3 dir, float R, float G, float B, float maxt, float pixelX, float pixelY, float* queue)
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
	if (currentCount + 1 > queuesize / SR_SIZE)
	{
		printf("ERROR: Queue overflow. Rays exceeded the %d indices of queue space.\n", rayQueueSize / R_SIZE);
	}


	// adding ray to array
	int index = (currentCount + 1) * SR_SIZE; //Keep the first entry in the queue free, to save some metadata there (queuesize, currentCount)
	queue[index + SR_OX] = (float)ori.x;
	queue[index + SR_OY] = (float)ori.y;
	queue[index + SR_OZ] = (float)ori.z;
	queue[index + SR_DX] = (float)dir.x;
	queue[index + SR_DY] = (float)dir.y;
	queue[index + SR_DZ] = (float)dir.z;
	queue[index + SR_R] = R;
	queue[index + SR_G] = G;
	queue[index + SR_B] = B;
	queue[index + SR_MAXT] = maxt;
	queue[index + SR_PIXX] = pixelX;
	queue[index + SR_PIXY] = pixelY;

	((int*)queue)[1]++; //Current count++
	raysPerFrame++;
	//no_rays++;
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

			collision.R = triangles[baseindex + T_COLORR];
			collision.G = triangles[baseindex + T_COLORG];
			collision.B = triangles[baseindex + T_COLORB];
			//collision.other = triangles + baseindex;
			collision.refraction = triangles[baseindex + T_REFRACTION];
			collision.specularity = triangles[baseindex + T_SPECULARITY];
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

