#include "precomp.h" // include (only) this in every .cpp file

float frame = 0;


// -----------------------------------------------------------
// Initialize the application
// -----------------------------------------------------------
void Game::Init()
{
	/*
	AVAILABLE SCENES:
		SCENE_OBJ_GLASS
		SCENE_OBJ_HALFREFLECT
		SCENE_TERRA_COTTA
		SCENE_MODERN_ART
		SCENE_FLOORONLY
		SCENE_CUBE
	*/
	scene = new SceneManager();
	scene->loadScene(SceneManager::SCENE_OBJ_HALFREFLECT, &camera);
	shadowRayQueueSize = (primaryRayCount * scene->numLights * SR_SIZE);
	shadowRays = new float[shadowRayQueueSize];

	// Settings
	SSAA = false;
	camera.DoF = false;
	use_bvh = true;
	bvhdebug = false;
	use_GPU = true;

	if (use_bvh) generateBVH();

	// Collecting GPU core count
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	num_multiprocessors = prop.multiProcessorCount;

	// Initializing the sizes of all ray queues
	((int*)rayQueue)[0] = rayQueueSize; //queue size, there can be at most just as many rays as we have pixels
	((int*)rayQueue)[1] = 0; //Num rays currently in this queue
	((int*)rayQueue)[2] = 0; //Counter used by generateprimaryray
	((int*)rayQueue)[3] = 0; //counter used by findcollisions
	((int*)rayQueue)[4] = 0; //counter used by traceray
	((int*)newRays)[0] = rayQueueSize; //queue size
	((int*)newRays)[1] = 0; //current count
	((int*)shadowRays)[0] = shadowRayQueueSize; //queue size, can be more than the number of pixels (for instance, half reflecting objects)
	((int*)shadowRays)[1] = 0; //current count

	SSAA_random = (float*)malloc(SSAA_random_size * sizeof(float));
	for (int i = 0; i < SSAA_random_size; i++) SSAA_random[i] = RandomFloat();

	// Moving everything to the GPU
	if (use_GPU)
	{
		cudaMalloc(&g_rayQueue, rayQueueSize * sizeof(float));
		cudaMalloc(&g_newRays, rayQueueSize * sizeof(float));
		cudaMalloc(&g_collisions, primaryRayCount * sizeof(Collision));
		cudaMalloc(&g_shadowRays, shadowRayQueueSize * sizeof(float));
		cudaMalloc(&g_intermediate, SCRWIDTH * SCRHEIGHT * sizeof(float4));
		cudaMalloc(&g_screen, SCRWIDTH * SCRHEIGHT * sizeof(uint));
		cudaMalloc(&g_DoF_random, SSAA_random_size * sizeof(float));
		cudaMalloc(&curandstate, SCRWIDTH * SCRHEIGHT * sizeof(curandState));
		cudaMalloc(&g_SSAA_random, SSAA_random_size * sizeof(float));

		cudaMemcpy(g_newRays, newRays, sizeof(float) * 2, cudaMemcpyHostToDevice);
		cudaMemcpy(g_shadowRays, shadowRays, sizeof(float) * 2, cudaMemcpyHostToDevice);
		cudaMemcpy(g_rayQueue, rayQueue, sizeof(float) * 5, cudaMemcpyHostToDevice);
		cudaMemcpy(g_SSAA_random, SSAA_random, SSAA_random_size * sizeof(float), cudaMemcpyHostToDevice);

		scene->moveSceneToGPU();

		setup_RNG <<<SCRWIDTH, SCRHEIGHT>>> (curandstate, DoF_seed);
	}

	//Random positions for the SSAA
	//random[0] = RandomFloat();
	//random[1] = RandomFloat();
	//random[2] = RandomFloat();
	//random[3] = RandomFloat();
	//random[4] = RandomFloat();
	//random[5] = RandomFloat();
	//random[6] = RandomFloat();
	//random[7] = RandomFloat();

	mytimer.reset();
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
	if (use_GPU)
	// ----------------------------------------------
	// Using the GPU
	// ----------------------------------------------
	{
		// Generating primary rays that shoot through the virtual screen
		camera.DoF = false; // DEBUG
		if (camera.DoF)
		{
			precalculate_RNG <<<SCRWIDTH, SCRHEIGHT>>> (g_DoF_random, curandstate, SSAA_val);
			cudaDeviceSynchronize();
		}
		float3 camPos = make_float3(camera.position.x, camera.position.y, camera.position.z);
		float3 TL = make_float3(camera.virtualScreenCornerTL.x, camera.virtualScreenCornerTL.y, camera.virtualScreenCornerTL.z);
		float3 TR = make_float3(camera.virtualScreenCornerTR.x, camera.virtualScreenCornerTR.y, camera.virtualScreenCornerTR.z);
		float3 BL = make_float3(camera.virtualScreenCornerBL.x, camera.virtualScreenCornerBL.y, camera.virtualScreenCornerBL.z);
		cudaMemset(g_rayQueue + 1, 0, sizeof(uint) * 4);

		GeneratePrimaryRay <<<num_multiprocessors, num_gpu_threads>>> (g_rayQueue, camera.DoF, camPos, TL, TR, BL, SSAA, SSAA_val, g_DoF_random, g_SSAA_random);
		CheckCudaError(1);

		bool finished = false;
		while (!finished)
		{
			// Extending primary rays to find collisions
			int numRays = ((int*)rayQueue)[1];
			cudaMemset(g_rayQueue + 2, 0, sizeof(uint) * 3);
			g_findCollisions <<<num_multiprocessors, num_gpu_threads>>> (scene->g_triangles, scene->numGeometries, g_rayQueue, g_collisions, use_bvh, g_BVH, g_orderedIndices);
			CheckCudaError(2);

			// Setting the ray counters for the new rays and shadowrays to 0
			cudaMemset(g_shadowRays + 1, 0, sizeof(uint) * 2);
			cudaMemset(g_newRays + 1, 0, sizeof(uint));

			// Evaluating all found collisions. Generating shadowrays and ray extensions for reflections & refractions
			g_Tracerays <<<num_multiprocessors, num_gpu_threads >>> (g_rayQueue, g_collisions, g_newRays, g_shadowRays, bvhdebug, g_intermediate, scene->numLights, scene->g_lightPos, scene->g_lightColor, scene->g_skybox, scene->skybox->texture->GetWidth(), scene->skybox->texture->GetHeight(), scene->skybox->texture->GetPitch());
			CheckCudaError(3);

			// Extending the shadowrays towards the light sources to check for occlusion
			cudaMemcpyAsync(rayQueue, g_rayQueue, sizeof(uint) * 2, cudaMemcpyDeviceToHost);
			cudaMemcpyAsync(&raysInGPU, g_shadowRays + 1, sizeof(uint), cudaMemcpyDeviceToHost);
			g_traceShadowRays <<<num_multiprocessors, num_gpu_threads>>> (g_shadowRays, scene->g_triangles, g_intermediate, g_BVH, g_orderedIndices, scene->numGeometries, use_bvh);
			CheckCudaError(4);

			//Flip the arrays
			float* temp = g_rayQueue;
			g_rayQueue = g_newRays;
			g_newRays = temp;

			//Get the new ray count from the gpu
			no_rays += raysInGPU + numRays;
			raysPerFrame += raysInGPU + numRays;
			if (((int*)rayQueue)[1] == 0) finished = true;
		}

		// Copying GPU screen buffer to CPU
		copyIntermediateToScreen<<<SCRWIDTH, SCRHEIGHT>>>(g_screen, g_intermediate, screen->GetPitch());
		cudaMemcpy(screen->GetBuffer(), g_screen, SCRWIDTH * SCRHEIGHT * sizeof(uint), cudaMemcpyDeviceToHost);
	}
	else
	// ----------------------------------------------
	// Using the CPU
	// ----------------------------------------------
	{
		rayQueue[1] = 0;
		rayQueue[2] = 0;
		rayQueue[3] = 0;
		rayQueue[4] = 0;

		// Generating primary rays that shoot through the virtual screen
		vec3 TL = vec3(camera.virtualScreenCornerTL.x, camera.virtualScreenCornerTL.y, camera.virtualScreenCornerTL.z);
		vec3 TR = vec3(camera.virtualScreenCornerTR.x, camera.virtualScreenCornerTR.y, camera.virtualScreenCornerTR.z);
		vec3 BL = vec3(camera.virtualScreenCornerBL.x, camera.virtualScreenCornerBL.y, camera.virtualScreenCornerBL.z);
		GeneratePrimaryRays(rayQueue, camera.DoF, camera.position, TL, TR, BL, SSAA);

		bool finished = false;
		while (!finished)
		{
			// Extending primary rays to find collisions
			int numRays = ((int*)rayQueue)[1];
			findCollisions(rayQueue, numRays, collisions);

			// Evaluating all found collisions. Generating shadowrays and ray extensions for reflections & refractions
			for (int i = 1; i <= numRays; i++)
			{
				TraceRay(rayQueue, i, numRays, collisions, newRays, shadowRays);
			}

			// Extending the shadowrays towards the light sources to check for occlusion
			int numShadowRays = ((int*)shadowRays)[1];
			for (int i = 1; i <= numShadowRays; i++)
			{
				TraceShadowRay(shadowRays, i);
			}

			// Flip the arrays
			float* temp = rayQueue;
			rayQueue = newRays;
			newRays = temp;

			((int*)newRays)[1] = 0;    //set new ray count to 0
			((int*)shadowRays)[1] = 0; //set new shadowray count to 0

			if (((int*)rayQueue)[1] == 0) finished = true;

		}

		// Plotting the intermediate screen buffer to the screen
		for (size_t pixelx = 0; pixelx < SCRWIDTH; pixelx++) for (size_t pixely = 0; pixely < SCRHEIGHT; pixely++)
		{
			screen->Plot(pixelx, pixely, (intermediate[(int)pixelx + ((int)pixely * SCRWIDTH)] >> 8).to_uint_safe());
		}
		memset(intermediate, 0, SCRWIDTH * SCRHEIGHT * sizeof(Color));
	}

	// User Interface
	if (keyW) camera.move(camera.getDirection() * 0.1f);
	if (keyS) camera.move(camera.getDirection() * -0.1f);
	if (keyD) camera.move(camera.getLeft() * 0.1f);
	if (keyA) camera.move(camera.getLeft() * -0.1f);
	if (keyQ) camera.move(camera.getUp() * 0.1f);
	if (keyE) camera.move(camera.getUp() * -0.1f);
	if (keyplus) camera.setZoom(camera.zoom * 1.1);
	if (keymin) camera.setZoom(camera.zoom / 1.1);
	if (keyComma) camera.setFocalPoint(camera.focalpoint / 1.1);
	if (keyPeriod) camera.setFocalPoint(camera.focalpoint * 1.1);

	// Calculating on-screen display
	frames++;
	if (mytimer.elapsed() > 1000) {
		prevsecframes = frames;
		raysPerSecond = no_rays * 1000 / mytimer.elapsed();
		avgFrameTime = mytimer.elapsed() / (float)frames;

		frames = 0;
		no_rays = 0;
		mytimer.reset();
	}
	float raysPerPixel = ((float) raysPerFrame) / (SCRWIDTH * SCRHEIGHT);
	int numPrimitives = scene->numGeometries;

	// Printing on-screen display
	screen->Bar(0, 0, 150, 40, 0x000000);
	char buffer[64];
	sprintf(buffer, "No. primitives: %i", numPrimitives);
	screen->Print(buffer, 1, 2, 0xffffff);
	sprintf(buffer, "FPS: %i", prevsecframes);
	screen->Print(buffer, 1, 10, 0xffffff);
	sprintf(buffer, "Avg time (ms): %.0f", avgFrameTime);
	screen->Print(buffer, 1, 18, 0xffffff);
	sprintf(buffer, "Rays/pixel: %.1f", raysPerPixel);
	screen->Print(buffer, 1, 26, 0xffffff);
	sprintf(buffer, "Rays/second: %i", raysPerSecond);
	screen->Print(buffer, 1, 34, 0xffffff);
	raysPerFrame = 0;
}
