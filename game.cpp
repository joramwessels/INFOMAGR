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
	use_GPU = true;

	mytimer.reset();

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	num_multiprocessors = prop.multiProcessorCount;

	((int*)rayQueue)[0] = rayQueueSize; //queue size, there can be at most just as many rays as we have pixels
	((int*)rayQueue)[1] = 0; //Num rays currently in this queue
	((int*)rayQueue)[2] = 0; //Counter used by generateprimaryray
	((int*)rayQueue)[3] = 0; //counter used by findcollisions
	((int*)rayQueue)[4] = 0; //counter used by traceray
	((int*)newRays)[0] = rayQueueSize; //queue size
	((int*)newRays)[1] = 0; //current count
	((int*)shadowRays)[0] = rayQueueSize; //queue size, can be more than the number of pixels (for instance, half reflecting objects)
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

	//Random positions for the SSAA
	random[0] = RandomFloat();
	random[1] = RandomFloat();
	random[2] = RandomFloat();
	random[3] = RandomFloat();
	random[4] = RandomFloat();
	random[5] = RandomFloat();
	random[6] = RandomFloat();
	random[7] = RandomFloat();
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
		float3 camPos = make_float3(camera.position.x, camera.position.y, camera.position.z);
		float3 TL = make_float3(camera.virtualScreenCornerTL.x, camera.virtualScreenCornerTL.y, camera.virtualScreenCornerTL.z);
		float3 TR = make_float3(camera.virtualScreenCornerTR.x, camera.virtualScreenCornerTR.y, camera.virtualScreenCornerTR.z);
		float3 BL = make_float3(camera.virtualScreenCornerBL.x, camera.virtualScreenCornerBL.y, camera.virtualScreenCornerBL.z);
		cudaMemset(g_rayQueue + 1, 0, sizeof(uint) * 4);

		GeneratePrimaryRay <<<num_multiprocessors, num_gpu_threads>>> (g_rayQueue, camera.DoF, camPos, TL, TR, BL, SSAA);
		CheckCudaError(1);

		bool finished = false;
		while (!finished)
		{
			int numRays = ((int*)rayQueue)[1];

			//Find collisions. Put in array 'collisions'
			cudaMemset(g_rayQueue + 2, 0, sizeof(uint) * 3);
			g_findCollisions <<<num_multiprocessors, num_gpu_threads>>> (scene->g_triangles, scene->numGeometries, g_rayQueue, g_collisions, use_bvh, g_BVH, g_orderedIndices);
			CheckCudaError(10);

			
			//Set the ray counters for the new rays and shadowrays to 0
			cudaMemset(g_shadowRays + 1, 0, sizeof(uint) * 2);
			cudaMemset(g_newRays + 1, 0, sizeof(uint));

			g_Tracerays <<<num_multiprocessors, num_gpu_threads >>> (g_rayQueue, g_collisions, g_newRays, g_shadowRays, bvhdebug, g_intermediate, scene->numLights, scene->g_lightPos, scene->g_lightColor, scene->g_skybox, scene->skybox->texture->GetWidth(), scene->skybox->texture->GetHeight(), scene->skybox->texture->GetPitch());
			CheckCudaError(15);

			cudaMemcpyAsync(rayQueue, g_rayQueue, sizeof(uint) * 2, cudaMemcpyDeviceToHost);
			g_traceShadowRays <<<num_multiprocessors, num_gpu_threads>>> (g_shadowRays, scene->g_triangles, g_intermediate, g_BVH, g_orderedIndices, scene->numGeometries, use_bvh);
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
	else
	// ----------------------------------------------
	// Using the CPU
	// ----------------------------------------------
	{
		rayQueue[1] = 0;
		rayQueue[2] = 0;
		rayQueue[3] = 0;
		rayQueue[4] = 0;

		//cudaMemcpy(g_rayQueue, rayQueue, rayQueueSize * sizeof(float), cudaMemcpyHostToDevice);
		vec3 TL = vec3(camera.virtualScreenCornerTL.x, camera.virtualScreenCornerTL.y, camera.virtualScreenCornerTL.z);
		vec3 TR = vec3(camera.virtualScreenCornerTR.x, camera.virtualScreenCornerTR.y, camera.virtualScreenCornerTR.z);
		vec3 BL = vec3(camera.virtualScreenCornerBL.x, camera.virtualScreenCornerBL.y, camera.virtualScreenCornerBL.z);

		GeneratePrimaryRays(rayQueue, DoF, camera.position, TL, TR, BL, SSAA);

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
		raysPerSecond = no_rays;
		avgFrameTime = mytimer.elapsed() / (float)frames;

		frames = 0;
		no_rays = 0;
		mytimer.reset();
	}
	float raysPerPixel = raysPerFrame / (SCRWIDTH * SCRHEIGHT);

	// Printing on-screen display
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

