#pragma once

namespace Tmpl8 {


class Game
{
public:
	void SetTarget( Surface* surface ) { screen = surface; }
	void Init();
	void Shutdown();
	void Tick( float deltaTime );

	void MouseUp( int button ) {   /* implement if you want to detect mouse button presses */ }
	void MouseDown( int button ) { /* implement if you want to detect mouse button presses */ }
	void MouseMove(int x, int y) { camera.rotate({ (float)y, (float)x, 0 }); }
	void KeyUp( int key ) { 
		switch (key)
		{
		case 26: //W
			keyW = false;
			break;
		case 4: //A
			keyA = false;
			break;
		case 22: //S
			keyS = false;
			break;
		case 7: //D
			keyD = false;
			break;
		case 20:
			keyQ = false;
			break;
		case 8:
			keyE = false;
			break;
		case 87: //+
			keyplus = false;
			break;
		case 86: //-
			keymin = false;
			break;
		case 54: //,
			keyComma = false;
			break;
		case 55: //.
			keyPeriod = false;

		default:
			break;
		}
	}
	void KeyDown( int key ) { 
		switch (key)
		{
		case 26: //W
			keyW = true;
			break;
		case 4: //A
			keyA = true;
			break;
		case 22: //S
			keyS = true;
			break;
		case 7: //D
			keyD = true;
			break;
		case 20:
			keyQ = true;
			break;
		case 8:
			keyE = true;
			break;
		case 87: //+
			keyplus = true;
			break;
		case 86: //-
			keymin = true;
			break;
		case 54: //,
			keyComma = true;
			break;
		case 55: //.
			keyPeriod = true;
			break;
		case 56: // /
			SSAA = !SSAA;
			break;
		case 16: // M
			camera.DoF = !camera.DoF;
			break;
		case 5: // M
			bvhdebug = !bvhdebug;
			break;
		default:
			printf("Key %i pressed. \n", key);
			break;
		}
	}

private:
	Camera camera;
	SceneManager *scene;

	// SSAA
	bool SSAA;
	int SSAA_val;
	float *SSAA_random;
	float *g_SSAA_random;
	bool DoF = false;
	int SSAA_random_size;

	// Tracing
	void GeneratePrimaryRays(float* rayQueue, bool DoF, vec3 position, vec3 TL, vec3 TR, vec3 BL, bool SSAA);
	void generateRayTroughVirtualScreen(float* ray, float pixelx, float pixely, bool DoF, vec3 position, vec3 TL, vec3 TR, vec3 BL);
	void TraceRay(float* rays, int ray, int numrays, Collision* collisions, float* newRays, float* shadowRays);
	void TraceShadowRay(float* shadowrays, int rayIndex);
	vec3 reflect(vec3 D, vec3 N) { return D - 2 * (dot(D, N)) * N; }

	// Screen buffer
	Surface* screen;
	uint* g_screen;

	// User Interaction
	vec3 camerapos = { 0,0,0 };
	bool keyW = false, keyA = false, keyS = false, keyD = false, keyQ = false, keyE = false, keymin = false, keyplus = false, keyComma = false, keyPeriod = false;

	//fps counter
	int frames = 0;
	int no_rays = 0;
	int prevsecframes = 0;
	timer mytimer;
	float avgFrameTime;
	float raysPerFrame = 0;
	int raysPerSecond = 0;

	// BVH
	bool use_bvh = false;
	BVH* bvh;
	float* g_BVH;
	uint* g_orderedIndices;
	bool bvhdebug = false;
	void generateBVH() {
		//BVH GENERATION
		bvh = new BVH;
		printf("Starting BVH generation... \n");
		mytimer.reset();
		bvh->Build(scene->triangles, scene->numGeometries);
		printf("BVH Generation done. Build time: %f, Depth: %i \n", mytimer.elapsed(), bvh->depth);
		cudaMalloc(&g_BVH, bvh->poolPtr * B_SIZE * sizeof(float));
		cudaMemcpy(g_BVH, bvh->pool, bvh->poolPtr * B_SIZE * sizeof(float), cudaMemcpyHostToDevice);
		cudaMalloc(&g_orderedIndices, bvh->totalNoElements * sizeof(uint));
		cudaMemcpy(g_orderedIndices, bvh->orderedIndices, bvh->totalNoElements * sizeof(uint), cudaMemcpyHostToDevice);
	}
	
	// ---------
	// Ray queue
	// ---------
	const int primaryRayCount = SCRHEIGHT * SCRWIDTH * 4;		// The number of pixels * the number of rays per pixel
	const int rayQueueSize = (primaryRayCount + 1) * R_SIZE;	// The number of floats in a ray queue
	const int shadowRayQueueSize = (rayQueueSize * 5 + R_SIZE);
	float *rayQueue = new float[rayQueueSize]; // ray queue; rays are represented as consecutive series of 13 floats, ordered as in the Ray struct
	float* newRays = new float[rayQueueSize];
	float* shadowRays = new float[shadowRayQueueSize];
	float *g_rayQueue, *g_newRays, *g_shadowRays;
	void addRayToQueue(Ray ray);
	void addRayToQueue(float *ray, float *queue);
	void addRayToQueue(vec3 ori, vec3 dir, bool inObj, float refrInd, int bvhTr, int depth, int x, int y, float energy, float* queue);
	void addShadowRayToQueue(vec3 ori, vec3 dir, float R, float G, float B, float maxt, float pixelX, float pixelY, float* queue);

	// Collisions
	Collision* collisions = new Collision[rayQueueSize];
	void* g_collisions;
	Collision nearestCollision(float* ray_ptr);
	void findCollisions(float* rays, int numrays, Collision* collisions);

	// Intermediate screen buffer
	Color intermediate[SCRWIDTH * SCRHEIGHT];
	float4* g_intermediate;
	void addToIntermediate(float x, float y, Color color) { intermediate[(int)x + ((int)y * SCRWIDTH)] += color; };

	//Animation
	bool animate = false;

	// GPU optimization
	bool use_GPU;
	int num_multiprocessors;
	const int num_gpu_threads = 32 * 8;
};

}; // namespace Tmpl8

Collision intersectTriangle(int i, vec3 origin, vec3 direction, float* triangles, bool isShadowRay = false);