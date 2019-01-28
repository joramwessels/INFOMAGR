
// GPU structs
__device__ struct g_Collision
{
	float4 Nt; //t saved in position 4
	float4 PosRefr; //Refraction index saved in position 4
	float4 ColorSpec; //Specularity saved in position 4
};


// GPU functions & kernels
__device__ void g_addToIntermediate(float4* buffer, float x, float y, float4 color);

__global__ void testkernel(float* a);

__device__ float* generateRayTroughVirtualScreen(float pixelx, float pixely, bool DoF, float3 position, float3 virtualScreenCornerTL, float3 virtualScreenCornerTR, float3 virtualScreenCornerBL, float* DoF_random);

__global__ void GeneratePrimaryRay(float* rayQueue, bool DoF, float3 position, float3 virtualScreenCornerTL, float3 virtualScreenCornerTR, float3 virtualScreenCornerBL, bool SSAA, int SSAA_val, float* DoF_random, float* g_SSAA_random);

__global__ void g_findCollisions(float* triangles, int numtriangles, float* rayQueue, void* collisions, bool useBVH, float* BVH, unsigned int* orderedIndices);

__global__ void g_traceShadowRays(float* shadowrays, float* scene, float4* intermediate, float* BVH, unsigned int* orderedIndices, int numGeometries, bool use_bvh);

__global__ void g_Tracerays(float* rayQueue, void* collisions, float* newRays, float* shadowRays, bool bvhdebug, float4* intermediate, int numLights, float* lightPos, float3* lightColor, unsigned int* skybox, int skyboxWidth, int skyboxHeight, int skyboxPitch);

__global__ void copyIntermediateToScreen(unsigned int* screen, float4* intermediate, int pitch);

__global__ void setup_RNG(void* curandstate, int seed);

__global__ void precalculate_RNG(float* g_DoF_random, void* curandstate, int SSAA_val);