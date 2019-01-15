#include "precomp.h"

__device__ float3 operator+(const float3 &a, const float3 &b) {

	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

}
__device__ float3 operator-(const float3 &a, const float3 &b) {

	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);

}
__device__ float3 operator*(const float3 &a, const float &b) {

	return make_float3(a.x * b, a.y * b, a.z * b);

}

__device__ uint g_seed = 0x12345678;
__device__ inline uint g_RandomUInt() { g_seed ^= g_seed << 13; g_seed ^= g_seed >> 17; g_seed ^= g_seed << 5; return g_seed; }
__device__ inline float g_RandomFloat() { return g_RandomUInt() * 2.3283064365387e-10f; }
 

__device__ float3 normalize(float3 in) {
	float mag = 1 / sqrtf(in.x * in.x + in.y * in.y + in.z*in.z);
	return make_float3(in.x * mag, in.y * mag, in.z * mag);
}

__global__ void testkernel(float* a)
{
	int i = threadIdx.x;
	if (i > 99) return;
	a[i] = 2 * a[i];
	printf("Gpu thread %i says hi! \n", i);
}

__device__ float* generateRayTroughVirtualScreen(float pixelx, float pixely, bool DoF, float3 position, float3 virtualScreenCornerTL, float3 virtualScreenCornerTR, float3 virtualScreenCornerBL)
{
	float3 pixelPosScaled;
	pixelPosScaled.x = pixelx / SCRWIDTH; //Scale the pixel position to be in the range 0..1
	pixelPosScaled.y = pixely / SCRHEIGHT;

	float3 DofRandomness = { 0, 0, 0 };
	if (DoF) DofRandomness = make_float3((g_RandomFloat() * 0.1 - 0.05), (g_RandomFloat() * 0.1 - 0.05), 0); //TODO: make random and maybe 7-gon instead of square?

	float3 origin = position + DofRandomness;
	//printf("ray origin: %f, %f, %f", origin.x, origin.y, origin.z);

	float3 positionOnVirtualScreen = virtualScreenCornerTL + (virtualScreenCornerTR - virtualScreenCornerTL) * pixelPosScaled.x + (virtualScreenCornerBL - virtualScreenCornerTL) * pixelPosScaled.y;
	float3 direction = normalize(positionOnVirtualScreen - origin);

	float* ray = new float[6];
	ray[0] = origin.x;
	ray[1] = origin.y;
	ray[2] = origin.z;
	ray[3] = direction.x;
	ray[4] = direction.y;
	ray[5] = direction.z;
	//float ray[6] = { origin.x, origin.y, origin.z, direction.x, direction.y, direction.z };

	return ray;
}

__global__ void GeneratePrimaryRay(float* rayQueue, bool DoF, float3 position, float3 virtualScreenCornerTL, float3 virtualScreenCornerTR, float3 virtualScreenCornerBL, bool SSAA)
{
	int pixelx = threadIdx.x;
	int pixely = threadIdx.y;

	if (pixelx > SCRWIDTH || pixely > SCRHEIGHT) return;

	//Generate the ray
	float* ray = generateRayTroughVirtualScreen(pixelx, pixely, DoF, position, virtualScreenCornerTL, virtualScreenCornerTR, virtualScreenCornerBL);

	int baseIndex = ((pixelx + SCRWIDTH * pixely) + 1) * R_SIZE;

	rayQueue[baseIndex + R_OX] = ray[0];
	rayQueue[baseIndex + R_OY] = ray[1];
	rayQueue[baseIndex + R_OZ] = ray[2];
	rayQueue[baseIndex + R_DX] = ray[3];
	rayQueue[baseIndex + R_DY] = ray[4];
	rayQueue[baseIndex + R_DZ] = ray[5];
	rayQueue[baseIndex + R_INOBJ] = 0;
	rayQueue[baseIndex + R_REFRIND] = 1.0f;
	rayQueue[baseIndex + R_BVHTRA] = 0;
	rayQueue[baseIndex + R_DEPTH] = 0;
	rayQueue[baseIndex + R_PIXX] = pixelx;
	rayQueue[baseIndex + R_PIXY] = pixely;
	rayQueue[baseIndex + R_ENERGY] = 1.0f;

	atomicInc(((uint*)rayQueue), 0xffffffff);


	delete ray;

}