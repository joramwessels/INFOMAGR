
// GPU structs
__device__ struct g_Color
{
	//Note: To support intermediate results > 255, store colors as 3 uints. This is probably terrible performance wise.

	unsigned int R = 0;
	unsigned int G = 0;
	unsigned int B = 0;

	__device__ g_Color(unsigned int R = 0, unsigned int G = 0, unsigned int B = 0)
	{
		this->B = B;
		this->G = G;
		this->R = R;
	}

	__device__ unsigned int to_uint_safe()
	{
		unsigned int Rtemp = R, Gtemp = G, Btemp = B;
		if (R > 255) Rtemp = 255; //Set the last bits all to one.
		if (G > 255) Gtemp = 255; //Set the last bits all to one.
		if (B > 255) Btemp = 255; //Set the last bits all to one.
		return ((Rtemp & 255) << 16) | ((Gtemp & 255) << 8) | (Btemp & 255);
	}


	__device__ g_Color operator+(const g_Color &operand) const { return g_Color(R + operand.R, G + operand.G, B + operand.B); }
	__device__ void operator+=(const g_Color &operand) { R += operand.R; G += operand.G; B += operand.B; }
	__device__ g_Color operator*(const float &operand) const { return g_Color((float)R * operand, (float)G * operand, (float)B * operand); }
	__device__ g_Color operator*(const g_Color &operand) const { return g_Color(R * operand.R, G * operand.G, B * operand.B); }

	__device__ g_Color operator>>(const int &operand) const { return g_Color(R >> operand, G >> operand, B >> operand); }
	__device__ g_Color operator<<(const int &operand) const { return g_Color(R << operand, G << operand, B << operand); }

	__device__ void from_uint(unsigned int color)
	{
		B = color & 255;
		G = (color >> 8) & 255;
		R = (color >> 16) & 255;
	}

	__device__ void operator=(const unsigned int &operand) { from_uint(operand); }


};

__device__ struct g_Collision
{
	float3 N;
	float3 Pos;
	float t;
	float R;
	float G;
	float B;
	float refraction;
	float specularity;
};


// GPU functions & kernels
__device__ void g_addToIntermediate(g_Color* buffer, float x, float y, g_Color color);

__global__ void testkernel(float* a);

__device__ float* generateRayTroughVirtualScreen(float pixelx, float pixely, bool DoF, float3 position, float3 virtualScreenCornerTL, float3 virtualScreenCornerTR, float3 virtualScreenCornerBL);

__global__ void GeneratePrimaryRay(float* rayQueue, bool DoF, float3 position, float3 virtualScreenCornerTL, float3 virtualScreenCornerTR, float3 virtualScreenCornerBL, bool SSAA);

__global__ void g_findCollisions(float* triangles, int numtriangles, float* rayQueue, void* collisions, bool useBVH, float* BVH, unsigned int* orderedIndices);

__global__ void g_traceShadowRays(float* shadowrays, float* scene, g_Color* intermediate, float* BVH, unsigned int* orderedIndices, int numGeometries, bool use_bvh);

__global__ void g_Tracerays(float* rayQueue, void* collisions, float* newRays, float* shadowRays, bool bvhdebug, g_Color* intermediate, int numLights, float* lightPos, g_Color* lightColor, unsigned int* skybox, int skyboxWidth, int skyboxHeight, int skyboxPitch);

__global__ void copyIntermediateToScreen(unsigned int* screen, g_Color* intermediate, int pitch);