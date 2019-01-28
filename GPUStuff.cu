#include "precomp.h"

// float3 operations
__device__ float3 operator+(const float3 &a, const float3 &b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 operator-(const float3 &a, const float3 &b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 operator*(const float3 &a, const float &b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}
__device__ float3 operator*(const float3 &a, const float3 &b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__device__ float3 normalize(float3 in)
{
	float mag = 1 / sqrtf(in.x * in.x + in.y * in.y + in.z*in.z);
	return make_float3(in.x * mag, in.y * mag, in.z * mag);
}
__device__ float3 cross(float3 a, float3 b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ float dot(float3 a, float3 b)
{
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}
__device__ float4 operator*(const float4 &a, const float &b)
{
	return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
__device__ float4 operator/(const float4 &a, const float &b)
{
	return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}


// SSAA random number generation
__device__ uint g_seed = 0x12345678;
__device__ inline uint g_RandomUInt() { g_seed ^= g_seed << 13; g_seed ^= g_seed >> 17; g_seed ^= g_seed << 5; return g_seed; }
__device__ inline float g_RandomFloat() { return g_RandomUInt() * 2.3283064365387e-10f; }

__device__ float g_random1 = 0.529669f;
__device__ float g_random2 = 0.083422f;
__device__ float g_random3 = 0.281753f;
__device__ float g_random4 = 0.506648f;
__device__ float g_random5 = 0.438385f;
__device__ float g_random6 = 0.162733f;
__device__ float g_random7 = 0.538243f;
__device__ float g_random8 = 0.769904f;

__global__ void setup_RNG(void* curandstate, int seed)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(seed, idx, 0, &((curandState*)curandstate)[idx]);
}

__global__ void precalculate_RNG(float* g_DoF_random, void* curandstate, int SSAA_val)
{
	// TODO make independent of x, y
	int ranPerPixel = 2 * SSAA_val;
	int x = threadIdx.x * ranPerPixel;
	int y = blockIdx.x * ranPerPixel;

	for (int i = 0; i < ranPerPixel; i++)
	{
		g_DoF_random[x + y * SCRWIDTH + i] = curand_uniform((curandState*)curandstate);
	}
}

__global__ void testkernel(float* a)
{
	int i = threadIdx.x;
	if (i > 99) return;
	a[i] = 2 * a[i];
	printf("Gpu thread %i says hi! \n", i);
}

// Adds the given color to the intermediate screen buffer
__device__ void g_addToIntermediate(float4* buffer, float x, float y, float4 color)
{
	//buffer[(int)x + ((int)y * SCRWIDTH)] += color;
	atomicAdd(&buffer[(int)x + ((int)y * SCRWIDTH)].x, color.x);
	atomicAdd(&buffer[(int)x + ((int)y * SCRWIDTH)].y, color.y);
	atomicAdd(&buffer[(int)x + ((int)y * SCRWIDTH)].z, color.z);

};

// Adds a given ray to the given ray queue and updates the queue size
__device__ void addRayToQueue(float* ray, float* queue)
{
	int id = atomicInc(((uint*)queue) + 1, 0xffffffff) + 1;
	int queuesize = ((uint*)queue)[0];

	if (id > queuesize / R_SIZE)
	{
		printf("ERROR: Queue overflow. Rays exceeded the %i indices of ray queue space.\n", (int)(queuesize / R_SIZE));
	}

	int baseIndex = id * R_SIZE;

	queue[baseIndex + R_OX] = ray[0];
	queue[baseIndex + R_OY] = ray[1];
	queue[baseIndex + R_OZ] = ray[2];
	queue[baseIndex + R_DX] = ray[3];
	queue[baseIndex + R_DY] = ray[4];
	queue[baseIndex + R_DZ] = ray[5];
	queue[baseIndex + R_INOBJ] = ray[6];
	queue[baseIndex + R_REFRIND] = ray[7];
	queue[baseIndex + R_BVHTRA] = ray[8];
	queue[baseIndex + R_DEPTH] = ray[9];
	queue[baseIndex + R_PIXX] = ray[10];
	queue[baseIndex + R_PIXY] = ray[11];
	queue[baseIndex + R_ENERGY] = ray[12];
}

// Generates and returns a primary ray given the virtual screen coordinates
__device__ float* generateRayTroughVirtualScreen(float pixelx, float pixely, bool DoF, float3 position, float3 virtualScreenCornerTL, float3 virtualScreenCornerTR, float3 virtualScreenCornerBL, float* DoF_random)
{
	float3 pixelPosScaled;
	pixelPosScaled.x = pixelx / SCRWIDTH; //Scale the pixel position to be in the range 0..1
	pixelPosScaled.y = pixely / SCRHEIGHT;

	float3 DofRandomness = { 0, 0, 0 };
	if (DoF)
	{
		DofRandomness = make_float3((DoF_random[0] * 0.1 - 0.05), (DoF_random[1] * 0.1 - 0.05), 0); //TODO: make random and maybe 7-gon instead of square?
	}

	float3 origin = position + DofRandomness;
	//printf("ray origin: %f, %f, %f", origin.x, origin.y, origin.z);

	float3 positionOnVirtualScreen = virtualScreenCornerTL + (virtualScreenCornerTR - virtualScreenCornerTL) * pixelPosScaled.x + (virtualScreenCornerBL - virtualScreenCornerTL) * pixelPosScaled.y;
	float3 direction = normalize(positionOnVirtualScreen - origin);

	float* ray = new float[R_SIZE];
	ray[0] = origin.x;
	ray[1] = origin.y;
	ray[2] = origin.z;
	ray[3] = direction.x;
	ray[4] = direction.y;
	ray[5] = direction.z;
	//float ray[6] = { origin.x, origin.y, origin.z, direction.x, direction.y, direction.z };

	return ray;
}

// Generates and collects primary rays in the given ray queue
__global__ void GeneratePrimaryRay(float* rayQueue, bool DoF, float3 position, float3 virtualScreenCornerTL, float3 virtualScreenCornerTR, float3 virtualScreenCornerBL, bool SSAA, int SSAA_val, float* DoF_random)
{
	uint numRays = SCRWIDTH * SCRHEIGHT;
	uint raynum = atomicInc(((uint*)rayQueue) + 2, 0xffffffff);

	while (raynum < numRays) {
		int pixelx = raynum % SCRWIDTH;
		int pixely = raynum / SCRWIDTH;
		float* DoF_ptr;
		//int pixelx = threadIdx.x;
		//int pixely = blockIdx.x;

		//printf("id: %i, x: %i, y: %i \n", raynum, pixelx, pixely);

		if (pixelx > SCRWIDTH || pixely > SCRHEIGHT) printf("wtf");

		//Generate the ray
		if (SSAA) for(int i=0; i < SSAA_val; i++)
		{
			if (DoF) DoF_ptr = DoF_random + ((raynum * SSAA_val) + i) * 2;
			float* ray = generateRayTroughVirtualScreen(pixelx, pixely, DoF, position, virtualScreenCornerTL, virtualScreenCornerTR, virtualScreenCornerBL, DoF_ptr);

			ray[R_INOBJ] = 0;
			ray[R_REFRIND] = 1.0f;
			ray[R_BVHTRA] = 0;
			ray[R_DEPTH] = 0;
			ray[R_PIXX] = pixelx;
			ray[R_PIXY] = pixely;
			ray[R_ENERGY] = 1.0f / SSAA_val;

			addRayToQueue(ray, rayQueue);
			delete ray;
		}
		else {
			if (DoF) DoF_ptr = DoF_random + raynum * 2;
			float* ray = generateRayTroughVirtualScreen(pixelx, pixely, DoF, position, virtualScreenCornerTL, virtualScreenCornerTR, virtualScreenCornerBL, DoF_ptr);

			ray[R_INOBJ] = 0;
			ray[R_REFRIND] = 1.0f;
			ray[R_BVHTRA] = 0;
			ray[R_DEPTH] = 0;
			ray[R_PIXX] = pixelx;
			ray[R_PIXY] = pixely;
			ray[R_ENERGY] = 1.0f;

			addRayToQueue(ray, rayQueue);
			delete ray;
		}
		raynum = atomicInc(((uint*)rayQueue) + 2, 0xffffffff);
	}
}

// Finds a collision with the triangle. Returns a collision with t = -1 if none were found.
__device__ g_Collision g_intersectTriangle(int i, float* ray_ptr, float *triangles, bool isShadowRay = false)
{
	int baseindex = i * FLOATS_PER_TRIANGLE;

	float3 v0 = {
		triangles[baseindex + T_V0X],
		triangles[baseindex + T_V0Y],
		triangles[baseindex + T_V0Z] };
	float3 v1 = {
		triangles[baseindex + T_V1X],
		triangles[baseindex + T_V1Y],
		triangles[baseindex + T_V1Z] };
	float3 v2 = {
		triangles[baseindex + T_V2X],
		triangles[baseindex + T_V2Y],
		triangles[baseindex + T_V2Z] };
	float3 e0 = {
		triangles[baseindex + T_E0X],
		triangles[baseindex + T_E0Y],
		triangles[baseindex + T_E0Z] };
	float3 e1 = {
		triangles[baseindex + T_E1X],
		triangles[baseindex + T_E1Y],
		triangles[baseindex + T_E1Z] };
	float3 e2 = {
		triangles[baseindex + T_E2X],
		triangles[baseindex + T_E2Y],
		triangles[baseindex + T_E2Z] };
	float3 N = { triangles[baseindex + T_NX],
		triangles[baseindex + T_NY],
		triangles[baseindex + T_NZ] };

	float3 direction = { ray_ptr[R_DX], ray_ptr[R_DY], ray_ptr[R_DZ] };
	float3 origin = { ray_ptr[R_OX], ray_ptr[R_OY], ray_ptr[R_OZ] };

	float D = triangles[baseindex + T_D];
	

	g_Collision collision;
	collision.Nt.w = -1;

	float NdotR = dot(direction, N);
	if (NdotR == 0) return collision; //Ray parrallel to plane, would cause division by 0

	float t = -(dot(origin, N) + D) / (NdotR);

	//From https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
	if (t > 0)
	{
		float3 P = origin + direction * t;
		if (dot(N, cross(e0, (P - v0))) > 0 && dot(N, cross(e1, (P - v1))) > 0 && dot(N, cross(e2, (P - v2))) > 0)
		{
			//Collision
			collision.Nt.w = t;

			if (isShadowRay) {
				return collision;
			}

			collision.ColorSpec.x = triangles[baseindex + T_COLORR];
			collision.ColorSpec.y = triangles[baseindex + T_COLORG];
			collision.ColorSpec.z = triangles[baseindex + T_COLORB];
			collision.ColorSpec.w = triangles[baseindex + T_SPECULARITY];
			//collision.other = triangles + baseindex;
			collision.PosRefr.w = triangles[baseindex + T_REFRACTION];
			if (NdotR > 0) N = N * -1;

			collision.Nt.x = N.x;
			collision.Nt.y = N.y;
			collision.Nt.z = N.z;

			collision.PosRefr.x = P.x;
			collision.PosRefr.y = P.y;
			collision.PosRefr.z = P.z;
			return collision;
		}
	}
	return collision;
}

// Checks if the ray intersects the BVH node. Returns 'tmin' if it does, and returns -99999 otherwise.
__device__ float g_IntersectAABB(float* ray_ptr, float* BVHNode)
{
	float xmin = BVHNode[B_AABB_MINX];
	float xmax = BVHNode[B_AABB_MAXX];
	float ymin = BVHNode[B_AABB_MINY];
	float ymax = BVHNode[B_AABB_MAXY];
	float zmin = BVHNode[B_AABB_MINZ];
	float zmax = BVHNode[B_AABB_MAXZ];

	float dirX = ray_ptr[R_DX];
	float dirY = ray_ptr[R_DY];
	float dirZ = ray_ptr[R_DZ];
	float OX = ray_ptr[R_OX];
	float OY = ray_ptr[R_OY];
	float OZ = ray_ptr[R_OZ];

	float invDirX = 1 / dirX;
	float tmin = (xmin - OX) * invDirX;
	float tmax = (xmax - OX) * invDirX;

	if (tmin > tmax) { 
		float temp = tmin;
		tmin = tmax;
		tmax = temp;
	}

	float invDirY = 1 / dirY;
	float tymin = (ymin - OY) * invDirY;
	float tymax = (ymax - OY) * invDirY;

	if (tymin > tymax) {
		float temp = tymin;
		tymin = tymax;
		tymax = temp;

		//swap(tymin, tymax);
	}

	if ((tmin > tymax) || (tymin > tmax))
		return -99999;

	tmin = max(tmin, tymin);
	tmax = min(tymax, tmax);

	float invDirZ = 1 / dirZ;
	float tzmin = (zmin - OZ) * invDirZ;
	float tzmax = (zmax - OZ) * invDirZ;

	if (tzmin > tzmax) {
		float temp = tzmin;
		tzmin = tzmax;
		tzmax = temp;

		//swap(tzmin, tzmax);
	}

	if ((tmin > tzmax) || (tzmin > tmax))
		return -99999;

	tmin = max(tmin, tzmin);
	tmax = min(tzmax, tmax);

	if (tmax < 0) return -99999;

	return tmin;
}

// Recursively traverses the BVH tree from the given node to find a collision. Returns a collision with t = -1 if none were found.
__device__ g_Collision g_TraverseBVHNode(float* ray_ptr, float* pool, uint* orderedIndices, float* scene, int index, int* stack, float* stackAABBEntrypoints)
{
	g_Collision closest;
	closest.Nt.w = -1;

	ray_ptr[R_BVHTRA]++;
	int count = pool[index + B_COUNT];
		// If leaf
	if (count != 0)
	{
		float closestdist = 0xffffff;


		// Find closest collision
		for (int i = 0; i < pool[index + B_COUNT]; i++)
		{
			//Collision collision = scene[orderedIndices[node->leftFirst + i]]->Intersect(*ray);
			int triangleindex = orderedIndices[(int)pool[index + B_LEFTFIRST] + i];

			g_Collision collision = g_intersectTriangle(triangleindex, ray_ptr, scene);
			float dist = collision.Nt.w;
			if (dist != -1 && dist < closestdist)
			{
				//Collision. Check if closest
				closest = collision;
				closestdist = dist;
			}
		}
		//printf("leaf: collision at %f \n", closest.t);
		return closest;
	}
	// If node
	else
	{
		//This is not a leaf
		// Check both children and return the closest collision if both intersected
		int leftchild = pool[(int)index + B_LEFTFIRST];
		int rightchild = leftchild + B_SIZE;

		float tleft = g_IntersectAABB(ray_ptr, pool + leftchild);
		float tright = g_IntersectAABB(ray_ptr, pool + rightchild);

		int flip = 0;

		int baseIndexNear = leftchild;
		int baseIndexFar = rightchild;

		float tEntryFarNode = tright;
		float tEntryNearNode = tleft;
		if (tright < tleft && tright > -99999) {
			baseIndexNear = rightchild;
			baseIndexFar = leftchild;
			tEntryFarNode = tleft;
			tEntryNearNode = tright;
		};

		if (tEntryFarNode > -99999) {
			int stackindex = ++stack[0];
			if (stackindex >= 32) printf("stack too small!. index: %i \n", stackindex);

			else {
				stack[stackindex] = baseIndexFar;
				stackAABBEntrypoints[stackindex] = tEntryFarNode;
				//printf("Added %i to stack location %i. Right child of %i \n", baseIndexFar, stackindex, index);
			}
		}
		if (tEntryNearNode > -99999) {
			int stackindex = ++stack[0];
			if (stackindex >= 32) printf("stack too small!. index: %i \n", stackindex);
			else {
				stack[stackindex] = baseIndexNear;
				stackAABBEntrypoints[stackindex] = tEntryNearNode;
				//printf("Added %i to stack location %i. This is the near child of %i \n", baseIndexNear, stackindex, index);

			}
		}

		return closest;
	}

	return closest;
}

// Finds the first geometry collision in its path. Returns a collision with t = -1 if none were found.
__device__ g_Collision g_nearestCollision(float* ray_ptr, bool use_bvh, int numGeometries, float* triangles, float* BVH, uint* orderedIndices)
{
	if (use_bvh)
	{
		int* stack = new int[32];
		float* aabbEntryPoints = new float[32];
		aabbEntryPoints[2] = -5000.0f;

		stack[0] = 1; //count, next one to evaluate;
		stack[1] = 0; //Root node

		g_Collision closest;
		closest.Nt.w = -1;

		while (stack[0] > 0)
		{
			int next = stack[0]--;
			//printf("next: stack[%i]: %i. AABB entrypoint: %f \n", next, stack[next], aabbEntryPoints[next]);

			if (closest.Nt.w != -1 && closest.Nt.w < aabbEntryPoints[next]) {
				continue;
			}

			g_Collision newcollision = g_TraverseBVHNode(ray_ptr, BVH, orderedIndices, triangles, stack[next], stack, aabbEntryPoints);

			if ((newcollision.Nt.w != -1 && newcollision.Nt.w < closest.Nt.w) || closest.Nt.w == -1) {
				closest = newcollision;
				//printf("closest t now %f \n", closest.Nt.w);
			}
		}
		delete stack;
		delete aabbEntryPoints;
		return closest;
	}
	else
	{
		float closestdist = 0xffffff;
		g_Collision closest;
		closest.Nt.w = -1;

		//Loop over all primitives to find the closest collision
		for (int i = 0; i < numGeometries; i++)
		{
			g_Collision collision = g_intersectTriangle(i, ray_ptr, triangles);
			float dist = collision.Nt.w;
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

// Generates and collects the nearest geometry intersections for the given ray queue
__global__ void
//__launch_bounds__(256, 6)
g_findCollisions(float* triangles, int numtriangles, float* rayQueue, void* collisions, bool useBVH, float* BVH, unsigned int* orderedIndices)
{
	uint numRays = ((uint*)rayQueue)[1];
	uint id = atomicInc(((uint*)rayQueue) + 3, 0xffffffff) + 1;

	while (id <= numRays)
	{
		if (id != 0)
		{
			float* rayptr = rayQueue + (id * R_SIZE);
			g_Collision collision = g_nearestCollision(rayptr, useBVH, numtriangles, triangles, BVH, orderedIndices);
			((g_Collision*)collisions)[id] = collision;
		}
		id = atomicInc(((uint*)rayQueue) + 3, 0xffffffff) + 1;
	}
}

// Checks for geometry intersections with the given shadow ray queue, and adds their energy to the intermediate screen buffer if unoccluded
__device__ void g_TraceShadowRay(float* shadowrays, int rayIndex, bool use_bvh, float* BVH, unsigned int* orderedIndices, int numGeometries, float* scene, float4* intermediate)
{
	int baseIndex = rayIndex * SR_SIZE;
	float maxt = shadowrays[baseIndex + SR_MAXT];
	bool collided = false;

	// Extracting shadow ray from ray queue
	/*float shadowray[R_SIZE] = {
		shadowrays[baseIndex + SR_OX],
		shadowrays[baseIndex + SR_OY],
		shadowrays[baseIndex + SR_OZ],
		shadowrays[baseIndex + SR_DX],
		shadowrays[baseIndex + SR_DY],
		shadowrays[baseIndex + SR_DZ]
	};*/
	float* shadowray = shadowrays + baseIndex;

	if (use_bvh)
	{
		// Initializing stack
		int* stack = new int[32];
		float* AABBEntryPoints = new float[32];
		stack[0] = 1; //count, next one to evaluate;
		stack[1] = 0; //root node

		// Traversing BVH
		while (stack[0] > 0)
		{
			int next = stack[0]--;
			g_Collision newcollision = g_TraverseBVHNode(shadowray, BVH, orderedIndices, scene, stack[next], stack, AABBEntryPoints);
			if (newcollision.Nt.w > 0 && newcollision.Nt.w < maxt)
			{
				collided = true;
				break;
			} // collision: light source is occluded
		}

		// Cleaning up
		delete stack;
		delete AABBEntryPoints;
	}
	else
	{
		// NOTE: Not using a BVH on the GPU will cause an automatic kernel shutdown after 5 seconds when there are too many triangles
		for (int i = 0; i < numGeometries; i++)
		{
			g_Collision shadowcollision = g_intersectTriangle(i, shadowray, scene, true);
			if (shadowcollision.Nt.w != -1 && shadowcollision.Nt.w < maxt)
			{
				collided = true;
				break;
			} // collision: light source is occluded
		}
	}

	if (collided) {
		return;
	}

	// Adding the unoccluded ray to the intermediate screen buffer
	float4 toadd = make_float4(shadowrays[baseIndex + SR_R], shadowrays[baseIndex + SR_G], shadowrays[baseIndex + SR_B], 1.0f);
	g_addToIntermediate(intermediate, shadowrays[baseIndex + SR_PIXX], shadowrays[baseIndex + SR_PIXY], toadd);
}

__global__ void
//__launch_bounds__(256, 6)
g_traceShadowRays(float* shadowrays, float* scene, float4* intermediate, float* BVH, unsigned int* orderedIndices, int numGeometries, bool use_bvh)
{
	uint numRays = ((uint*)shadowrays)[1];
	uint id = atomicInc(((uint*)shadowrays) + 2, 0xffffffff) + 1;

	while (id <= numRays)
	{
		if (id != 0)
		{
			float* rayptr = shadowrays + (id * SR_SIZE);
			g_TraceShadowRay(shadowrays, id, use_bvh, BVH, orderedIndices, numGeometries, scene, intermediate);
		}
		id = atomicInc(((uint*)shadowrays) + 2, 0xffffffff) + 1;
	}

}

__device__ float3 g_reflect(float3 D, float3 N)
{
	return D - N * (2 * (dot(D, N)));
}

__device__ float sqrLentgh(float3 a) //Not my typo. It was in the template and I'm keeping it to keep it consistent
{
	return a.x * a.x + a.y * a.y + a.z * a.z;
}

__device__ void g_addShadowRayToQueue(float3 ori, float3 dir, float R, float G, float B, float maxt, float pixelX, float pixelY, float* queue)
{
	int id = atomicInc(((uint*)queue) + 1, 0xffffffff) + 1;
	int queuesize = ((uint*)queue)[0];

	// array is full
	if (id > queuesize / SR_SIZE)
	{
		printf("ERROR: Queue overflow. Rays exceeded the %d indices of shadowray queue space.\n", queuesize / R_SIZE);
	}

	// adding ray to array
	int index = id * SR_SIZE; //Keep the first entry in the queue free, to save some metadata there (queuesize, currentCount)
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
}

__device__ float4 uint2float4(unsigned int in)
{
	return make_float4((in >> 16) & 255, (in >> 8) & 255, in & 255, 1.0f);
}

__device__ float4 skyboxColorAt(uint* skybox, float3 Direction, int skyboxWidth, int skyboxHeight, int skyboxPitch)
{
	float u;
	float v;
	u = 0.5f + (atan2f(-Direction.z, -Direction.x) * INV2PI);
	v = 0.5f - (asinf(-Direction.y) * INVPI);
	return uint2float4(skybox[(int)((skyboxWidth - 1) * u) + (int)((skyboxHeight - 1) * v) * skyboxPitch]);

}

__device__ void g_TraceRay(float* rays, int ray, g_Collision* collisions, float* newRays, float* shadowRays, bool bvhdebug, float4* intermediate, int numLights, float* lightPos, float3* lightColor, unsigned int* skybox, int skyboxWidth, int skyboxHeight, int skyboxPitch)
{
	//printf("traceray");

	float* ray_ptr = rays + (ray * R_SIZE);
	// unpacking ray pointer
	float3 direction = make_float3(ray_ptr[R_DX], ray_ptr[R_DY], ray_ptr[R_DZ]);
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
	g_Collision collision = collisions[ray];
	if (bvhdebug) {
		g_addToIntermediate(intermediate, pixelx, pixely, (make_float4(255, 0, 0, 0) * ray_ptr[R_BVHTRA]) * 8.0f);
		return;
	}

	// if ray collides
	if (collision.Nt.w > 0)
	{
		float3 collPos = make_float3(collision.PosRefr.x, collision.PosRefr.y, collision.PosRefr.z);
		float3 collN = make_float3(collision.Nt.x, collision.Nt.y, collision.Nt.z);
		// if opaque
		if (collision.PosRefr.w== 0.0f)
		{
			float4 albedo, reflection;
			float specularity = collision.ColorSpec.w;

			// diffuse aspect
			if (specularity < 1.0f)
			{
				//Generate shadow rays
				for (int light = 0; light < numLights; light++)
				{
					float3 lightPosition = make_float3(lightPos[light * 3 + 0], lightPos[light * 3 + 1], lightPos[light * 3 + 2]);
					float3 direction = normalize(lightPosition - collPos);
					float3 origin = collPos + ( direction * 0.00025f); //move away a little bit from the surface, to avoid self-collision in the outward direction.
					float maxt = (lightPos[light * 3 + 0] - collPos.x) / direction.x; //calculate t where the shadowray hits the light source. Because we don't want to count collisions that are behind the light source.


					float3 collisioncolor = make_float3(collision.ColorSpec.x, collision.ColorSpec.y, collision.ColorSpec.z);
					float3 lightColorAsFloat3 = lightColor[light];

					float3 shadowRayEnergy = collisioncolor * energy * (1 - specularity) * lightColorAsFloat3 * (max(0.0f, dot(collN, direction)) * INV4PI / sqrLentgh(lightPosition - collPos));

					if (shadowRayEnergy.x >= 1.0f | shadowRayEnergy.y >= 1.0f | shadowRayEnergy.z >= 1.0f) { //Ray will not contribute if all components < 1
						g_addShadowRayToQueue(origin, direction, shadowRayEnergy.x, shadowRayEnergy.y, shadowRayEnergy.z, maxt, pixelx, pixely, shadowRays);
					}
				}
			}

			// specular aspect
			if (specularity > 0)
			{
				float3 newdirection = g_reflect(direction, collN);
				float3 newOrigin = collPos + newdirection * 0.00001f;
				float* newray = new float[R_SIZE];
				newray[R_OX] = newOrigin.x;
				newray[R_OY] = newOrigin.y;
				newray[R_OZ] = newOrigin.z;
				newray[R_DX] = newdirection.x;
				newray[R_DY] = newdirection.y;
				newray[R_DZ] = newdirection.z;
				newray[R_INOBJ] = (float)inobj;
				newray[R_REFRIND] = refind;
				newray[R_BVHTRA] = 0;
				newray[R_DEPTH] = rdepth + 1;
				newray[R_PIXX] = pixelx;
				newray[R_PIXY] = pixely;
				newray[R_ENERGY] = energy * specularity;

				addRayToQueue(newray, newRays);
				delete newray;
				
			}
		}
		// if transparent
		else
		{
			float n1, n2;
			if (inobj) n1 = refind, n2 = 1.0f;
			else				n1 = refind, n2 = collision.PosRefr.w;
			float transition = n1 / n2;
			float costheta = dot(collN, direction * -1);
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
			if (Fr > 0.0f)
			{
				float3 newdirection = g_reflect(direction, collN);


				float3 newOrigin = collPos + newdirection * 0.00001f;
				float* newray = new float[R_SIZE];
				newray[R_OX] = newOrigin.x;
				newray[R_OY] = newOrigin.y;
				newray[R_OZ] = newOrigin.z;
				newray[R_DX] = newdirection.x;
				newray[R_DY] = newdirection.y;
				newray[R_DZ] = newdirection.z;
				newray[R_INOBJ] = (float)inobj;
				newray[R_REFRIND] = refind;
				newray[R_BVHTRA] = 0;
				newray[R_DEPTH] = rdepth + 1;
				newray[R_PIXX] = pixelx;
				newray[R_PIXY] = pixely;
				newray[R_ENERGY] = energy * Fr;

				addRayToQueue(newray, newRays);
				delete newray;

			}

			// Snell refraction
			if (Fr < 1.0f)
			{
				float3 newdirection = direction * transition + collN * (transition * costheta - sqrt(k));
				float3 newOrigin = collPos + newdirection * 0.00001f;
				float* newray = new float[R_SIZE];
				newray[R_OX] = newOrigin.x;
				newray[R_OY] = newOrigin.y;
				newray[R_OZ] = newOrigin.z;
				newray[R_DX] = newdirection.x;
				newray[R_DY] = newdirection.y;
				newray[R_DZ] = newdirection.z;
				newray[R_INOBJ] = (float)inobj;
				newray[R_REFRIND] = refind;
				newray[R_BVHTRA] = 0;
				newray[R_DEPTH] = rdepth + 1;
				newray[R_PIXX] = pixelx;
				newray[R_PIXY] = pixely;
				newray[R_ENERGY] = energy * (1 - Fr);

				addRayToQueue(newray, newRays);
				delete newray;


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
		//TODO: implement skybox
		//g_addToIntermediate(intermediate, pixelx, pixely, (g_Color(40, 20, 150) << 8) * energy);
		g_addToIntermediate(intermediate, pixelx, pixely, (skyboxColorAt(skybox, direction, skyboxWidth, skyboxHeight, skyboxPitch) * 255.0f) * energy);
	}
}

__global__ void g_Tracerays(float* rayQueue, void* collisions, float* newRays, float* shadowRays, bool bvhdebug, float4* intermediate, int numLights, float* lightPos, float3* lightColor, unsigned int* skybox, int skyboxWidth, int skyboxHeight, int skyboxPitch)
{
	uint numRays = ((uint*)rayQueue)[1];
	uint id = atomicInc(((uint*)rayQueue) + 4, 0xffffffff) + 1;

	while (id <= numRays)
	{
		if (id != 0)
		{
			float* rayptr = rayQueue + (id * R_SIZE);
			g_TraceRay(rayQueue, id, (g_Collision*)collisions, newRays, shadowRays, bvhdebug, intermediate, numLights, lightPos, lightColor, skybox, skyboxWidth, skyboxHeight, skyboxPitch);
		}
		id = atomicInc(((uint*)rayQueue) + 4, 0xffffffff) + 1;
	}
}


__global__ void copyIntermediateToScreen(unsigned int* screen, float4* intermediate, int pitch)
{
	int pixelx = blockIdx.x;
	int pixely = threadIdx.x;

	if (pixelx > SCRWIDTH || pixely > SCRHEIGHT) return;

	float4 pixelcolorf = intermediate[(int)pixelx + ((int)pixely * SCRWIDTH)] / 255.0f;

	if (pixelcolorf.x > 255)
	{
		pixelcolorf.x = 255;
	}
	if (pixelcolorf.y > 255)
	{
		pixelcolorf.y = 255;
	}
	if (pixelcolorf.z > 255)
	{
		pixelcolorf.z = 255;
	}

	unsigned int pixelcolori = (((int)pixelcolorf.x & 255) << 16) + (((int)pixelcolorf.y & 255) << 8) + ((int)pixelcolorf.z & 255);
	screen[pixelx + pixely * SCRWIDTH] = pixelcolori;
	intermediate[(int)pixelx + ((int)pixely * SCRWIDTH)] = make_float4(0, 0, 0, 0);
}