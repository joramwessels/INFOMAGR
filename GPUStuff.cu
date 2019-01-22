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

__device__ float g_random1 = 0.529669f;
__device__ float g_random2 = 0.083422f;
__device__ float g_random3 = 0.281753f;
__device__ float g_random4 = 0.506648f;
__device__ float g_random5 = 0.438385f;
__device__ float g_random6 = 0.162733f;
__device__ float g_random7 = 0.538243f;
__device__ float g_random8 = 0.769904f;


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

__device__ void addRayToQueue(float* ray, float* queue)
{
	int id = atomicInc(((uint*)queue) + 1, 0xffffffff) + 1;
	int queuesize = ((uint*)queue)[0];

	if (id > queuesize / R_SIZE)
	{
		printf("ERROR: Queue overflow. Rays exceeded the %i indices of queue space.\n", (int)(queuesize / R_SIZE));
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

__global__ void GeneratePrimaryRay(float* rayQueue, bool DoF, float3 position, float3 virtualScreenCornerTL, float3 virtualScreenCornerTR, float3 virtualScreenCornerBL, bool SSAA)
{
	if (threadIdx.x == 0 & blockIdx.x == 0) {
		((int*)rayQueue)[0] = ((SCRHEIGHT * SCRWIDTH * 4) + 1) * R_SIZE;
		//((uint*)rayQueue)[2] = 0;

	}

	uint numRays = SCRWIDTH * SCRHEIGHT;
	uint raynum = atomicInc(((uint*)rayQueue) + 2, 0xffffffff);

	while (raynum < numRays) {
		int pixelx = raynum % SCRWIDTH;
		int pixely = raynum / SCRWIDTH;
		//int pixelx = threadIdx.x;
		//int pixely = blockIdx.x;

		//printf("id: %i, x: %i, y: %i \n", raynum, pixelx, pixely);

		if (pixelx > SCRWIDTH || pixely > SCRHEIGHT) printf("wtf");

		//Generate the ray
		if (SSAA) {
			float* ray1 = generateRayTroughVirtualScreen((float)pixelx + g_random1, (float)pixely + g_random2, DoF, position, virtualScreenCornerTL, virtualScreenCornerTR, virtualScreenCornerBL);

			ray1[R_INOBJ] = 0;
			ray1[R_REFRIND] = 1.0f;
			ray1[R_BVHTRA] = 0;
			ray1[R_DEPTH] = 0;
			ray1[R_PIXX] = pixelx;
			ray1[R_PIXY] = pixely;
			ray1[R_ENERGY] = 0.25f;

			addRayToQueue(ray1, rayQueue);
			delete ray1;

			float* ray2 = generateRayTroughVirtualScreen((float)pixelx + g_random3, (float)pixely + g_random4, DoF, position, virtualScreenCornerTL, virtualScreenCornerTR, virtualScreenCornerBL);

			ray2[R_INOBJ] = 0;
			ray2[R_REFRIND] = 1.0f;
			ray2[R_BVHTRA] = 0;
			ray2[R_DEPTH] = 0;
			ray2[R_PIXX] = pixelx;
			ray2[R_PIXY] = pixely;
			ray2[R_ENERGY] = 0.25f;

			addRayToQueue(ray2, rayQueue);
			delete ray2;

			float* ray3 = generateRayTroughVirtualScreen((float)pixelx + g_random5, (float)pixely + g_random6, DoF, position, virtualScreenCornerTL, virtualScreenCornerTR, virtualScreenCornerBL);

			ray3[R_INOBJ] = 0;
			ray3[R_REFRIND] = 1.0f;
			ray3[R_BVHTRA] = 0;
			ray3[R_DEPTH] = 0;
			ray3[R_PIXX] = pixelx;
			ray3[R_PIXY] = pixely;
			ray3[R_ENERGY] = 0.25f;

			addRayToQueue(ray3, rayQueue);
			delete ray3;

			float* ray4 = generateRayTroughVirtualScreen((float)pixelx + g_random7, (float)pixely + g_random8, DoF, position, virtualScreenCornerTL, virtualScreenCornerTR, virtualScreenCornerBL);

			ray4[R_INOBJ] = 0;
			ray4[R_REFRIND] = 1.0f;
			ray4[R_BVHTRA] = 0;
			ray4[R_DEPTH] = 0;
			ray4[R_PIXX] = pixelx;
			ray4[R_PIXY] = pixely;
			ray4[R_ENERGY] = 0.25f;

			addRayToQueue(ray4, rayQueue);
			delete ray4;

		}
		else {
			float* ray = generateRayTroughVirtualScreen(pixelx, pixely, DoF, position, virtualScreenCornerTL, virtualScreenCornerTR, virtualScreenCornerBL);

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

__device__ float dot(float3 a, float3 b)
{
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

__device__ float3 cross(float3 a, float3 b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ struct g_Collision
{
	float3 N;
	float3 Pos;
	float t;
	//Color colorAt;
	float R;
	float G;
	float B;
	float refraction;
	float specularity;

	//bool isTranslated;
	//vec3 translation = { 0, 0, 0 };
};

__device__ g_Collision g_intersectTriangle(int i, float* ray_ptr, float * triangles, bool isShadowRay = false)
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
	collision.t = -1;

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
			if (NdotR > 0) collision.N = N * -1;
			else collision.N = N;
			collision.Pos = P;
			return collision;
		}
	}
	return collision;
}

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


// Recursively traverses the BVH tree from the given node to find a collision. Returns collision with t = -1 if none were found.
__device__ g_Collision g_TraverseBVHNode(float* ray_ptr, float* pool, uint* orderedIndices, float* scene, int index, int* stack, float* stackAABBEntrypoints)
{
	g_Collision closest;
	closest.t = -1;

	ray_ptr[R_BVHTRA]++;

	//vec3 ray_origin = { ray_ptr[R_OX], ray_ptr[R_OY], ray_ptr[R_OZ] };
	//vec3 ray_direction = { ray_ptr[R_DX], ray_ptr[R_DY], ray_ptr[R_DZ] };
	int count = pool[index + B_COUNT];
	//printf("Count: %i \n", count);

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
			float dist = collision.t;
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
		//printf("Not leaf \n");
		// Check both children and return the closest collision if both intersected
		//AABB::AABBIntersection tleft = pool[(int)pool[index + B_LEFTFIRST]].bounds.Intersects(ray_origin, ray_direction);
		//AABB::AABBIntersection tright = pool[node->leftFirst + 1].bounds.Intersects(ray_origin, ray_direction);
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

		g_Collision colclose, colfar;
		colclose.t = -1;
		colfar.t = -1;

		if (tEntryNearNode > -99999) {
			//colclose = g_TraverseBVHNode(ray_ptr, pool, orderedIndices, scene, baseIndexNear);
			//if (colclose.t < tEntryFarNode && colclose.t > 0) {
			//	return colclose;
			//}
			int index = 2 + stack[0]++;
			if (index >= 2048) printf("stack too small!. index: %i \n", index);
			else {
				stack[index] = baseIndexNear;
				stackAABBEntrypoints[index] = tEntryNearNode;

				//printf("Added %i to stack location %i \n", baseIndexNear, index);

			}
		}
		if (tEntryFarNode > -99999) {
			//colfar = g_TraverseBVHNode(ray_ptr, pool, orderedIndices, scene, baseIndexFar);
			int index = 2 + stack[0]++;
			if (index >= 2048) printf("stack too small!. index: %i \n", index);

			else {
				stack[index] = baseIndexFar;
				stackAABBEntrypoints[index] = tEntryFarNode;

				//printf("Added %i to stack location %i \n", baseIndexFar, index);
			}
		}

		//if (colfar.t == -1) return colclose;
		//if (colclose.t == -1) return colfar;
		return (colclose.t < colfar.t ? colclose : colfar);
	}

	return closest;
}

__device__ g_Collision g_nearestCollision(float* ray_ptr, bool use_bvh, int numGeometries, float* triangles, float* BVH, uint* orderedIndices)
{
	if (use_bvh)
	{
		//printf("BVH TRAVERSAL ");
		int* stack = new int[2048];
		float* aabbEntryPoints = new float[2048];

		stack[0] = 1; //count;
		stack[1] = 2; //next one to evaluate
		stack[2] = 0; //root node

		g_Collision closest;
		closest.t = -1;

		while ((stack[1] - 2) < stack[0])
		{
			int next = stack[1]++;

			if (closest.t > 0 && closest.t < (aabbEntryPoints[next])) {
				delete stack;
				delete aabbEntryPoints;
				return closest;
			}

			g_Collision newcollision = g_TraverseBVHNode(ray_ptr, BVH, orderedIndices, triangles, stack[next], stack, aabbEntryPoints);

			if ((newcollision.t != -1 && newcollision.t < closest.t) || closest.t == -1) {
				closest = newcollision;
			}
		}
		delete stack;
		delete aabbEntryPoints;
		return closest;

		//return g_TraverseBVHNode(ray_ptr, BVH, orderedIndices, triangles, 0);
	}
	else
	{
		float closestdist = 0xffffff;
		g_Collision closest;
		closest.t = -1;

		//Loop over all primitives to find the closest collision
		for (int i = 0; i < numGeometries; i++)
		{
			//Collision collision = geometry[i]->Intersect(*ray);
			//printf("Trying to intersect ray with triangle %i... \n", i);
			g_Collision collision = g_intersectTriangle(i, ray_ptr, triangles);
			//printf("(nearestcollision) got collision with t %f. \n", collision.t);



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

__device__ g_Collision test(int i) {
	g_Collision coll;
	coll.t = i;
	return coll;
}
__global__ void g_findCollisions(float* triangles, int numtriangles, float* rayQueue, void* collisions, bool useBVH, float* BVH, uint* orderedIndices)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		((uint*)rayQueue)[3] = 0;
	}

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