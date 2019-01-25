#include "precomp.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "lib\tinyobjloader\tiny_obj_loader.h"

// Loads a predefined scene into the scene manager
void SceneManager::loadScene(SceneManager::SCENES scene, Camera camera)
{
	triangles = new float[SCENE_ARRAY_SIZE * FLOATS_PER_TRIANGLE];

	switch (scene)
	{
	case SCENE_FLOORONLY:
	{
		createFloor(Material(0.0f, 0.0f, 0xffffff));
		numGeometries = 4;

		//Vertex positions
		triangles[2 * FLOATS_PER_TRIANGLE + T_V0X] = -10.0f; //v0.x
		triangles[2 * FLOATS_PER_TRIANGLE + T_V0Y] = 1.5f; //v0.y
		triangles[2 * FLOATS_PER_TRIANGLE + T_V0Z] = 10.0f; //v0.z

		triangles[2 * FLOATS_PER_TRIANGLE + T_V1X] = -10.0f; //v1.x
		triangles[2 * FLOATS_PER_TRIANGLE + T_V1Y] = 1.5f; //v1.y
		triangles[2 * FLOATS_PER_TRIANGLE + T_V1Z] = -10.0f; //v1.z

		triangles[2 * FLOATS_PER_TRIANGLE + T_V2X] = -10.0f; //v2.x
		triangles[2 * FLOATS_PER_TRIANGLE + T_V2Y] = 0.0f; //v2.y
		triangles[2 * FLOATS_PER_TRIANGLE + T_V2Z] = -10.0f; //v2.z

		//TODO: Completely remove material class?
		//Color
		triangles[2 * FLOATS_PER_TRIANGLE + T_COLORR] = 255; //R
		triangles[2 * FLOATS_PER_TRIANGLE + T_COLORG] = 255; //G
		triangles[2 * FLOATS_PER_TRIANGLE + T_COLORB] = 255; //B

		//Material properties
		triangles[2 * FLOATS_PER_TRIANGLE + T_SPECULARITY] = 0.0f; //Specularity
		triangles[2 * FLOATS_PER_TRIANGLE + T_REFRACTION] = 0.0f; //Refractionindex

		//Calculate the edges, normal and D
		initializeTriangle(2, triangles);

		//Vertex positions
		triangles[3 * FLOATS_PER_TRIANGLE + T_V0X] = 10.0f; //v0.x
		triangles[3 * FLOATS_PER_TRIANGLE + T_V0Y] = 1.5f; //v0.y
		triangles[3 * FLOATS_PER_TRIANGLE + T_V0Z] = 10.0f; //v0.z

		triangles[3 * FLOATS_PER_TRIANGLE + T_V1X] = 10.0f; //v1.x
		triangles[3 * FLOATS_PER_TRIANGLE + T_V1Y] = 1.5f; //v1.y
		triangles[3 * FLOATS_PER_TRIANGLE + T_V1Z] = -10.0f; //v1.z

		triangles[3 * FLOATS_PER_TRIANGLE + T_V2X] = 10.0f; //v2.x
		triangles[3 * FLOATS_PER_TRIANGLE + T_V2Y] = 0.0f; //v2.y
		triangles[3 * FLOATS_PER_TRIANGLE + T_V2Z] = -10.0f; //v2.z

		//TODO: Completely remove material class?
		//Color
		triangles[3 * FLOATS_PER_TRIANGLE + T_COLORR] = 255; //R
		triangles[3 * FLOATS_PER_TRIANGLE + T_COLORG] = 255; //G
		triangles[3 * FLOATS_PER_TRIANGLE + T_COLORB] = 255; //B

		//Material properties
		triangles[3 * FLOATS_PER_TRIANGLE + T_SPECULARITY] = 0.0f; //Specularity
		triangles[3 * FLOATS_PER_TRIANGLE + T_REFRACTION] = 0.0f; //Refractionindex

		//Calculate the edges, normal and D
		initializeTriangle(3, triangles);


		numLights = 3;
		lightPos = new float[numLights * 3];
		lightColor = new Color[numLights];

		lightPos[0] = -5.0f; //X
		lightPos[1] = -5.0f; //Y
		lightPos[2] = 20.0f; //Z
		lightColor[0] = 0xffffff;
		lightColor[0] = lightColor[0] * 700;

		lightPos[(1 * 3) + 0] = 5.0f; //X
		lightPos[(1 * 3) + 1] = -5.0f; //Y
		lightPos[(1 * 3) + 2] = 0.0f; //Z
		lightColor[1] = 0xffffff;
		lightColor[1] = lightColor[1] * 700;

		lightPos[(2 * 3) + 0] = -5.0f; //X
		lightPos[(2 * 3) + 1] = -5.0f; //Y
		lightPos[(2 * 3) + 2] = 0.0f; //Z
		lightColor[2] = 0xffffff;
		lightColor[2] = lightColor[2] * 700;


		skybox = new Skybox("assets\\skybox4.jpg");
		//generateBVH(); // TODO
		break;

	}
	case SCENE_CUBE:
	{
		numGeometries = 0;
		//createfloor(Material(0.0f, 0.0f, 0xffffff));

		loadObj("assets\\cube.obj", { 1, 1, 1 }, { 0.05, 0, 3 }, Material(0.0f, 0.0f, 0xff0000));

		numLights = 3;
		lightPos = new float[numLights * 3];
		lightColor = new Color[numLights];

		lightPos[0] = -5.0f; //X
		lightPos[1] = -5.0f; //Y
		lightPos[2] = 20.0f; //Z
		lightColor[0] = 0xffffff;
		lightColor[0] = lightColor[0] * 700;

		lightPos[(1 * 3) + 0] = 5.0f; //X
		lightPos[(1 * 3) + 1] = -5.0f; //Y
		lightPos[(1 * 3) + 2] = 0.0f; //Z
		lightColor[1] = 0xffffff;
		lightColor[1] = lightColor[1] * 700;

		lightPos[(2 * 3) + 0] = -5.0f; //X
		lightPos[(2 * 3) + 1] = -5.0f; //Y
		lightPos[(2 * 3) + 2] = 0.0f; //Z
		lightColor[2] = 0xffffff;
		lightColor[2] = lightColor[2] * 700;

		skybox = new Skybox("assets\\skybox4.jpg");
		//generateBVH(); // TODO
		break;

	}
	case SCENE_OBJ_GLASS:
	{
		camera.rotate({ -20, 180, 0 });
		//geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, Material(Material(0.0f, 0.0f, Material::TEXTURE, new Surface("assets\\tiles.jpg"))));

		numGeometries = 0;
		createFloor(Material(1.0f, 0.0f, 0xffffff));
		loadObj("assets\\MaleLow.obj", { 0.5f, -0.5f, 0.5f }, { 0, 1.5f, -9 }, Material(0.0f, 1.52f, 0xffffff));

		numLights = 3;
		lightPos = new float[numLights * 3];
		lightColor = new Color[numLights];

		lightPos[0] = -5.0f; //X
		lightPos[1] = -5.0f; //Y
		lightPos[2] = 20.0f; //Z
		lightColor[0] = 0xffffff;
		lightColor[0] = lightColor[0] * 700;

		lightPos[(1 * 3) + 0] = 5.0f; //X
		lightPos[(1 * 3) + 1] = -5.0f; //Y
		lightPos[(1 * 3) + 2] = 0.0f; //Z
		lightColor[1] = 0xffffff;
		lightColor[1] = lightColor[1] * 700;

		lightPos[(2 * 3) + 0] = -5.0f; //X
		lightPos[(2 * 3) + 1] = -5.0f; //Y
		lightPos[(2 * 3) + 2] = 0.0f; //Z
		lightColor[2] = 0xffffff;
		lightColor[2] = lightColor[2] * 700;


		skybox = new Skybox("assets\\skybox4.jpg");
		// generateBVH(); // TODO
		break;
	}
	case SCENE_TRIANGLETEST:
	{
		//camera.rotate({ -20, 180, 0 });
		//geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, Material(Material(0.0f, 0.0f, Material::TEXTURE, new Surface("assets\\tiles.jpg"))));

		numGeometries = 0;
		createFloor(Material(0.0f, 0.0f, 0xffffff));

		loadObj("assets\\cube.obj", { 1.0f, 1.0f, 1.0f }, { 0, 0, 0 }, Material(1.0f, 0.0f, 0xffffff));
		loadObj("assets\\cube.obj", { 1.0f, 1.0f, 1.0f }, { -2, 0, 0 }, Material(1.0f, 0.0f, 0xffffff));
		loadObj("assets\\cube.obj", { 1.0f, 1.0f, 1.0f }, { 2, 0, 0 }, Material(1.0f, 0.0f, 0xffffff));

		numLights = 3;
		lightPos = new float[numLights * 3];
		lightColor = new Color[numLights];

		lightPos[0] = -5.0f; //X
		lightPos[1] = -5.0f; //Y
		lightPos[2] = 20.0f; //Z
		lightColor[0] = 0xffffff;
		lightColor[0] = lightColor[0] * 700;

		lightPos[(1 * 3) + 0] = 5.0f; //X
		lightPos[(1 * 3) + 1] = -5.0f; //Y
		lightPos[(1 * 3) + 2] = 0.0f; //Z
		lightColor[1] = 0xffffff;
		lightColor[1] = lightColor[1] * 700;

		lightPos[(2 * 3) + 0] = -5.0f; //X
		lightPos[(2 * 3) + 1] = -5.0f; //Y
		lightPos[(2 * 3) + 2] = 0.0f; //Z
		lightColor[2] = 0xffffff;
		lightColor[2] = lightColor[2] * 700;


		skybox = new Skybox("assets\\skybox4.jpg");
		// generateBVH(); // TODO
		break;
	}

	case SCENE_OBJ_HALFREFLECT:
	{
		//camera.rotate({ -40, 0, 0 });
		//geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, Material(Material(0.0f, 0.0f, Material::TEXTURE, new Surface("assets\\tiles.jpg"))));

		numGeometries = 0;
		createFloor(Material(0.0f, 0.0f, 0xffffff));

		loadObj("assets\\Banana.obj", { 0.02f, -0.02f, 0.02f }, { -2.5, 1.5f, 10 }, Material(0.5f, 0.0f, 0xffff00));

		numLights = 3;
		lightPos = new float[numLights * 3];
		lightColor = new Color[numLights];

		lightPos[0] = -5.0f; //X
		lightPos[1] = -5.0f; //Y
		lightPos[2] = 20.0f; //Z
		lightColor[0] = 0xffffff;
		lightColor[0] = lightColor[0] * 700;

		lightPos[(1 * 3) + 0] = 5.0f; //X
		lightPos[(1 * 3) + 1] = -5.0f; //Y
		lightPos[(1 * 3) + 2] = 0.0f; //Z
		lightColor[1] = 0xffffff;
		lightColor[1] = lightColor[1] * 700;

		lightPos[(2 * 3) + 0] = -5.0f; //X
		lightPos[(2 * 3) + 1] = -5.0f; //Y
		lightPos[(2 * 3) + 2] = 0.0f; //Z
		lightColor[2] = 0xffffff;
		lightColor[2] = lightColor[2] * 700;


		skybox = new Skybox("assets\\skybox4.jpg");

		// generateBVH(); // TODO
		break;
	}
	case SCENE_STRESSTEST:
	{
		delete triangles;
		triangles = new float[900004 * FLOATS_PER_TRIANGLE];
		//cudaMalloc(&g_triangles, 900004 * FLOATS_PER_TRIANGLE * sizeof(float)); // TODo

		//Set up the scene
		numGeometries = 0;
		createFloor(Material(0.0f, 0.0f, 0xffffff));
		//geometry[0] = new Plane(vec3(0, 1, 0), -1.5f, Material(Material(0.0f, 0.0f, Material::TEXTURE, new Surface("assets\\tiles.jpg"))));

		for (size_t i = 0; i < 200; i++)
		{
			float ix = i % 14;
			float iy = i / 14;


			loadObj("assets\\MaleLow.obj", { 0.2f, -0.2f, 0.2f }, { ix * 3 - 10, 1.5f, -5 - (2 * iy) }, Material(0.0f, 0.0f, 0xffffff));
			loadObj("assets\\MaleLow.obj", { 0.2f, -0.2f, 0.2f }, { ix * 3 - 10, -2.0f, -5 - (2 * iy) }, Material(0.0f, 0.0f, 0xffffff));
			loadObj("assets\\MaleLow.obj", { 0.2f, -0.2f, 0.2f }, { ix * 3 - 10, -5.5f, -5 - (2 * iy) }, Material(0.0f, 0.0f, 0xffffff));

		}

		camera.rotate({ 30, 150, 0 });
		camera.move({ 0.0f, -5.0f, 0.0f });

		numLights = 3;
		lightPos = new float[numLights * 3];
		lightColor = new Color[numLights];

		lightPos[0] = -5.0f; //X
		lightPos[1] = -5.0f; //Y
		lightPos[2] = 20.0f; //Z
		lightColor[0] = 0xffffff;
		lightColor[0] = lightColor[0] * 700;

		lightPos[(1 * 3) + 0] = 5.0f; //X
		lightPos[(1 * 3) + 1] = -5.0f; //Y
		lightPos[(1 * 3) + 2] = 0.0f; //Z
		lightColor[1] = 0xffffff;
		lightColor[1] = lightColor[1] * 700;

		lightPos[(2 * 3) + 0] = -5.0f; //X
		lightPos[(2 * 3) + 1] = -5.0f; //Y
		lightPos[(2 * 3) + 2] = 0.0f; //Z
		lightColor[2] = 0xffffff;
		lightColor[2] = lightColor[2] * 700;

		skybox = new Skybox("assets\\skybox4.jpg");
		// generateBVH(); // TODO
		break;
	}
	default:
		break;
	}
}

// Loads a collection of triangles into the scene using tinyobjloader
void SceneManager::loadObj(string filename, vec3 scale, vec3 translate, Material material)
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

// Adds a standard size floor with the given material to the scene
void SceneManager::createFloor(Material material)
{
	numGeometries = 2;

	//Vertex positions
	triangles[0 + T_V0X] = -10.0f; //v0.x
	triangles[0 + T_V0Y] = 1.5f; //v0.y
	triangles[0 + T_V0Z] = 10.0f; //v0.z

	triangles[0 + T_V1X] = -10.0f; //v1.x
	triangles[0 + T_V1Y] = 1.5f; //v1.y
	triangles[0 + T_V1Z] = -10.0f; //v1.z

	triangles[0 + T_V2X] = 10.0f; //v2.x
	triangles[0 + T_V2Y] = 1.5f; //v2.y
	triangles[0 + T_V2Z] = 10.0f; //v2.z

	//TODO: Completely remove material class?
	//Color
	triangles[0 + T_COLORR] = material.color.R; //R
	triangles[0 + T_COLORG] = material.color.G; //G
	triangles[0 + T_COLORB] = material.color.B; //B

	//Material properties
	triangles[0 + T_SPECULARITY] = material.specularity; //Specularity
	triangles[0 + T_REFRACTION] = material.refractionIndex; //Refractionindex

	//Calculate the edges, normal and D
	initializeTriangle(0, triangles);



	//Vertex positions
	triangles[FLOATS_PER_TRIANGLE + T_V0X] = 10.0f; //v0.x
	triangles[FLOATS_PER_TRIANGLE + T_V0Y] = 1.5f; //v0.y
	triangles[FLOATS_PER_TRIANGLE + T_V0Z] = -10.0f; //v0.z

	triangles[FLOATS_PER_TRIANGLE + T_V1X] = -10.0f; //v1.x
	triangles[FLOATS_PER_TRIANGLE + T_V1Y] = 1.5f; //v1.y
	triangles[FLOATS_PER_TRIANGLE + T_V1Z] = -10.0f; //v1.z

	triangles[FLOATS_PER_TRIANGLE + T_V2X] = 10.0f; //v2.x
	triangles[FLOATS_PER_TRIANGLE + T_V2Y] = 1.5f; //v2.y
	triangles[FLOATS_PER_TRIANGLE + T_V2Z] = 10.0f; //v2.z

	//TODO: Completely remove material class?
	//Color
	triangles[FLOATS_PER_TRIANGLE + T_COLORR] = material.color.R;//R
	triangles[FLOATS_PER_TRIANGLE + T_COLORG] = material.color.G; //G
	triangles[FLOATS_PER_TRIANGLE + T_COLORB] = material.color.B;//B

	//Material properties
	triangles[FLOATS_PER_TRIANGLE + T_SPECULARITY] = material.specularity; //Specularity
	triangles[FLOATS_PER_TRIANGLE + T_REFRACTION] = material.refractionIndex; //Refractionindex

	//Calculate the edges, normal and D
	initializeTriangle(1, triangles);

}

// Calculates the edges, normal, D and aabb for a triangle
void SceneManager::initializeTriangle(int i, float * triangles)
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

// Copies all scene manager fields to the GPU
void SceneManager::moveSceneToGPU()
{
	// Triangles
	cudaMalloc(&g_triangles, SCENE_ARRAY_SIZE * FLOATS_PER_TRIANGLE * sizeof(float));
	cudaMemcpy(g_triangles, triangles, numGeometries * FLOATS_PER_TRIANGLE * sizeof(float), cudaMemcpyHostToDevice);

	// Lights
	cudaMalloc(&g_lightPos, numLights * FLOATS_PER_LIGHTPOS * sizeof(float));
	cudaMalloc(&g_lightColor, numLights * sizeof(g_Color));
	cudaMemcpy(g_lightPos, lightPos, numLights * FLOATS_PER_LIGHTPOS * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(g_lightColor, lightColor, numLights * sizeof(g_Color), cudaMemcpyHostToDevice);
	
	// Skybox
	cudaMalloc(&g_skybox, (skybox->texture->GetWidth() * skybox->texture->GetHeight()) * sizeof(uint));
	cudaMemcpy(g_skybox, skybox->texture->GetBuffer(), skybox->texture->GetWidth() * skybox->texture->GetHeight() * sizeof(uint), cudaMemcpyHostToDevice);
}