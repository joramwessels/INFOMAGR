#pragma once

namespace Tmpl8 {


class Game
{
public:
	void SetTarget( Surface* surface ) { screen = surface; }
	void Init();
	void Shutdown();
	void Tick( float deltaTime );
	void MouseUp( int button ) { /* implement if you want to detect mouse button presses */ }
	void MouseDown( int button ) { /* implement if you want to detect mouse button presses */ }
	void MouseMove(int x, int y);
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
	Surface* screen;
	Camera camera;

	Collision nearestCollision(Ray* ray);

	Color TraceRay(Ray ray, int recursionDepth = 0);
	int numGeometries = 0;
	float* triangles;

	Color DirectIllumination(Collision collision);

	int numLights = 2;
	Light* lights;

	vec3 reflect(vec3 D, vec3 N);

	Skybox* skybox;

	//For moving camera, just for fun :)
	vec3 camerapos = { 0,0,0 };

	bool keyW = false, keyA = false, keyS = false, keyD = false, keyQ = false, keyE = false, keymin = false, keyplus = false, keyComma = false, keyPeriod = false;
	
	enum SCENES {
		SCENE_OBJ_GLASS,
		SCENE_OBJ_HALFREFLECT,
		SCENE_STRESSTEST,
		SCENE_TRIANGLETEST
	};


	void loadscene(SCENES scene);
	void loadobj(string filename, vec3 scale, vec3 translate, Material material);

	bool SSAA;
	bool DoF = false;
	bool use_bvh = false;

	//fps counter
	int frames = 0;
	int prevsecframes = 0;
	timer mytimer;
	float avgFrameTime;


	//BVH
	BVH* bvh;
	bool bvhdebug = false;
	void generateBVH() {
		//BVH GENERATION
		bvh = new BVH;
		printf("Starting BVH generation... \n");
		mytimer.reset();
		//bvh->Build(geometry, numGeometries);
		printf("BVH Generation done. Build time: %f, Depth: %i \n", mytimer.elapsed(), bvh->depth);
	}

	void initializeTriangle(int i, float* triangles);
	Collision intersectTriangle(int i, Ray ray, float* triangles, bool isShadowRay = false);

	//Animation
	bool animate = false;
};

}; // namespace Tmpl8