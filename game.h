#pragma once

namespace Tmpl8 {

#define INVPI					0.31830988618379067153776752674502872406891929148091289749533468811779359526845307018022760553250617191f
#define INV2PI					0.15915494309189533576888376337251436203445964574045644874766734405889679763422653509011380276625308595f
#define INV4PI					0.07957747154594766788444188168625718101722982287022822437383367202944839881711326754505690138312654297f


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
		default:
			break;
		}
	}
private:
	Surface* screen;
	Camera camera;

	Collision nearestCollision(Ray ray);

	Color TraceRay(Ray ray, int recursionDepth = 0);
	Sphere* sphere;
	Plane* plane;

	int numGeometries = 0;
	Geometry** geometry;

	Color DirectIllumination(Collision collision);

	int numLights = 2;
	Light* lights;

	vec3 reflect(vec3 D, vec3 N);

	Surface* skybox;

	//For moving camera, just for fun :)
	vec3 camerapos = { 0,0,0 };

	bool keyW = false, keyA = false, keyS = false, keyD = false, keyQ = false, keyE = false;
	
	enum SCENES {
		SCENE_TEST,
		SCENE_OBJ_GLASS,
		SCENE_OBJ_HALFREFLECT,
		SCENE_LIGHTING_AMBIENT,
		SCENE_LIGHTING_SPOT,
		SCENE_LIGHTING_DIRECTIONAL,
		SCENE_PERFORMANCE
	};


	void loadscene(SCENES scene);
	void loadobj(string filename, vec3 scale, vec3 translate, Material material);

	//fps counter
	int frames = 0;
	int prevsecframes = 0;
	timer mytimer;
};

}; // namespace Tmpl8