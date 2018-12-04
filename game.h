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

	Skybox* skybox;

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

	bool SSAA;

	//fps counter
	int frames = 0;
	int prevsecframes = 0;
	timer mytimer;
};

}; // namespace Tmpl8