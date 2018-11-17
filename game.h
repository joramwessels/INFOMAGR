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
	void KeyUp( int key ) { /* implement if you want to handle keys */ }
	void KeyDown( int key ) { /* implement if you want to handle keys */ }
private:
	Surface* screen;
	Camera camera;

	Collision nearestCollision(Ray ray);

	Color TraceRay(Ray ray, int recursionDepth = 0);
	Sphere* sphere;
	Plane* plane;

	int numGeometries = 2;
	Geometry** geometry;

	Color DirectIllumination(Collision collision);

	int numLights = 2;
	Light* lights;

	vec3 reflect(vec3 D, vec3 N);

	//For moving camera, just for fun :)
	vec3 camerapos = { 0,0,0 };

};

}; // namespace Tmpl8