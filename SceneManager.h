#pragma once

class SceneManager
{
public:

	float numGeometries;
	float *triangles, *g_triangles;

	float numLights;
	float *lightPos, *g_lightPos;
	Color *lightColor;
	g_Color *g_lightColor;

	Skybox *skybox;
	uint *g_skybox;

	/*
		SCENE_OBJ_GLASS
		SCENE_OBJ_HALFREFLECT
		SCENE_STRESSTEST
		SCENE_TRIANGLETEST
		SCENE_FLOORONLY
		SCENE_CUBE
	*/
	enum SCENES {
		SCENE_OBJ_GLASS,
		SCENE_OBJ_HALFREFLECT,
		SCENE_STRESSTEST,
		SCENE_TRIANGLETEST,
		SCENE_FLOORONLY,
		SCENE_CUBE
	};

	void loadScene(SCENES scene, Camera camera);
	void loadObj(string filename, vec3 scale, vec3 translate, Material material);
	void createFloor(Material material);
	void initializeTriangle(int i, float * triangles);
	void moveSceneToGPU();

private:

};