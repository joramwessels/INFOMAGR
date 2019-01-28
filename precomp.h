// add your includes to this file instead of to individual .cpp files
// to enjoy the benefits of precompiled headers:
// - fast compilation
// - solve issues with the order of header files once (here)
// do not include headers in header files (ever).

// Prevent expansion clashes (when using std::min and std::max):
#define NOMINMAX

#define SCRWIDTH 512
#define SCRHEIGHT 512
#define MAX_RECURSION_DEPTH 5

// Scene Manager
#define FLOATS_PER_TRIANGLE 33
#define FLOATS_PER_LIGHTPOS 3
#define SCENE_ARRAY_SIZE 5000

// Triangle array positions
#define T_V0X 0
#define T_V0Y 1
#define T_V0Z 2
#define T_V1X 3
#define T_V1Y 4
#define T_V1Z 5
#define T_V2X 6
#define T_V2Y 7
#define T_V2Z 8
#define T_COLORR 9
#define T_COLORG 10
#define T_COLORB 11
#define T_SPECULARITY 12
#define T_REFRACTION 13
#define T_E0X 14
#define T_E0Y 15
#define T_E0Z 16
#define T_E1X 17
#define T_E1Y 18
#define T_E1Z 19
#define T_E2X 20
#define T_E2Y 21
#define T_E2Z 22
#define T_NX 23
#define T_NY 24
#define T_NZ 25
#define T_D 26
#define T_AABBMINX 27
#define T_AABBMAXX 28
#define T_AABBMINY 29
#define T_AABBMAXY 30
#define T_AABBMINZ 31
#define T_AABBMAXZ 32

// Shadow Ray array positions
#define SR_OX 0
#define SR_OY 1
#define SR_OZ 2
#define SR_DX 3
#define SR_DY 4
#define SR_DZ 5
#define SR_R 6
#define SR_G 7
#define SR_B 8
#define SR_MAXT 9
#define SR_PIXX 10
#define SR_PIXY 11
#define SR_SIZE 12

// Ray array positions
#define R_OX 0
#define R_OY 1
#define R_OZ 2
#define R_DX 3
#define R_DY 4
#define R_DZ 5
#define R_INOBJ 6
#define R_REFRIND 7
#define R_BVHTRA 8
#define R_DEPTH 9
#define R_PIXX 10
#define R_PIXY 11
#define R_ENERGY 12
#define R_SIZE 13

// BVH array positions
#define B_AABB_MINX 0
#define B_AABB_MAXX 1
#define B_AABB_MINY 2
#define B_AABB_MAXY 3
#define B_AABB_MINZ 4
#define B_AABB_MAXZ 5
#define B_LEFTFIRST 6
#define B_COUNT 7
#define B_SIZE 8

/*
BVH ARRAY POSITIONS IN FLOAT4 ARRAY:
float4 0
AABB_MINX
AABB_MINY
AABB_MINZ
LEFTFIRST

float4 1
AABB_MAXX
AABB_MAXY
AABB_MAXZ
COUNT
*/
#define B_F4_SIZE 2

#define INVPI					0.31830988618379067153776752674502872406891929148091289749533468811779359526845307018022760553250617191f
#define INV2PI					0.15915494309189533576888376337251436203445964574045644874766734405889679763422653509011380276625308595f
#define INV4PI					0.07957747154594766788444188168625718101722982287022822437383367202944839881711326754505690138312654297f

//#define SCRWIDTH 1920
//#define SCRHEIGHT 1080

// #define FULLSCREEN
// #define ADVANCEDGL	// faster if your system supports it

// Glew should be included first
#include <GL/glew.h>
// Comment for autoformatters: prevent reordering these two.
#include <GL/gl.h>

#ifdef _WIN32
// Followed by the Windows header
#include <Windows.h>

// Then import wglext: This library tries to include the Windows
// header WIN32_LEAN_AND_MEAN, unless it was already imported.
#include <GL/wglext.h>

// Extra definitions for redirectIO
#include <fcntl.h>
#include <io.h>
#endif

// External dependencies:
#include <FreeImage.h>
#include <SDL2/SDL.h>

// C++ headers
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

// Namespaced C headers:
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// Header for AVX, and every technology before it.
// If your CPU does not support this, include the appropriate header instead.
// See: https://stackoverflow.com/a/11228864/2844473
#include <immintrin.h>

// clang-format off

// "Leak" common namespaces to all compilation units. This is not standard
// C++ practice but a mere simplification for this small project.
using namespace std;

#include "surface.h"
#include "template.h"

using namespace Tmpl8;
//tiny obj loader

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include "GPUStuff.cuh"

// Own headers
#include "Structs.h"
//#include "Geometry.h"
//#include "Sphere.h"
//#include "Plane.h"
//#include "Triangle.h"
#include "Camera.h"
#include "BVH.h"
#include "SceneManager.h"

#include "game.h"
// clang-format on

