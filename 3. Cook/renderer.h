#pragma once

#define EPSILON		0.0001f
#define MAXDEPTH	4

namespace Tmpl8
{

class Renderer : public TheApp
{
public:
	// game flow methods
	void Init();
	float3 Trace( const Scene& sceneCopy, Ray& ray, int depth = 0 );
	float3 DirectIllumination( const Scene& sceneCopy, const float3& I, const float3& N );
	void Tick( float deltaTime );
	void UI();
	void Shutdown() { /* implement if you want to do things on shutdown */ }
	// input handling
	void MouseUp( int button ) { /* implement if you want to detect mouse button presses */ }
	void MouseDown( int button ) { /* implement if you want to detect mouse button presses */ }
	void MouseMove( int x, int y ) { mousePos.x = x, mousePos.y = y; }
	void MouseWheel( float y ) { /* implement if you want to handle the mouse wheel */ }
	void KeyUp( int key ) { /* implement if you want to handle keys */ }
	void KeyDown( int key ) { /* implement if you want to handle keys */ }
	// data members
	int2 mousePos;
	float4* accumulator;
	Scene scene;
	Camera camera;
	bool animating = true;
	float anim_time = 0;
	int passes = 1;
	// fps smoothing
	float avg = 10, alpha = 1;
};

} // namespace Tmpl8