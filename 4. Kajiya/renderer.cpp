#include "precomp.h"

// -----------------------------------------------------------
// Initialize the renderer
// -----------------------------------------------------------
void Renderer::Init()
{
	// create fp32 rgb pixel buffer to render to
	accumulator = (float4*)MALLOC64( SCRWIDTH * SCRHEIGHT * 16 );
	ClearAccumulator();
}

// -----------------------------------------------------------
// Restart converging
// -----------------------------------------------------------
void Renderer::ClearAccumulator()
{
	spp = 0;
	memset( accumulator, 0, SCRWIDTH * SCRHEIGHT * 16 );
}

// -----------------------------------------------------------
// Helpers for mirrors and dielectrics
// -----------------------------------------------------------
float3 Renderer::HandleMirror( const Ray& ray, uint& seed, const float3& I, const float3& N, const int depth )
{
	float3 R = reflect( ray.D, N );
	Ray r( I + R * EPSILON, R );
	return Sample( r, seed, depth + 1 );
}
float3 Renderer::HandleDielectric( const Ray& ray, uint& seed, const float3& I, const float3& N, const int depth )
{
	float3 R = reflect( ray.D, N );
	Ray r( I + R * EPSILON, R );
	float n1 = ray.inside ? 1.2f : 1, n2 = ray.inside ? 1 : 1.2f;
	float eta = n1 / n2, cosi = dot( -ray.D, N );
	float cost2 = 1.0f - eta * eta * (1 - cosi * cosi);
	float Fr = 1;
	if (cost2 > 0)
	{
		float a = n1 - n2, b = n1 + n2, R0 = (a * a) / (b * b), c = 1 - cosi;
		Fr = R0 + (1 - R0) * (c * c * c * c * c);
		float3 T = eta * ray.D + ((eta * cosi - sqrtf( fabs( cost2 ) )) * N);
		Ray t( I + T * EPSILON, T );
		t.inside = !ray.inside;
		if (RandomFloat( seed ) > Fr) return Sample( t, seed, depth + 1 );
	}
	return Sample( r, seed, depth + 1 );
}

// -----------------------------------------------------------
// Evaluate light transport
// -----------------------------------------------------------
float3 Renderer::Sample( Ray& ray, uint& seed, int depth )
{
	// intersect the ray with the scene
	scene.FindNearest( ray );
	if (ray.objIdx == -1) /* ray left the scene */ return 0;
	if (depth > MAXDEPTH) /* bounced too many times */ return 0;
	// gather shading data
	float3 I = ray.O + ray.t * ray.D;
	float3 N = scene.GetNormal( ray.objIdx, I, ray.D );
	float3 albedo = scene.GetAlbedo( ray.objIdx, I );
	// implicit light connection
	if (ray.objIdx == 0) return albedo;
	// do Kajiya
	float3 out_radiance( 0 );
	float reflectivity = scene.GetReflectivity( ray.objIdx, I );
	float refractivity = scene.GetRefractivity( ray.objIdx, I );
	// handle Beer's law
	float3 medium_scale( 1 );
	if (ray.inside)
	{
		float3 absorption = scene.GetAbsorption( ray.objIdx );
		medium_scale = expf( absorption * -ray.t );
	}
	// choose a type of transport
	float r = RandomFloat( seed );
	if (r < reflectivity) // handle pure speculars
	{
		return albedo * medium_scale * HandleMirror( ray, seed, I, N, depth );
	}
	else if (r < reflectivity + refractivity) // handle dielectrics
	{
		return albedo * medium_scale * HandleDielectric( ray, seed, I, N, depth );
	}
	else // diffuse surface
	{
		float3 R = diffusereflection( N, seed );
		float3 brdf = albedo * INVPI;
		Ray r( I + R * EPSILON, R );
		return medium_scale * brdf * 2 * PI * dot( R, N ) * Sample( r, seed, depth + 1 );
	}
}

// -----------------------------------------------------------
// Draw an 16x16 tile of pixels
// -----------------------------------------------------------
void Renderer::ProcessTile( int tx, int ty, float& sum )
{
	float scale = 1.0f / (spp + passes);
	uint seed = InitSeed( tx + ty * SCRWIDTH + spp * 1799 );
	for (int y = ty * 16, v = 0; v < 16; v++, y++) for (int x = tx * 16, u = 0; u < 16; u++, x++)
	{
		for (int p = 0; p < passes; p++)
			accumulator[x + y * SCRWIDTH] +=
			float4( Sample( camera.GetPrimaryRay( (float)x + RandomFloat( seed ),
				(float)y + RandomFloat( seed ) ), seed ), 0 );
		float4 pixel = accumulator[x + y * SCRWIDTH] * scale;
		sum += pixel.x + pixel.y + pixel.z;
		screen->pixels[x + y * SCRWIDTH] = RGBF32_to_RGB8( &pixel );
	}
}
static struct TileJob : public Job
{
	void Main() { renderer->ProcessTile( tx, ty, sum ); }
	Job* Init( Renderer* r, int x, int y ) { renderer = r, tx = x, ty = y, sum = 0; return this; }
	Renderer* renderer;
	int tx, ty;
	float sum;
} tileJob[4096];

// -----------------------------------------------------------
// Main application tick function - Executed once per frame
// -----------------------------------------------------------
void Renderer::Tick( float deltaTime )
{
	// animation
	if (animating) scene.SetTime( anim_time += deltaTime * 0.002f ), ClearAccumulator();
	// pixel loop
	Timer t;
	// render tiles using a simple job system
	for (int jobIdx = 0, y = 0; y < SCRHEIGHT / 16; y++) for (int x = 0; x < SCRWIDTH / 16; x++)
		jm->AddJob2( tileJob[jobIdx++].Init( this, x, y ) );
	jm->RunJobs();
		// gather energy received on tiles
	energy = 0;
	for (int tiles = (SCRWIDTH / 16) * (SCRHEIGHT / 16), i = 0; i < tiles; i++)
		energy += tileJob[i].sum;
	// performance report - running average - ms, MRays/s
	avg = (1 - alpha) * avg + alpha * t.elapsed() * 1000;
	if (alpha > 0.05f) alpha *= 0.75f;
	// handle user input
	if (camera.HandleInput( deltaTime )) ClearAccumulator(); else spp += passes;
}

// -----------------------------------------------------------
// Update user interface (imgui)
// -----------------------------------------------------------
void Renderer::UI()
{
	// animation toggle
	bool changed = ImGui::Checkbox( "Animate scene", &animating );
	// ray query on mouse
	Ray r = camera.GetPrimaryRay( (float)mousePos.x, (float)mousePos.y );
	scene.FindNearest( r );
	ImGui::Text( "Object id %i", r.objIdx );
	ImGui::Text( "spp: %i", spp );
	ImGui::Text( "Energy: %fk", energy / 1000 );
	ImGui::Text( "Frame: %5.2fms (%.1ffps)", avg, 1000 / avg );
	ImGui::Separator();
	changed |= ImGui::SliderInt( "spp", &passes, 1, 4, "%i" );
	// reset accumulator if changes have been made
	if (changed) ClearAccumulator();
}