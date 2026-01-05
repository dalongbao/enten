#ifndef FISH_H
#define FISH_H
#include <math.h>

#define MAX_FOOD 32

typedef struct {
    float x;
    float y;
} Vec2;

typedef struct {
    float mass;
    float drag_coeff;
    float max_thrust;
    float turn_rate;
    float length;
} FishParams;

typedef struct {
    float phase;
    float frequency;
    float amplitude;
} TailState;

typedef struct {
    Vec2 pos;
    Vec2 vel;
    float angle;
    float angular_vel;
    TailState tail;
} FishState;

typedef struct {
    FishParams params;
    FishState state;
} Fish;

// Core physics functions (implemented in fish.c)
FishParams fish_default_params(void);
Fish fish_create(float x, float y);
void fish_update(Fish* fish, float thrust, float turn, float dt, int screen_w, int screen_h);

// Perception functions (implemented in fish.c)
// Cast rays to detect food AOE, output is [distance, intensity] pairs
void fish_cast_rays(
    const Fish* fish,
    const Vec2* food_positions,
    const float* food_ages,
    int food_count,
    int num_rays,
    float arc_radians,
    float max_distance,
    float* output  // size: num_rays * 2
);

// Compute lateral line pressure gradients from nearby food
void fish_sense_lateral(
    const Fish* fish,
    const Vec2* food_positions,
    const float* food_ages,
    int food_count,
    int num_sensors,
    float* output  // size: num_sensors * 2
);

// Get proprioceptive features: [forward_vel, lateral_vel, angular_vel]
void fish_get_proprioception(
    const Fish* fish,
    float* output  // size: 3
);

#endif
