#ifndef FISH_H
#define FISH_H
#include <math.h>

#define MAX_FOOD 32

typedef struct {
    float x;
    float y;
} Vec2;

// Goldfish variety types
typedef enum {
    GOLDFISH_COMMON,    // Balanced, medium speed
    GOLDFISH_COMET,     // Small fins, fast & agile
    GOLDFISH_FANCY      // Large fins, slow & graceful
} GoldfishVariety;

// Physical parameters - vary by goldfish type
typedef struct {
    float body_length;      // Visual length in pixels
    float fin_area_mult;    // Fin size multiplier (affects thrust & drag)
    float thrust_coeff;     // Base thrust coefficient
    float drag_coeff;       // Base drag coefficient
    float turn_rate;        // Max angular velocity
    float mass;             // Body mass
} FishParams;

// Tail animation state (computed from movement, not directly controlled)
typedef struct {
    float phase;            // Current oscillation phase [0, 2pi]
    float frequency;        // Current beat frequency (Hz)
    float amplitude;        // Current tail swing amplitude
} TailState;

// Internal homeostasis state (for equilibrium reward)
typedef struct {
    float hunger;           // [0, 1] - 0=starving, 1=satiated
    float stress;           // [0, 1] - 0=calm, 1=panicked
    float social_comfort;   // [0, 1] - 0=lonely/crowded, 1=comfortable
    float energy;           // [0, 1] - 0=exhausted, 1=rested
} FishInternalState;

// Dynamic state
typedef struct {
    Vec2 pos;
    Vec2 vel;
    float angle;
    float angular_vel;
    float current_speed;        // Actual speed (magnitude of velocity)
    TailState tail;
    // Animated fin positions (computed from movement)
    float body_curve;           // Body curvature for rendering [-1, 1]
    float left_pectoral;        // Left fin angle for rendering
    float right_pectoral;       // Right fin angle for rendering
    // Internal state
    FishInternalState internal;
} FishState;

typedef struct {
    FishParams params;
    FishState state;
    GoldfishVariety variety;
} Fish;

// === Goldfish variety presets ===
FishParams goldfish_common_params(void);   // Balanced
FishParams goldfish_comet_params(void);    // Fast & agile
FishParams goldfish_fancy_params(void);    // Slow & graceful

// === Core functions ===
Fish fish_create(float x, float y, GoldfishVariety variety);

// New hybrid physics: model outputs speed/direction/urgency
// Fin animation is computed automatically from movement
void fish_update(
    Fish* fish,
    float speed,        // [0, 1] - desired forward speed
    float direction,    // [-1, 1] - turn rate (-1=left, +1=right)
    float urgency,      // [0, 1] - movement intensity (affects frequency)
    float dt,
    int screen_w,
    int screen_h
);

// === Perception functions ===
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

// Get proprioceptive features: [forward_vel, lateral_vel, angular_vel, speed]
void fish_get_proprioception(
    const Fish* fish,
    float* output  // size: 4
);

// Get internal state features: [hunger, stress, social_comfort, energy]
void fish_get_internal_state(
    const Fish* fish,
    float* output  // size: 4
);

#endif
