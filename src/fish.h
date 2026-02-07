#ifndef FISH_H
#define FISH_H
#include <math.h>

#define MAX_FOOD 2048
#define MAX_PARTICLES_PER_FISH 384
#define NUM_BODY_SEGMENTS 3
#define NUM_TAIL_RAYS 3
#define NUM_TAIL_JOINTS 6  // joints per ray (flexible tail)
#define NUM_FIN_RAYS 2     // rays for pectoral/dorsal fins (2 layers)
#define NUM_FIN_JOINTS 4   // joints per ray for pectoral/dorsal fins

typedef struct {
    float x;
    float y;
} Vec2;

typedef enum {
    GOLDFISH_COMMON,
    GOLDFISH_COMET,
    GOLDFISH_FANCY
} GoldfishVariety;

typedef struct {
    float body_length;
    float fin_area_mult;
    float thrust_coeff;
    float drag_coeff;
    float turn_rate;
    float mass;
} FishParams;

typedef struct {
    float phase;
    float frequency;
    float amplitude;
} TailState;

typedef struct {
    float hunger;
    float stress;
    float social_comfort;
    float energy;
} FishInternalState;

typedef struct {
    float body_freq;      // Body wave frequency (0-1) -> maps to 0.5-4.0 Hz
    float body_amp;       // Body wave amplitude (0-1)
    float left_pec_freq;  // Left pectoral frequency (0-1)
    float left_pec_amp;   // Left pectoral amplitude (0-1)
    float right_pec_freq; // Right pectoral frequency (0-1)
    float right_pec_amp;  // Right pectoral amplitude (0-1)
} FinAction;

// Particle for trail effects (brushstroke style)
typedef struct {
    Vec2 pos;
    Vec2 prev_pos;   // Previous position for brushstroke trail
    Vec2 vel;
    float life;      // 0-1, decreases over time
    float size;      // Width of brushstroke
    float length;    // Length of brushstroke
    float alpha;     // Transparency
    float angle;     // Orientation angle
} Particle;

typedef struct {
    Particle particles[MAX_PARTICLES_PER_FISH];
    int count;
    float emit_timer;
} ParticleSystem;

// Body: 3 segments (trapezium -> rectangle -> triangle)
typedef struct {
    float lengths[NUM_BODY_SEGMENTS];      // 4:3:2 ratio
    float widths[NUM_BODY_SEGMENTS + 1];   // width at each joint (4 points for 3 segments)
    Vec2 points[NUM_BODY_SEGMENTS + 1];    // position of each joint
} BodyState;

// Tail: 3 rays forming 2 fin segments, each ray has multiple joints
typedef struct {
    float lengths[NUM_TAIL_RAYS];          // total length: top=70, mid=60, bottom=80
    float amplitudes[NUM_TAIL_RAYS];       // wave amplitude per ray (bottom > top)
    Vec2 base;                             // attachment point (same for all rays)
    Vec2 joints[NUM_TAIL_RAYS][NUM_TAIL_JOINTS];  // joint positions per ray
} TailFinState;

// Pectoral fins (one on each side) - 2 rays forming 1 fin segment
typedef struct {
    float lengths[NUM_FIN_RAYS];      // length per ray
    float amplitudes[NUM_FIN_RAYS];   // wave amplitude per ray
    Vec2 base;                         // attachment point on body centerline
    Vec2 joints[NUM_FIN_RAYS][NUM_FIN_JOINTS];  // joint positions per ray
    float pressure;                    // pressure from turning (bends the fin)
} PectoralFin;

// Dorsal fin (on top/back of fish) - 2 rays forming 1 fin segment
typedef struct {
    float lengths[NUM_FIN_RAYS];      // length per ray
    float amplitudes[NUM_FIN_RAYS];   // wave amplitude per ray
    Vec2 base;                         // attachment point on body centerline
    Vec2 joints[NUM_FIN_RAYS][NUM_FIN_JOINTS];  // joint positions per ray
} DorsalFin;

typedef struct {
    Vec2 pos;
    Vec2 vel;
    float angle;
    float angular_vel;
    float current_speed;
    TailState tail;
    float body_curve;
    float left_pectoral;
    float right_pectoral;
    FishInternalState internal;
    BodyState body;
    TailFinState tail_fin;
    PectoralFin pectoral_left;
    PectoralFin pectoral_right;
    ParticleSystem particles;
} FishState;

typedef struct {
    FishParams params;
    FishState state;
    GoldfishVariety variety;
} Fish;

FishParams goldfish_common_params(void);
FishParams goldfish_comet_params(void);
FishParams goldfish_fancy_params(void);

Fish fish_create(float x, float y, GoldfishVariety variety);
void fish_compute_body(Fish* fish);

void fish_update(
    Fish* fish,
    float speed,
    float direction,
    float urgency,
    float dt,
    int screen_w,
    int screen_h
);

void fish_update_fin(
    Fish* fish,
    const FinAction* action,
    float dt,
    int screen_w,
    int screen_h
);

void fish_emit_particles(Fish* fish, float dt);
void fish_update_particles(Fish* fish, float dt);

void fish_cast_rays(
    const Fish* fish,
    const Vec2* food_positions,
    const float* food_ages,
    int food_count,
    int num_rays,
    float arc_radians,
    float max_distance,
    float* output
);

void fish_sense_lateral(
    const Fish* fish,
    const Vec2* food_positions,
    const float* food_ages,
    int food_count,
    int num_sensors,
    float* output
);

void fish_get_proprioception(
    const Fish* fish,
    float* output
);

void fish_get_internal_state(
    const Fish* fish,
    float* output
);

#endif
