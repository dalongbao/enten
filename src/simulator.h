#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "fish.h"
#include <stdbool.h>

// Simulation constants
#define SIM_EAT_RADIUS 40.0f  // Increased for larger fish
#define SIM_DEFAULT_WIDTH 800
#define SIM_DEFAULT_HEIGHT 600
#define SIM_MAX_STEPS 1000

// Multi-fish constants
#define MAX_FISH 8
#define DEFAULT_FISH_COUNT 3
#define SIM_NUM_RAYS 16
#define SIM_RAY_ARC M_PI           // 180 degrees
#define SIM_RAY_MAX_DIST 200.0f
#define SIM_NUM_LATERAL 8

// Observation dimensions
#define OBS_RAYCAST_SIZE (SIM_NUM_RAYS * 2)      // 32
#define OBS_LATERAL_SIZE (SIM_NUM_LATERAL * 2)   // 16
#define OBS_PROPRIO_SIZE 4                       // forward_vel, lateral_vel, angular_vel, speed
#define OBS_INTERNAL_SIZE 4                      // hunger, stress, social_comfort, energy
#define OBS_TOTAL_SIZE (OBS_RAYCAST_SIZE + OBS_LATERAL_SIZE + OBS_PROPRIO_SIZE + OBS_INTERNAL_SIZE)  // 56

// Hunger parameters
#define HUNGER_INITIAL 1.0f
#define HUNGER_DECAY_RATE 0.001f
#define HUNGER_EAT_RESTORE 0.3f

// Food item with age tracking
typedef struct {
    Vec2 pos;
    float age;  // seconds since spawned
} Food;

// Simulation state
typedef struct {
    Fish fish[MAX_FISH];
    int fish_count;
    Food food[MAX_FOOD];
    int food_count;
    int screen_width;
    int screen_height;
    int step_count;
    float dt;
} Simulation;

// Initialize simulation with default parameters
void sim_init(Simulation* sim, int width, int height, float dt);

// Reset simulation for new episode
void sim_reset(Simulation* sim);

// Step simulation with per-fish actions
// actions: array of [speed, direction, urgency] for each fish
// eaten_out: optional array to receive per-fish food eaten counts
void sim_step(Simulation* sim, const float actions[][3], int* eaten_out);

// Add food at position
bool sim_add_food(Simulation* sim, float x, float y);

// Check and eat food for all fish, returns total eaten
// eaten_per_fish: optional array to receive per-fish counts
int sim_eat_food(Simulation* sim, int* eaten_per_fish);

// Remove food outside screen bounds
void sim_cull_food(Simulation* sim);

// Get observation vector for specific fish (52 features)
void sim_get_obs(const Simulation* sim, int fish_id, float* obs);

// Check if episode is done (truncated)
bool sim_is_truncated(const Simulation* sim);

// Get food positions and ages (for external use)
void sim_get_food_data(const Simulation* sim, Vec2* positions, float* ages, int* count);

#endif
