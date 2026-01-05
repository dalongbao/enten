#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "fish.h"
#include <stdbool.h>

// Simulation constants
#define SIM_EAT_RADIUS 25.0f
#define SIM_DEFAULT_WIDTH 800
#define SIM_DEFAULT_HEIGHT 600
#define SIM_MAX_STEPS 1000
#define SIM_NUM_RAYS 16
#define SIM_RAY_ARC M_PI           // 180 degrees
#define SIM_RAY_MAX_DIST 400.0f
#define SIM_NUM_LATERAL 8

// Observation dimensions
#define OBS_RAYCAST_SIZE (SIM_NUM_RAYS * 2)      // 32
#define OBS_LATERAL_SIZE (SIM_NUM_LATERAL * 2)   // 16
#define OBS_PROPRIO_SIZE 3
#define OBS_TOTAL_SIZE (OBS_RAYCAST_SIZE + OBS_LATERAL_SIZE + OBS_PROPRIO_SIZE)  // 51

// Food item with age tracking
typedef struct {
    Vec2 pos;
    float age;  // seconds since spawned
} Food;

// Simulation state
typedef struct {
    Fish fish;
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

// Step simulation with action, returns reward (food eaten count)
int sim_step(Simulation* sim, float thrust, float turn);

// Add food at position
bool sim_add_food(Simulation* sim, float x, float y);

// Check and eat food within radius, returns count eaten
int sim_eat_food(Simulation* sim);

// Remove food outside screen bounds
void sim_cull_food(Simulation* sim);

// Get observation vector (51 features)
void sim_get_obs(const Simulation* sim, float* obs);

// Check if episode is done (truncated)
bool sim_is_truncated(const Simulation* sim);

// Get food positions and ages (for external use)
void sim_get_food_data(const Simulation* sim, Vec2* positions, float* ages, int* count);

#endif
