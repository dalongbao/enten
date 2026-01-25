#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "fish.h"
#include <stdbool.h>

#define SIM_EAT_RADIUS 40.0f
#define SIM_DEFAULT_WIDTH 800
#define SIM_DEFAULT_HEIGHT 600
#define SIM_MAX_STEPS 1000

#define MAX_FISH 8
#define DEFAULT_FISH_COUNT 3
#define SIM_NUM_RAYS 16
#define SIM_RAY_ARC (M_PI * 1.5)   // 270 degrees FOV
#define SIM_RAY_MAX_DIST 450.0f    // Longer sensing range
#define SIM_NUM_LATERAL 8

#define OBS_RAYCAST_SIZE (SIM_NUM_RAYS * 2)
#define OBS_LATERAL_SIZE (SIM_NUM_LATERAL * 2)
#define OBS_PROPRIO_SIZE 4
#define OBS_INTERNAL_SIZE 4
#define OBS_TOTAL_SIZE (OBS_RAYCAST_SIZE + OBS_LATERAL_SIZE + OBS_PROPRIO_SIZE + OBS_INTERNAL_SIZE)

#define HUNGER_INITIAL 1.0f
#define HUNGER_DECAY_RATE 0.001f
#define HUNGER_EAT_RESTORE 0.3f

typedef struct {
    Vec2 pos;
    float age;
} Food;

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

void sim_init(Simulation* sim, int width, int height, float dt);
void sim_reset(Simulation* sim);
void sim_step(Simulation* sim, const float actions[][3], int* eaten_out);
bool sim_add_food(Simulation* sim, float x, float y);
int sim_eat_food(Simulation* sim, int* eaten_per_fish);
void sim_cull_food(Simulation* sim);
void sim_get_obs(const Simulation* sim, int fish_id, float* obs);
bool sim_is_truncated(const Simulation* sim);
void sim_get_food_data(const Simulation* sim, Vec2* positions, float* ages, int* count);

#endif
