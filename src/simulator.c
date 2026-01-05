#include "simulator.h"
#include <string.h>
#include <stdlib.h>

void sim_init(Simulation* sim, int width, int height, float dt) {
    sim->screen_width = width;
    sim->screen_height = height;
    sim->dt = dt;
    sim_reset(sim);
}

void sim_reset(Simulation* sim) {
    // Create fish at center of screen
    sim->fish = fish_create(
        sim->screen_width / 2.0f,
        sim->screen_height / 2.0f
    );
    
    // Clear food
    sim->food_count = 0;
    
    // Reset step counter
    sim->step_count = 0;
}

int sim_step(Simulation* sim, float thrust, float turn) {
    // Update fish physics
    fish_update(&sim->fish, thrust, turn, sim->dt, 
                sim->screen_width, sim->screen_height);
    
    // Age all food
    for (int i = 0; i < sim->food_count; i++) {
        sim->food[i].age += sim->dt;
    }
    
    // Check for eaten food
    int eaten = sim_eat_food(sim);
    
    // Increment step counter
    sim->step_count++;
    
    return eaten;
}

bool sim_add_food(Simulation* sim, float x, float y) {
    if (sim->food_count >= MAX_FOOD) {
        return false;
    }
    
    sim->food[sim->food_count].pos.x = x;
    sim->food[sim->food_count].pos.y = y;
    sim->food[sim->food_count].age = 0.0f;
    sim->food_count++;
    
    return true;
}

int sim_eat_food(Simulation* sim) {
    int eaten = 0;
    float fx = sim->fish.state.pos.x;
    float fy = sim->fish.state.pos.y;
    float r2 = SIM_EAT_RADIUS * SIM_EAT_RADIUS;
    
    for (int i = 0; i < sim->food_count; i++) {
        float dx = fx - sim->food[i].pos.x;
        float dy = fy - sim->food[i].pos.y;
        
        if (dx * dx + dy * dy < r2) {
            // Remove food by swapping with last
            sim->food[i] = sim->food[--sim->food_count];
            i--;  // Recheck this index
            eaten++;
        }
    }
    
    return eaten;
}

void sim_cull_food(Simulation* sim) {
    for (int i = 0; i < sim->food_count; i++) {
        if (sim->food[i].pos.x < 0 || sim->food[i].pos.x >= sim->screen_width ||
            sim->food[i].pos.y < 0 || sim->food[i].pos.y >= sim->screen_height) {
            // Remove food by swapping with last
            sim->food[i] = sim->food[--sim->food_count];
            i--;  // Recheck this index
        }
    }
}

void sim_get_obs(const Simulation* sim, float* obs) {
    // Extract food positions and ages for perception functions
    Vec2 food_positions[MAX_FOOD];
    float food_ages[MAX_FOOD];
    
    for (int i = 0; i < sim->food_count; i++) {
        food_positions[i] = sim->food[i].pos;
        food_ages[i] = sim->food[i].age;
    }
    
    // Raycast observations (32 features)
    fish_cast_rays(
        &sim->fish,
        food_positions,
        food_ages,
        sim->food_count,
        SIM_NUM_RAYS,
        SIM_RAY_ARC,
        SIM_RAY_MAX_DIST,
        obs
    );
    
    // Lateral line observations (16 features)
    fish_sense_lateral(
        &sim->fish,
        food_positions,
        food_ages,
        sim->food_count,
        SIM_NUM_LATERAL,
        obs + OBS_RAYCAST_SIZE
    );
    
    // Proprioception (3 features)
    fish_get_proprioception(
        &sim->fish,
        obs + OBS_RAYCAST_SIZE + OBS_LATERAL_SIZE
    );
}

bool sim_is_truncated(const Simulation* sim) {
    return sim->step_count >= SIM_MAX_STEPS;
}

void sim_get_food_data(const Simulation* sim, Vec2* positions, float* ages, int* count) {
    *count = sim->food_count;
    for (int i = 0; i < sim->food_count; i++) {
        positions[i] = sim->food[i].pos;
        ages[i] = sim->food[i].age;
    }
}
