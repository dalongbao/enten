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
    // Create multiple fish at random positions
    sim->fish_count = DEFAULT_FISH_COUNT;

    for (int i = 0; i < sim->fish_count; i++) {
        // Spread fish around the screen (avoid edges)
        float margin = 100.0f;
        float x = margin + (float)rand() / RAND_MAX * (sim->screen_width - 2 * margin);
        float y = margin + (float)rand() / RAND_MAX * (sim->screen_height - 2 * margin);

        sim->fish[i] = fish_create(x, y, GOLDFISH_COMMON);
        // Randomize initial angle
        sim->fish[i].state.angle = (float)rand() / RAND_MAX * 2.0f * M_PI;
    }

    // Clear food
    sim->food_count = 0;

    // Reset step counter
    sim->step_count = 0;
}

void sim_step(Simulation* sim, const float actions[][3], int* eaten_out) {
    // Update all fish with their respective actions
    for (int i = 0; i < sim->fish_count; i++) {
        float speed = actions[i][0];
        float direction = actions[i][1];
        float urgency = actions[i][2];

        fish_update(&sim->fish[i], speed, direction, urgency, sim->dt,
                    sim->screen_width, sim->screen_height);
    }

    // Age all food
    for (int i = 0; i < sim->food_count; i++) {
        sim->food[i].age += sim->dt;
    }

    // Check for eaten food
    sim_eat_food(sim, eaten_out);

    // Increment step counter
    sim->step_count++;
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

int sim_eat_food(Simulation* sim, int* eaten_per_fish) {
    int total_eaten = 0;
    float r2 = SIM_EAT_RADIUS * SIM_EAT_RADIUS;

    // Initialize per-fish counts if provided
    if (eaten_per_fish) {
        for (int f = 0; f < sim->fish_count; f++) {
            eaten_per_fish[f] = 0;
        }
    }

    // Check each food item against all fish (first fish to reach it eats it)
    for (int i = 0; i < sim->food_count; i++) {
        bool food_eaten = false;

        for (int f = 0; f < sim->fish_count && !food_eaten; f++) {
            float fx = sim->fish[f].state.pos.x;
            float fy = sim->fish[f].state.pos.y;
            float dx = fx - sim->food[i].pos.x;
            float dy = fy - sim->food[i].pos.y;

            if (dx * dx + dy * dy < r2) {
                // This fish eats the food
                sim->fish[f].state.internal.hunger += HUNGER_EAT_RESTORE;
                if (sim->fish[f].state.internal.hunger > 1.0f) {
                    sim->fish[f].state.internal.hunger = 1.0f;
                }

                if (eaten_per_fish) {
                    eaten_per_fish[f]++;
                }
                total_eaten++;
                food_eaten = true;

                // Remove food by swapping with last
                sim->food[i] = sim->food[--sim->food_count];
                i--;  // Recheck this index
            }
        }
    }

    return total_eaten;
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

void sim_get_obs(const Simulation* sim, int fish_id, float* obs) {
    // Bounds check
    if (fish_id < 0 || fish_id >= sim->fish_count) {
        return;
    }

    const Fish* fish = &sim->fish[fish_id];

    // Extract food positions and ages for perception functions
    Vec2 food_positions[MAX_FOOD];
    float food_ages[MAX_FOOD];

    for (int i = 0; i < sim->food_count; i++) {
        food_positions[i] = sim->food[i].pos;
        food_ages[i] = sim->food[i].age;
    }

    // Raycast observations (32 features)
    fish_cast_rays(
        fish,
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
        fish,
        food_positions,
        food_ages,
        sim->food_count,
        SIM_NUM_LATERAL,
        obs + OBS_RAYCAST_SIZE
    );

    // Proprioception (4 features)
    fish_get_proprioception(
        fish,
        obs + OBS_RAYCAST_SIZE + OBS_LATERAL_SIZE
    );

    // Internal state (4 features: hunger, stress, social_comfort, energy)
    fish_get_internal_state(
        fish,
        obs + OBS_RAYCAST_SIZE + OBS_LATERAL_SIZE + OBS_PROPRIO_SIZE
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
