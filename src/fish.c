#include "fish.h"
#include <string.h>

// AOE constants for food detection
#define BASE_AOE 30.0f
#define AOE_GROWTH_RATE 5.0f

FishParams fish_default_params(void) {
    return (FishParams){
        .mass = 1.0f,
        .drag_coeff = 2.0f,
        .max_thrust = 500.0f,
        .turn_rate = 3.0f,
        .length = 30.0f
    };
}

Fish fish_create(float x, float y) {
    return (Fish){
        .params = fish_default_params(),
        .state = {
            .pos = {x, y},
            .vel = {0.0f, 0.0f},
            .angle = 0.0f,
            .angular_vel = 0.0f,
            .tail = {0.0f, 2.0f, 0.0f}
        }
    };
}

void fish_update(Fish* fish, float thrust, float turn, float dt, int screen_w, int screen_h) {
    FishState* s = &fish->state;
    const FishParams* p = &fish->params;

    // Update tail animation
    s->tail.amplitude = thrust;
    s->tail.phase += s->tail.frequency * 2.0f * M_PI * dt;
    if (s->tail.phase > 2.0f * M_PI) {
        s->tail.phase -= 2.0f * M_PI;
    }

    // Update rotation
    s->angular_vel = turn * p->turn_rate;
    s->angle += s->angular_vel * dt;

    // Compute thrust force
    float thrust_force = thrust * p->max_thrust;
    float fx = thrust_force * cosf(s->angle);
    float fy = thrust_force * sinf(s->angle);

    // Apply drag
    float speed = sqrtf(s->vel.x * s->vel.x + s->vel.y * s->vel.y);
    if (speed > 0.0001f) {
        float drag = p->drag_coeff * speed;
        fx -= drag * (s->vel.x / speed);
        fy -= drag * (s->vel.y / speed);
    }

    // Apply acceleration
    float ax = fx / p->mass;
    float ay = fy / p->mass;

    s->vel.x += ax * dt;
    s->vel.y += ay * dt;
    s->pos.x += s->vel.x * dt;
    s->pos.y += s->vel.y * dt;

    // Wrap around screen (toroidal)
    if (s->pos.x < 0) s->pos.x += screen_w;
    if (s->pos.x >= screen_w) s->pos.x -= screen_w;
    if (s->pos.y < 0) s->pos.y += screen_h;
    if (s->pos.y >= screen_h) s->pos.y -= screen_h;
}

// Helper: compute AOE radius for food of given age
static inline float food_aoe_radius(float age) {
    return BASE_AOE + age * AOE_GROWTH_RATE;
}

// Helper: ray-circle intersection, returns distance to intersection or -1 if no hit
static float ray_circle_intersect(
    float ray_ox, float ray_oy,      // ray origin
    float ray_dx, float ray_dy,      // ray direction (normalized)
    float cx, float cy, float radius // circle center and radius
) {
    // Vector from ray origin to circle center
    float ocx = cx - ray_ox;
    float ocy = cy - ray_oy;
    
    // Project onto ray direction
    float proj = ocx * ray_dx + ocy * ray_dy;
    
    // Closest point on ray to circle center
    float closest_x = ray_ox + proj * ray_dx;
    float closest_y = ray_oy + proj * ray_dy;
    
    // Distance from closest point to circle center
    float dist_sq = (cx - closest_x) * (cx - closest_x) + 
                    (cy - closest_y) * (cy - closest_y);
    
    if (dist_sq > radius * radius) {
        return -1.0f; // No intersection
    }
    
    // Distance from closest point back to intersection
    float half_chord = sqrtf(radius * radius - dist_sq);
    
    // Entry point distance
    float t = proj - half_chord;
    
    // Only count if intersection is in front of ray
    if (t < 0) {
        t = proj + half_chord; // Try exit point
        if (t < 0) return -1.0f;
    }
    
    return t;
}

void fish_cast_rays(
    const Fish* fish,
    const Vec2* food_positions,
    const float* food_ages,
    int food_count,
    int num_rays,
    float arc_radians,
    float max_distance,
    float* output
) {
    const FishState* s = &fish->state;
    
    // Initialize output to max distance, no food
    for (int i = 0; i < num_rays; i++) {
        output[i * 2] = 1.0f;      // normalized distance (max)
        output[i * 2 + 1] = 0.0f;  // food intensity (none)
    }
    
    // Cast each ray
    for (int i = 0; i < num_rays; i++) {
        // Ray angle: spread across arc centered on fish heading
        float t = (num_rays > 1) ? (float)i / (num_rays - 1) : 0.5f;
        float ray_angle = s->angle + (t - 0.5f) * arc_radians;
        
        float ray_dx = cosf(ray_angle);
        float ray_dy = sinf(ray_angle);
        
        float closest_dist = max_distance;
        float closest_intensity = 0.0f;
        
        // Check intersection with each food's AOE
        for (int f = 0; f < food_count; f++) {
            float aoe = food_aoe_radius(food_ages[f]);
            float dist = ray_circle_intersect(
                s->pos.x, s->pos.y,
                ray_dx, ray_dy,
                food_positions[f].x, food_positions[f].y,
                aoe
            );
            
            if (dist > 0 && dist < closest_dist) {
                closest_dist = dist;
                // Intensity based on how deep into AOE (higher = closer to center)
                float dist_to_center = sqrtf(
                    (s->pos.x + ray_dx * dist - food_positions[f].x) * 
                    (s->pos.x + ray_dx * dist - food_positions[f].x) +
                    (s->pos.y + ray_dy * dist - food_positions[f].y) * 
                    (s->pos.y + ray_dy * dist - food_positions[f].y)
                );
                closest_intensity = 1.0f - (dist_to_center / aoe);
                if (closest_intensity < 0) closest_intensity = 0;
            }
        }
        
        output[i * 2] = closest_dist / max_distance;
        output[i * 2 + 1] = closest_intensity;
    }
}

void fish_sense_lateral(
    const Fish* fish,
    const Vec2* food_positions,
    const float* food_ages,
    int food_count,
    int num_sensors,
    float* output
) {
    const FishState* s = &fish->state;
    const FishParams* p = &fish->params;
    
    // Initialize output to zero
    memset(output, 0, num_sensors * 2 * sizeof(float));
    
    // Precompute fish orientation vectors
    float cos_a = cosf(s->angle);
    float sin_a = sinf(s->angle);
    
    // Sensor positions along fish body (in local coords)
    // 4 on left side, 4 on right side
    float sensor_spacing = p->length / (num_sensors / 2 + 1);
    
    for (int i = 0; i < num_sensors; i++) {
        // Determine sensor position in fish-local coords
        int side = (i < num_sensors / 2) ? -1 : 1;  // -1 = left, 1 = right
        int idx = i % (num_sensors / 2);
        
        // Position along body (front to back)
        float local_x = p->length / 2 - (idx + 1) * sensor_spacing;
        float local_y = side * (p->length / 6);  // offset from centerline
        
        // Transform to world coords
        float world_x = s->pos.x + local_x * cos_a - local_y * sin_a;
        float world_y = s->pos.y + local_x * sin_a + local_y * cos_a;
        
        // Accumulate pressure gradient from all food
        float grad_x = 0.0f;
        float grad_y = 0.0f;
        
        for (int f = 0; f < food_count; f++) {
            float dx = food_positions[f].x - world_x;
            float dy = food_positions[f].y - world_y;
            float dist_sq = dx * dx + dy * dy;
            float dist = sqrtf(dist_sq);
            
            float aoe = food_aoe_radius(food_ages[f]);
            
            if (dist < aoe && dist > 0.1f) {
                // Pressure gradient: stronger closer to food
                float strength = (aoe - dist) / aoe;
                strength = strength * strength;  // quadratic falloff
                
                grad_x += strength * (dx / dist);
                grad_y += strength * (dy / dist);
            }
        }
        
        // Transform gradient to fish-local coordinates
        float local_grad_x = grad_x * cos_a + grad_y * sin_a;
        float local_grad_y = -grad_x * sin_a + grad_y * cos_a;
        
        output[i * 2] = local_grad_x;
        output[i * 2 + 1] = local_grad_y;
    }
}

void fish_get_proprioception(
    const Fish* fish,
    float* output
) {
    const FishState* s = &fish->state;
    
    // Transform velocity to fish-local coordinates
    float cos_a = cosf(s->angle);
    float sin_a = sinf(s->angle);
    
    // Forward velocity (in direction of heading)
    output[0] = s->vel.x * cos_a + s->vel.y * sin_a;
    
    // Lateral velocity (perpendicular to heading)
    output[1] = -s->vel.x * sin_a + s->vel.y * cos_a;
    
    // Angular velocity
    output[2] = s->angular_vel;
}
