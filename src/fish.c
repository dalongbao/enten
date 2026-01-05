#include "fish.h"
#include <string.h>

// AOE constants for food detection
#define BASE_AOE 30.0f
#define AOE_GROWTH_RATE 5.0f

// Tail frequency range (very slow for goldfish: 0.3-0.5 Hz)
#define TAIL_FREQ_MIN 0.3f
#define TAIL_FREQ_MAX 0.5f

// Energy cost of movement
#define ENERGY_COST_BASE 0.0001f
#define ENERGY_RECOVERY_RATE 0.0005f

// === Goldfish variety presets ===

FishParams goldfish_common_params(void) {
    return (FishParams){
        .body_length = 120.0f,
        .fin_area_mult = 1.0f,
        .thrust_coeff = 120.0f,
        .drag_coeff = 4.0f,
        .turn_rate = 1.5f,
        .mass = 1.0f
    };
}

FishParams goldfish_comet_params(void) {
    return (FishParams){
        .body_length = 120.0f,
        .fin_area_mult = 0.7f,      // Small fins
        .thrust_coeff = 150.0f,     // More thrust
        .drag_coeff = 3.0f,         // Less drag
        .turn_rate = 2.0f,          // Faster turning
        .mass = 0.8f                // Lighter
    };
}

FishParams goldfish_fancy_params(void) {
    return (FishParams){
        .body_length = 120.0f,
        .fin_area_mult = 1.5f,      // Large flowing fins
        .thrust_coeff = 80.0f,      // Less thrust
        .drag_coeff = 6.0f,         // More drag from fins
        .turn_rate = 1.0f,          // Slower turning
        .mass = 1.2f                // Heavier
    };
}

Fish fish_create(float x, float y, GoldfishVariety variety) {
    FishParams params;
    switch (variety) {
        case GOLDFISH_COMET:
            params = goldfish_comet_params();
            break;
        case GOLDFISH_FANCY:
            params = goldfish_fancy_params();
            break;
        case GOLDFISH_COMMON:
        default:
            params = goldfish_common_params();
            break;
    }

    return (Fish){
        .params = params,
        .variety = variety,
        .state = {
            .pos = {x, y},
            .vel = {0.0f, 0.0f},
            .angle = 0.0f,
            .angular_vel = 0.0f,
            .current_speed = 0.0f,
            .tail = {0.0f, TAIL_FREQ_MIN, 0.0f},
            .body_curve = 0.0f,
            .left_pectoral = 0.0f,
            .right_pectoral = 0.0f,
            .internal = {
                .hunger = 0.6f,         // Start slightly hungry
                .stress = 0.0f,         // Start calm
                .social_comfort = 0.5f, // Neutral
                .energy = 1.0f          // Start rested
            }
        }
    };
}

void fish_update(Fish* fish, float speed, float direction, float urgency, float dt, int screen_w, int screen_h) {
    FishState* s = &fish->state;
    const FishParams* p = &fish->params;

    // Clamp inputs
    speed = fmaxf(0.0f, fminf(1.0f, speed));
    direction = fmaxf(-1.0f, fminf(1.0f, direction));
    urgency = fmaxf(0.0f, fminf(1.0f, urgency));

    // === TAIL ANIMATION (auto-computed from speed & urgency) ===
    // Frequency: 0.3-0.5 Hz based on urgency
    float tail_freq = TAIL_FREQ_MIN + urgency * (TAIL_FREQ_MAX - TAIL_FREQ_MIN);
    s->tail.frequency = tail_freq;
    s->tail.amplitude = speed * 0.8f;
    s->tail.phase += tail_freq * 2.0f * M_PI * dt;
    if (s->tail.phase > 2.0f * M_PI) {
        s->tail.phase -= 2.0f * M_PI;
    }

    // === BODY CURVE (auto-computed from direction) ===
    // Smooth transition to target curve
    float target_curve = direction * 0.4f;
    s->body_curve += (target_curve - s->body_curve) * 5.0f * dt;

    // === PECTORAL FINS (auto-computed from direction & speed) ===
    // When turning, outer fin extends, inner fin tucks
    // When braking (low speed), both fins extend forward
    float base_pec = 0.3f - speed * 0.2f;  // More extended when slow
    s->left_pectoral = base_pec + direction * 0.3f;
    s->right_pectoral = base_pec - direction * 0.3f;

    // === ROTATION ===
    // Turn rate scales with urgency (faster response when urgent)
    float effective_turn_rate = p->turn_rate * (0.5f + urgency * 0.5f);
    s->angular_vel = direction * effective_turn_rate;
    s->angle += s->angular_vel * dt;

    // === THRUST ===
    // Thrust scales with fin area, speed intent, and urgency
    // Pulsed based on tail phase for natural movement
    float tail_pulse = sinf(s->tail.phase);
    tail_pulse = tail_pulse * tail_pulse;  // sin^2 for smooth pulses

    // Effective thrust: base + urgency boost, modulated by tail pulse
    float thrust_factor = 0.3f + urgency * 0.7f;  // 30% base, up to 100%
    float thrust = speed * p->thrust_coeff * p->fin_area_mult * thrust_factor;
    thrust *= (0.5f + tail_pulse * 0.5f);  // Pulse modulation

    float cos_a = cosf(s->angle);
    float sin_a = sinf(s->angle);

    // Thrust in heading direction (slightly affected by body curve)
    float thrust_angle = s->angle - s->body_curve * 0.2f;
    float fx = thrust * cosf(thrust_angle);
    float fy = thrust * sinf(thrust_angle);

    // === DRAG ===
    // Drag scales with fin area (big fins = more drag)
    float current_speed = sqrtf(s->vel.x * s->vel.x + s->vel.y * s->vel.y);
    s->current_speed = current_speed;

    float effective_drag = p->drag_coeff * p->fin_area_mult;
    // Extra drag from body curve (turning creates resistance)
    effective_drag += fabsf(s->body_curve) * 2.0f;

    if (current_speed > 0.0001f) {
        float drag_force = effective_drag * current_speed;
        fx -= drag_force * (s->vel.x / current_speed);
        fy -= drag_force * (s->vel.y / current_speed);
    }

    // === INTEGRATION ===
    float ax = fx / p->mass;
    float ay = fy / p->mass;

    s->vel.x += ax * dt;
    s->vel.y += ay * dt;
    s->pos.x += s->vel.x * dt;
    s->pos.y += s->vel.y * dt;

    // === INTERNAL STATE UPDATES ===
    // Energy: depleted by movement, recovered by resting
    float movement_intensity = speed * (0.3f + urgency * 0.7f);
    s->internal.energy -= ENERGY_COST_BASE * movement_intensity;
    s->internal.energy += ENERGY_RECOVERY_RATE * (1.0f - speed) * dt;
    s->internal.energy = fmaxf(0.0f, fminf(1.0f, s->internal.energy));

    // Hunger: slowly decays (handled by simulator for food eating)
    // Stress: slowly decays when calm (handled by simulator for threats)
    s->internal.stress *= (1.0f - 0.1f * dt);  // Slow decay

    // === WRAP AROUND ===
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
    float sensor_spacing = p->body_length / (num_sensors / 2 + 1);

    for (int i = 0; i < num_sensors; i++) {
        // Determine sensor position in fish-local coords
        int side = (i < num_sensors / 2) ? -1 : 1;  // -1 = left, 1 = right
        int idx = i % (num_sensors / 2);

        // Position along body (front to back)
        float local_x = p->body_length / 2 - (idx + 1) * sensor_spacing;
        float local_y = side * (p->body_length / 6);  // offset from centerline

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

// Helper to clamp value to [-1, 1]
static inline float clamp_unit(float x) {
    if (x < -1.0f) return -1.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

void fish_get_proprioception(
    const Fish* fish,
    float* output
) {
    const FishState* s = &fish->state;
    const FishParams* p = &fish->params;

    // Transform velocity to fish-local coordinates
    float cos_a = cosf(s->angle);
    float sin_a = sinf(s->angle);

    // Forward velocity (in direction of heading), normalized by 100
    float vel_forward = (s->vel.x * cos_a + s->vel.y * sin_a) / 100.0f;
    output[0] = clamp_unit(vel_forward);

    // Lateral velocity (perpendicular to heading), normalized by 100
    float vel_lateral = (-s->vel.x * sin_a + s->vel.y * cos_a) / 100.0f;
    output[1] = clamp_unit(vel_lateral);

    // Angular velocity, normalized by max turn rate
    float angular_vel_norm = s->angular_vel / p->turn_rate;
    output[2] = clamp_unit(angular_vel_norm);

    // Current speed, normalized by max expected speed (~100 px/s)
    output[3] = clamp_unit(s->current_speed / 100.0f);
}

void fish_get_internal_state(
    const Fish* fish,
    float* output
) {
    const FishInternalState* i = &fish->state.internal;
    output[0] = i->hunger;
    output[1] = i->stress;
    output[2] = i->social_comfort;
    output[3] = i->energy;
}
