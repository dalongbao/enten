#include "simulator.h"
#include <emscripten.h>
#include <emscripten/html5.h>
#include <stdbool.h>
#include <stdio.h>

#define MIN_WIDTH 256
#define MIN_HEIGHT 256
#define FRAME_DT (1.0f / 60.0f)

static Simulation gSim;
static int gScreenWidth = 1920;
static int gScreenHeight = 1080;

// Legacy action mode
static float gSpeed = 0.5f;
static float gDirection = 0.0f;
static float gUrgency = 0.3f;

// New fin-based action mode
static FinAction gFinAction = {0.5f, 0.5f, 0.5f, 0.3f, 0.5f, 0.3f};
static int gUseFinMode = 0;  // 0 = legacy/boid, 1 = fin-based control


static EM_BOOL on_resize(int type, const EmscriptenUiEvent *e, void *data);
static void main_loop(void);

// Legacy action interface - maps to fin-based internally
EMSCRIPTEN_KEEPALIVE
void set_action(float speed, float direction, float urgency) {
    gSpeed = speed;
    gDirection = direction;
    gUrgency = urgency;
    gUseFinMode = 0;
}

// New fin-based action interface (6 actions)
EMSCRIPTEN_KEEPALIVE
void set_fin_action(float body_freq, float body_amp,
                    float left_pec_freq, float left_pec_amp,
                    float right_pec_freq, float right_pec_amp) {
    gFinAction.body_freq = body_freq;
    gFinAction.body_amp = body_amp;
    gFinAction.left_pec_freq = left_pec_freq;
    gFinAction.left_pec_amp = left_pec_amp;
    gFinAction.right_pec_freq = right_pec_freq;
    gFinAction.right_pec_amp = right_pec_amp;
    gUseFinMode = 1;
}

EMSCRIPTEN_KEEPALIVE
float get_x(void) { return gSim.fish[0].state.pos.x; }

EMSCRIPTEN_KEEPALIVE
float get_y(void) { return gSim.fish[0].state.pos.y; }

EMSCRIPTEN_KEEPALIVE
float get_angle(void) { return gSim.fish[0].state.angle; }

EMSCRIPTEN_KEEPALIVE
int get_food_count(void) { return gSim.food_count; }

EMSCRIPTEN_KEEPALIVE
float get_food_x(int idx) {
    if (idx >= 0 && idx < gSim.food_count)
        return gSim.food[idx].pos.x;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_food_y(int idx) {
    if (idx >= 0 && idx < gSim.food_count)
        return gSim.food[idx].pos.y;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
void spawn_food(float x, float y) {
    sim_add_food(&gSim, x, y);
}

EMSCRIPTEN_KEEPALIVE
int get_fish_count_sim(void) { return gSim.fish_count; }

EMSCRIPTEN_KEEPALIVE
int get_screen_w(void) { return gScreenWidth; }

EMSCRIPTEN_KEEPALIVE
int get_screen_h(void) { return gScreenHeight; }

EMSCRIPTEN_KEEPALIVE
float get_fov_arc(void) { return SIM_RAY_ARC; }

EMSCRIPTEN_KEEPALIVE
float get_fov_range(void) { return SIM_RAY_MAX_DIST; }

EMSCRIPTEN_KEEPALIVE
float get_fish_angle(int fish_id) {
    if (fish_id >= 0 && fish_id < gSim.fish_count)
        return gSim.fish[fish_id].state.angle;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_fish_x(int fish_id) {
    if (fish_id >= 0 && fish_id < gSim.fish_count)
        return gSim.fish[fish_id].state.pos.x;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_fish_y(int fish_id) {
    if (fish_id >= 0 && fish_id < gSim.fish_count)
        return gSim.fish[fish_id].state.pos.y;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
int get_num_body_segments(void) { return NUM_BODY_SEGMENTS; }

EMSCRIPTEN_KEEPALIVE
float get_body_point_x(int fish_id, int point_idx) {
    if (fish_id >= 0 && fish_id < gSim.fish_count && point_idx >= 0 && point_idx <= NUM_BODY_SEGMENTS)
        return gSim.fish[fish_id].state.body.points[point_idx].x;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_body_point_y(int fish_id, int point_idx) {
    if (fish_id >= 0 && fish_id < gSim.fish_count && point_idx >= 0 && point_idx <= NUM_BODY_SEGMENTS)
        return gSim.fish[fish_id].state.body.points[point_idx].y;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_body_width(int fish_id, int point_idx) {
    if (fish_id >= 0 && fish_id < gSim.fish_count && point_idx >= 0 && point_idx <= NUM_BODY_SEGMENTS)
        return gSim.fish[fish_id].state.body.widths[point_idx];
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
int get_num_tail_rays(void) { return NUM_TAIL_RAYS; }

EMSCRIPTEN_KEEPALIVE
int get_num_tail_joints(void) { return NUM_TAIL_JOINTS; }

EMSCRIPTEN_KEEPALIVE
float get_tail_joint_x(int fish_id, int ray_idx, int joint_idx) {
    if (fish_id >= 0 && fish_id < gSim.fish_count &&
        ray_idx >= 0 && ray_idx < NUM_TAIL_RAYS &&
        joint_idx >= 0 && joint_idx < NUM_TAIL_JOINTS)
        return gSim.fish[fish_id].state.tail_fin.joints[ray_idx][joint_idx].x;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_tail_joint_y(int fish_id, int ray_idx, int joint_idx) {
    if (fish_id >= 0 && fish_id < gSim.fish_count &&
        ray_idx >= 0 && ray_idx < NUM_TAIL_RAYS &&
        joint_idx >= 0 && joint_idx < NUM_TAIL_JOINTS)
        return gSim.fish[fish_id].state.tail_fin.joints[ray_idx][joint_idx].y;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
int get_num_fin_rays(void) { return NUM_FIN_RAYS; }

EMSCRIPTEN_KEEPALIVE
int get_num_fin_joints(void) { return NUM_FIN_JOINTS; }

EMSCRIPTEN_KEEPALIVE
float get_pectoral_left_joint_x(int fish_id, int ray_idx, int joint_idx) {
    if (fish_id >= 0 && fish_id < gSim.fish_count &&
        ray_idx >= 0 && ray_idx < NUM_FIN_RAYS &&
        joint_idx >= 0 && joint_idx < NUM_FIN_JOINTS)
        return gSim.fish[fish_id].state.pectoral_left.joints[ray_idx][joint_idx].x;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_pectoral_left_joint_y(int fish_id, int ray_idx, int joint_idx) {
    if (fish_id >= 0 && fish_id < gSim.fish_count &&
        ray_idx >= 0 && ray_idx < NUM_FIN_RAYS &&
        joint_idx >= 0 && joint_idx < NUM_FIN_JOINTS)
        return gSim.fish[fish_id].state.pectoral_left.joints[ray_idx][joint_idx].y;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_pectoral_right_joint_x(int fish_id, int ray_idx, int joint_idx) {
    if (fish_id >= 0 && fish_id < gSim.fish_count &&
        ray_idx >= 0 && ray_idx < NUM_FIN_RAYS &&
        joint_idx >= 0 && joint_idx < NUM_FIN_JOINTS)
        return gSim.fish[fish_id].state.pectoral_right.joints[ray_idx][joint_idx].x;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_pectoral_right_joint_y(int fish_id, int ray_idx, int joint_idx) {
    if (fish_id >= 0 && fish_id < gSim.fish_count &&
        ray_idx >= 0 && ray_idx < NUM_FIN_RAYS &&
        joint_idx >= 0 && joint_idx < NUM_FIN_JOINTS)
        return gSim.fish[fish_id].state.pectoral_right.joints[ray_idx][joint_idx].y;
    return 0.0f;
}

// Particle system accessors
EMSCRIPTEN_KEEPALIVE
int get_particle_count(int fish_id) {
    if (fish_id >= 0 && fish_id < gSim.fish_count)
        return gSim.fish[fish_id].state.particles.count;
    return 0;
}

EMSCRIPTEN_KEEPALIVE
float get_particle_x(int fish_id, int particle_idx) {
    if (fish_id >= 0 && fish_id < gSim.fish_count &&
        particle_idx >= 0 && particle_idx < gSim.fish[fish_id].state.particles.count)
        return gSim.fish[fish_id].state.particles.particles[particle_idx].pos.x;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_particle_y(int fish_id, int particle_idx) {
    if (fish_id >= 0 && fish_id < gSim.fish_count &&
        particle_idx >= 0 && particle_idx < gSim.fish[fish_id].state.particles.count)
        return gSim.fish[fish_id].state.particles.particles[particle_idx].pos.y;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_particle_alpha(int fish_id, int particle_idx) {
    if (fish_id >= 0 && fish_id < gSim.fish_count &&
        particle_idx >= 0 && particle_idx < gSim.fish[fish_id].state.particles.count)
        return gSim.fish[fish_id].state.particles.particles[particle_idx].alpha;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_particle_size(int fish_id, int particle_idx) {
    if (fish_id >= 0 && fish_id < gSim.fish_count &&
        particle_idx >= 0 && particle_idx < gSim.fish[fish_id].state.particles.count)
        return gSim.fish[fish_id].state.particles.particles[particle_idx].size;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_particle_angle(int fish_id, int particle_idx) {
    if (fish_id >= 0 && fish_id < gSim.fish_count &&
        particle_idx >= 0 && particle_idx < gSim.fish[fish_id].state.particles.count)
        return gSim.fish[fish_id].state.particles.particles[particle_idx].angle;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_particle_length(int fish_id, int particle_idx) {
    if (fish_id >= 0 && fish_id < gSim.fish_count &&
        particle_idx >= 0 && particle_idx < gSim.fish[fish_id].state.particles.count)
        return gSim.fish[fish_id].state.particles.particles[particle_idx].length;
    return 0.0f;
}

static EM_BOOL on_resize(int type, const EmscriptenUiEvent *e, void *data) {
    (void)type; (void)e; (void)data;

    double w = EM_ASM_DOUBLE({ return window.innerWidth; });
    double h = EM_ASM_DOUBLE({ return window.innerHeight; });

    gScreenWidth = (int)w < MIN_WIDTH ? MIN_WIDTH : (int)w;
    gScreenHeight = (int)h < MIN_HEIGHT ? MIN_HEIGHT : (int)h;

    emscripten_set_canvas_element_size("#canvas", gScreenWidth, gScreenHeight);

    gSim.screen_width = gScreenWidth;
    gSim.screen_height = gScreenHeight;
    sim_cull_food(&gSim);

    for (int i = 0; i < gSim.fish_count; i++) {
        if (gSim.fish[i].state.pos.x >= gScreenWidth)
            gSim.fish[i].state.pos.x = gScreenWidth - 1;
        if (gSim.fish[i].state.pos.y >= gScreenHeight)
            gSim.fish[i].state.pos.y = gScreenHeight - 1;
    }

    return EM_TRUE;
}

static void main_loop(void) {
    float actions[MAX_FISH][3];
    for (int i = 0; i < gSim.fish_count; i++) {
        actions[i][0] = gSpeed;
        actions[i][1] = gDirection;
        actions[i][2] = gUrgency;
    }
    sim_step(&gSim, actions, NULL);

    for (int i = 0; i < gSim.fish_count; i++) {
        fish_compute_body(&gSim.fish[i]);
        fish_emit_particles(&gSim.fish[i], FRAME_DT);
        fish_update_particles(&gSim.fish[i], FRAME_DT);
    }
}

static bool init(void) {
    emscripten_set_resize_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, NULL, EM_FALSE, on_resize);
    on_resize(0, NULL, NULL);

    sim_init(&gSim, gScreenWidth, gScreenHeight, FRAME_DT);

    for (int i = 0; i < gSim.fish_count; i++) {
        fish_compute_body(&gSim.fish[i]);
    }

    printf("Init: screen %dx%d, %d fish\n", gScreenWidth, gScreenHeight, gSim.fish_count);

    return true;
}

int main(int argc, char *args[]) {
    (void)argc; (void)args;

    if (!init()) {
        printf("Failed to init\n");
        return 1;
    }

    emscripten_set_main_loop(main_loop, 0, 1);
    return 0;
}
