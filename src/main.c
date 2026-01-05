#include "simulator.h"
#include <emscripten.h>
#include <emscripten/html5.h>
#include <stdbool.h>
#include <stdio.h>

#define MIN_WIDTH 256
#define MIN_HEIGHT 256
#define FRAME_DT (1.0f / 60.0f)

// Global state
static Simulation gSim;
static int gScreenWidth = 1920;
static int gScreenHeight = 1080;
// Action inputs (3 controls - hybrid system)
static float gSpeed = 0.5f;        // Desired speed [0, 1]
static float gDirection = 0.0f;    // Turn direction [-1, 1]
static float gUrgency = 0.3f;      // Movement urgency [0, 1]

// Forward declarations
static EM_BOOL on_resize(int type, const EmscriptenUiEvent *e, void *data);
static void main_loop(void);

// --- WASM Exports ---

EMSCRIPTEN_KEEPALIVE
void set_action(float speed, float direction, float urgency) {
    gSpeed = speed;
    gDirection = direction;
    gUrgency = urgency;
}

EMSCRIPTEN_KEEPALIVE
float get_x(void) { return gSim.fish[0].state.pos.x; }

EMSCRIPTEN_KEEPALIVE
float get_y(void) { return gSim.fish[0].state.pos.y; }

EMSCRIPTEN_KEEPALIVE
float get_angle(void) { return gSim.fish[0].state.angle; }

EMSCRIPTEN_KEEPALIVE
float get_vx(void) { return gSim.fish[0].state.vel.x; }

EMSCRIPTEN_KEEPALIVE
float get_vy(void) { return gSim.fish[0].state.vel.y; }

EMSCRIPTEN_KEEPALIVE
int get_food_count(void) { return gSim.food_count; }

EMSCRIPTEN_KEEPALIVE
void get_observation(float* obs) {
    sim_get_obs(&gSim, 0, obs);
}

EMSCRIPTEN_KEEPALIVE
void drop_food(int x, int y) {
    sim_add_food(&gSim, (float)x, (float)y);
}

// --- Multi-fish exports ---

EMSCRIPTEN_KEEPALIVE
int get_fish_count_sim(void) { return gSim.fish_count; }

EMSCRIPTEN_KEEPALIVE
void set_action_for_fish(int fish_id, float speed, float direction, float urgency) {
    if (fish_id >= 0 && fish_id < MAX_FISH) {
        // Store in per-fish action array for next step
        // For now, we use global action for all fish controlled via set_action()
        // This function is for future per-fish control from JS
    }
}

EMSCRIPTEN_KEEPALIVE
float get_x_for_fish(int fish_id) {
    if (fish_id >= 0 && fish_id < gSim.fish_count)
        return gSim.fish[fish_id].state.pos.x;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_y_for_fish(int fish_id) {
    if (fish_id >= 0 && fish_id < gSim.fish_count)
        return gSim.fish[fish_id].state.pos.y;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_angle_for_fish(int fish_id) {
    if (fish_id >= 0 && fish_id < gSim.fish_count)
        return gSim.fish[fish_id].state.angle;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
void get_observation_for_fish(int fish_id, float* obs) {
    sim_get_obs(&gSim, fish_id, obs);
}

// --- Fish state exports for JS rendering ---

EMSCRIPTEN_KEEPALIVE
float get_tail_phase(void) { return gSim.fish[0].state.tail.phase; }

EMSCRIPTEN_KEEPALIVE
float get_tail_amplitude(void) { return gSim.fish[0].state.tail.amplitude; }

EMSCRIPTEN_KEEPALIVE
float get_body_curve(void) { return gSim.fish[0].state.body_curve; }

EMSCRIPTEN_KEEPALIVE
float get_left_pectoral(void) { return gSim.fish[0].state.left_pectoral; }

EMSCRIPTEN_KEEPALIVE
float get_right_pectoral(void) { return gSim.fish[0].state.right_pectoral; }

// Per-fish animation state exports
EMSCRIPTEN_KEEPALIVE
float get_tail_phase_for_fish(int fish_id) {
    if (fish_id >= 0 && fish_id < gSim.fish_count)
        return gSim.fish[fish_id].state.tail.phase;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_tail_amplitude_for_fish(int fish_id) {
    if (fish_id >= 0 && fish_id < gSim.fish_count)
        return gSim.fish[fish_id].state.tail.amplitude;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_body_curve_for_fish(int fish_id) {
    if (fish_id >= 0 && fish_id < gSim.fish_count)
        return gSim.fish[fish_id].state.body_curve;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_left_pectoral_for_fish(int fish_id) {
    if (fish_id >= 0 && fish_id < gSim.fish_count)
        return gSim.fish[fish_id].state.left_pectoral;
    return 0.0f;
}

EMSCRIPTEN_KEEPALIVE
float get_right_pectoral_for_fish(int fish_id) {
    if (fish_id >= 0 && fish_id < gSim.fish_count)
        return gSim.fish[fish_id].state.right_pectoral;
    return 0.0f;
}

// --- Food data exports ---

EMSCRIPTEN_KEEPALIVE
void get_food_positions(float* out) {
    for (int i = 0; i < gSim.food_count; i++) {
        out[i * 2] = gSim.food[i].pos.x;
        out[i * 2 + 1] = gSim.food[i].pos.y;
    }
}

// --- Screen dimension exports ---

EMSCRIPTEN_KEEPALIVE
int get_screen_w(void) { return gScreenWidth; }

EMSCRIPTEN_KEEPALIVE
int get_screen_h(void) { return gScreenHeight; }

// --- Event Handlers ---

static EM_BOOL on_resize(int type, const EmscriptenUiEvent *e, void *data) {
    (void)type;
    (void)e;
    (void)data;

    double w = EM_ASM_DOUBLE({ return window.innerWidth; });
    double h = EM_ASM_DOUBLE({ return window.innerHeight; });

    gScreenWidth = (int)w < MIN_WIDTH ? MIN_WIDTH : (int)w;
    gScreenHeight = (int)h < MIN_HEIGHT ? MIN_HEIGHT : (int)h;

    emscripten_set_canvas_element_size("#canvas", gScreenWidth, gScreenHeight);

    // Update simulation dimensions
    gSim.screen_width = gScreenWidth;
    gSim.screen_height = gScreenHeight;
    sim_cull_food(&gSim);

    // Clamp fish positions
    for (int i = 0; i < gSim.fish_count; i++) {
        if (gSim.fish[i].state.pos.x >= gScreenWidth)
            gSim.fish[i].state.pos.x = gScreenWidth - 1;
        if (gSim.fish[i].state.pos.y >= gScreenHeight)
            gSim.fish[i].state.pos.y = gScreenHeight - 1;
    }

    return EM_TRUE;
}

// --- Main Loop ---

static void main_loop(void) {
    // Build actions array for all fish (use same action for all in browser mode)
    float actions[MAX_FISH][3];
    for (int i = 0; i < gSim.fish_count; i++) {
        actions[i][0] = gSpeed;
        actions[i][1] = gDirection;
        actions[i][2] = gUrgency;
    }
    // Step simulation (rendering is handled by JavaScript)
    sim_step(&gSim, actions, NULL);
}

// --- Initialization ---

static bool init(void) {
    // Set up resize callback and trigger initial resize
    emscripten_set_resize_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, NULL, EM_FALSE, on_resize);
    on_resize(0, NULL, NULL);

    // Initialize simulation
    sim_init(&gSim, gScreenWidth, gScreenHeight, FRAME_DT);

    printf("Init: screen %dx%d, %d fish, fish[0] at %.1f,%.1f\n",
           gScreenWidth, gScreenHeight, gSim.fish_count,
           gSim.fish[0].state.pos.x, gSim.fish[0].state.pos.y);

    return true;
}

int main(int argc, char *args[]) {
    (void)argc;
    (void)args;

    if (!init()) {
        printf("Failed to init\n");
        return 1;
    }

    emscripten_set_main_loop(main_loop, 0, 1);
    return 0;
}
