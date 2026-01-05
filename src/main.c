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
// Action inputs (4 controls)
static float gTailMag = 0.5f;      // Tail magnitude [0, 1]
static float gBodyCurve = 0.0f;    // Body curve [-1, 1]
static float gLeftPec = 0.0f;      // Left pectoral [-1, 1]
static float gRightPec = 0.0f;     // Right pectoral [-1, 1]

// Forward declarations
static EM_BOOL on_resize(int type, const EmscriptenUiEvent *e, void *data);
static void main_loop(void);

// --- WASM Exports ---

EMSCRIPTEN_KEEPALIVE
void set_action(float tail_mag, float body_curve, float left_pec, float right_pec) {
    gTailMag = tail_mag;
    gBodyCurve = body_curve;
    gLeftPec = left_pec;
    gRightPec = right_pec;
}

EMSCRIPTEN_KEEPALIVE
float get_x(void) { return gSim.fish.state.pos.x; }

EMSCRIPTEN_KEEPALIVE
float get_y(void) { return gSim.fish.state.pos.y; }

EMSCRIPTEN_KEEPALIVE
float get_angle(void) { return gSim.fish.state.angle; }

EMSCRIPTEN_KEEPALIVE
float get_vx(void) { return gSim.fish.state.vel.x; }

EMSCRIPTEN_KEEPALIVE
float get_vy(void) { return gSim.fish.state.vel.y; }

EMSCRIPTEN_KEEPALIVE
int get_food_count(void) { return gSim.food_count; }

EMSCRIPTEN_KEEPALIVE
void get_observation(float* obs) {
    sim_get_obs(&gSim, obs);
}

EMSCRIPTEN_KEEPALIVE
void drop_food(int x, int y) {
    sim_add_food(&gSim, (float)x, (float)y);
}

// --- Fish state exports for JS rendering ---

EMSCRIPTEN_KEEPALIVE
float get_tail_phase(void) { return gSim.fish.state.tail.phase; }

EMSCRIPTEN_KEEPALIVE
float get_tail_amplitude(void) { return gSim.fish.state.tail.amplitude; }

EMSCRIPTEN_KEEPALIVE
float get_body_curve(void) { return gSim.fish.state.body_curve; }

EMSCRIPTEN_KEEPALIVE
float get_left_pectoral(void) { return gSim.fish.state.left_pectoral; }

EMSCRIPTEN_KEEPALIVE
float get_right_pectoral(void) { return gSim.fish.state.right_pectoral; }

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

    // Clamp fish position
    if (gSim.fish.state.pos.x >= gScreenWidth)
        gSim.fish.state.pos.x = gScreenWidth - 1;
    if (gSim.fish.state.pos.y >= gScreenHeight)
        gSim.fish.state.pos.y = gScreenHeight - 1;

    return EM_TRUE;
}

// --- Main Loop ---

static void main_loop(void) {
    // Step simulation (rendering is handled by JavaScript)
    sim_step(&gSim, gTailMag, gBodyCurve, gLeftPec, gRightPec);
}

// --- Initialization ---

static bool init(void) {
    // Set up resize callback and trigger initial resize
    emscripten_set_resize_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, NULL, EM_FALSE, on_resize);
    on_resize(0, NULL, NULL);

    // Initialize simulation
    sim_init(&gSim, gScreenWidth, gScreenHeight, FRAME_DT);

    printf("Init: screen %dx%d, fish at %.1f,%.1f\n",
           gScreenWidth, gScreenHeight,
           gSim.fish.state.pos.x, gSim.fish.state.pos.y);

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
