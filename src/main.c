#include "simulator.h"
#include "renderer.h"
#include <SDL.h>
#include <emscripten.h>
#include <emscripten/html5.h>
#include <stdbool.h>
#include <stdio.h>

#define MIN_WIDTH 256
#define MIN_HEIGHT 256
#define FRAME_DT (1.0f / 60.0f)

// Global state
static Simulation gSim;
static Renderer gRenderer;
static int gScreenWidth = 1920;
static int gScreenHeight = 1080;
static float gThrust = 0.5f;
static float gTurn = 0.0f;

// Forward declarations
static EM_BOOL on_resize(int type, const EmscriptenUiEvent *e, void *data);
static void main_loop(void);

// --- WASM Exports ---

EMSCRIPTEN_KEEPALIVE
void set_action(float thrust, float turn) {
    gThrust = thrust;
    gTurn = turn;
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
void drop_food(int x, int y) {
    sim_add_food(&gSim, (float)x, (float)y);
}

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
    renderer_resize(&gRenderer, gScreenWidth, gScreenHeight);

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
    // Handle events
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) {
            emscripten_cancel_main_loop();
        } else if (e.type == SDL_MOUSEBUTTONDOWN) {
            drop_food(e.button.x, e.button.y);
        }
    }

    // Step simulation
    sim_step(&gSim, gThrust, gTurn);

    // Render
    renderer_clear(&gRenderer);

    // Draw food (extract positions for renderer)
    Vec2 food_positions[MAX_FOOD];
    for (int i = 0; i < gSim.food_count; i++) {
        food_positions[i] = gSim.food[i].pos;
    }
    renderer_draw_all_food(&gRenderer, food_positions, gSim.food_count);

    // Draw fish
    renderer_draw_fish(&gRenderer, &gSim.fish);

    renderer_present(&gRenderer);
}

// --- Initialization ---

static bool init(void) {
    // Initialize renderer
    if (!renderer_init(&gRenderer, "goldfish", gScreenWidth, gScreenHeight)) {
        printf("Failed to initialize renderer\n");
        return false;
    }

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
    renderer_close(&gRenderer);
    return 0;
}
