#include <stdio.h>
#include <stdbool.h>
#include <SDL.h>
#include <emscripten.h>
#include <emscripten/html5.h>
#include "fish.h"

#define MIN_WIDTH 256
#define MIN_HEIGHT 256

SDL_Window *gWindow = NULL;
SDL_Renderer *gRenderer = NULL;
Fish gFish;
int gScreenWidth = 1920;
int gScreenHeight = 1080;
float gThrust = 0.5f;
float gTurn = 0.0f;
Vec2 Food[MAX_FOOD];
int gFoodCount = 0;

void cull_food() {
    for (int i = 0; i < gFoodCount; i++) {
        if (Food[i].x >= gScreenWidth || Food[i].y >= gScreenHeight) {
            Food[i] = Food[gFoodCount - 1];
            gFoodCount--;
            i--;
        }
    }
}

EM_BOOL on_resize(int type, const EmscriptenUiEvent *e, void *data) {
    (void)type; (void)e; (void)data;
    double w = EM_ASM_DOUBLE({ return window.innerWidth; });
    double h = EM_ASM_DOUBLE({ return window.innerHeight; });
    gScreenWidth = (int)w < MIN_WIDTH ? MIN_WIDTH : (int)w;
    gScreenHeight = (int)h < MIN_HEIGHT ? MIN_HEIGHT : (int)h;
    emscripten_set_canvas_element_size("#canvas", gScreenWidth, gScreenHeight);
    SDL_SetWindowSize(gWindow, gScreenWidth, gScreenHeight);
    cull_food();
    if (gFish.state.pos.x >= gScreenWidth) gFish.state.pos.x = gScreenWidth - 1;
    if (gFish.state.pos.y >= gScreenHeight) gFish.state.pos.y = gScreenHeight - 1;
    return EM_TRUE;
}

EMSCRIPTEN_KEEPALIVE
void set_action(float thrust, float turn) {
    gThrust = thrust;
    gTurn = turn;
}

EMSCRIPTEN_KEEPALIVE
float get_x(void) { return gFish.state.pos.x; }

EMSCRIPTEN_KEEPALIVE
float get_y(void) { return gFish.state.pos.y; }

EMSCRIPTEN_KEEPALIVE
float get_angle(void) { return gFish.state.angle; }

EMSCRIPTEN_KEEPALIVE
float get_vx(void) { return gFish.state.vel.x; }

EMSCRIPTEN_KEEPALIVE
float get_vy(void) { return gFish.state.vel.y; }

EMSCRIPTEN_KEEPALIVE
int get_food_count(void) { return gFoodCount; }

void drop_food(int x, int y) {
    if (gFoodCount >= MAX_FOOD) return;
    Food[gFoodCount].x = (float)x;
    Food[gFoodCount].y = (float)y;
    gFoodCount++;
}

void eat_food() {
    float eat_radius = 25.0f;
    for (int i = 0; i < gFoodCount; i++) {
        float dx = gFish.state.pos.x - Food[i].x;
        float dy = gFish.state.pos.y - Food[i].y;
        if (dx*dx + dy*dy < eat_radius*eat_radius) {
            Food[i] = Food[gFoodCount - 1];
            gFoodCount--;
            i--;
        }
    }
}

bool init() {
    bool success = true;
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL failed to init: %s\n", SDL_GetError());
        return false;
    }

    gWindow = SDL_CreateWindow("goldfish", 0, 0, gScreenWidth, gScreenHeight, 0);
    if (gWindow == NULL) {
      printf("Window could not be created: %s\n", SDL_GetError());
        return false;
    }

    gRenderer = SDL_CreateRenderer(gWindow, -1, SDL_RENDERER_ACCELERATED);
    if (gRenderer == NULL) {
        printf("Renderer could not be created :%s\n", SDL_GetError());
        return false;
    }

    emscripten_set_resize_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, NULL, EM_FALSE, on_resize);
    on_resize(0, NULL, NULL);
    gFish = fish_create(gScreenWidth/2, gScreenHeight/2);
    printf("Init: screen %dx%d, fish at %.1f,%.1f\n", gScreenWidth, gScreenHeight, gFish.state.pos.x, gFish.state.pos.y);
    return success;
}

void task_loop() {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) {
            emscripten_cancel_main_loop();
        } else if (e.type == SDL_MOUSEBUTTONDOWN) {
            int x = e.button.x;
            int y = e.button.y;
            drop_food(x, y);
        }
    }

    fish_update(&gFish, gThrust, gTurn, 1.0f/60.0f, gScreenWidth, gScreenHeight);
    eat_food();

    SDL_SetRenderDrawColor(gRenderer, 20, 30, 60, 255);
    SDL_RenderClear(gRenderer);

    SDL_SetRenderDrawColor(gRenderer, 100, 200, 100, 255);
    for (int i = 0; i < gFoodCount; i++) {
        SDL_Rect food_rect = {(int)Food[i].x - 5, (int)Food[i].y - 5, 10, 10};
        SDL_RenderFillRect(gRenderer, &food_rect);
    }

    SDL_SetRenderDrawColor(gRenderer, 255, 160, 50, 255);
    SDL_Rect fish_rect = {
        (int)gFish.state.pos.x - 15,
        (int)gFish.state.pos.y - 8,
        30, 16
    };
    SDL_RenderFillRect(gRenderer, &fish_rect);

    SDL_RenderPresent(gRenderer); 
}

void close() {
    SDL_DestroyRenderer(gRenderer);
    gRenderer = NULL;
    SDL_DestroyWindow(gWindow);
    gWindow = NULL;

    SDL_Quit();
}

int main(int argc, char* args[]) {
    if (!init()) {
        printf("failed to init");
        return 1;
    } 

    emscripten_set_main_loop(task_loop, 0, 1);
    close();
    return 0;
}
