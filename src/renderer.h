#ifndef RENDERER_H
#define RENDERER_H

#include "fish.h"
#include <SDL.h>
#include <stdbool.h>

// Render constants
#define FOOD_HALF_SIZE 5
#define FISH_HALF_W 15
#define FISH_HALF_H 8

// Colors
#define COLOR_BG_R 20
#define COLOR_BG_G 30
#define COLOR_BG_B 60
#define COLOR_FOOD_R 100
#define COLOR_FOOD_G 200
#define COLOR_FOOD_B 100
#define COLOR_FISH_R 255
#define COLOR_FISH_G 160
#define COLOR_FISH_B 50

// Renderer state
typedef struct {
    SDL_Window* window;
    SDL_Renderer* sdl_renderer;
} Renderer;

// Initialize SDL and create window/renderer
bool renderer_init(Renderer* r, const char* title, int width, int height);

// Resize the renderer
void renderer_resize(Renderer* r, int width, int height);

// Clear screen with background color
void renderer_clear(Renderer* r);

// Draw a single food item
void renderer_draw_food(Renderer* r, const Vec2* pos);

// Draw all food items
void renderer_draw_all_food(Renderer* r, const Vec2* food, int count);

// Draw the fish
void renderer_draw_fish(Renderer* r, const Fish* fish);

// Present the rendered frame
void renderer_present(Renderer* r);

// Cleanup renderer resources
void renderer_close(Renderer* r);

#endif
