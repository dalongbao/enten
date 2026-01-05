#include "renderer.h"
#include <stdio.h>

bool renderer_init(Renderer* r, const char* title, int width, int height) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL failed to init: %s\n", SDL_GetError());
        return false;
    }

    r->window = SDL_CreateWindow(title, 0, 0, width, height, 0);
    if (r->window == NULL) {
        printf("Window could not be created: %s\n", SDL_GetError());
        return false;
    }

    r->sdl_renderer = SDL_CreateRenderer(r->window, -1, SDL_RENDERER_ACCELERATED);
    if (r->sdl_renderer == NULL) {
        printf("Renderer could not be created: %s\n", SDL_GetError());
        SDL_DestroyWindow(r->window);
        r->window = NULL;
        return false;
    }

    return true;
}

void renderer_resize(Renderer* r, int width, int height) {
    SDL_SetWindowSize(r->window, width, height);
}

void renderer_clear(Renderer* r) {
    SDL_SetRenderDrawColor(r->sdl_renderer, COLOR_BG_R, COLOR_BG_G, COLOR_BG_B, 255);
    SDL_RenderClear(r->sdl_renderer);
}

void renderer_draw_food(Renderer* r, const Vec2* pos) {
    SDL_SetRenderDrawColor(r->sdl_renderer, COLOR_FOOD_R, COLOR_FOOD_G, COLOR_FOOD_B, 255);
    SDL_Rect food_rect = {
        (int)pos->x - FOOD_HALF_SIZE,
        (int)pos->y - FOOD_HALF_SIZE,
        FOOD_HALF_SIZE * 2,
        FOOD_HALF_SIZE * 2
    };
    SDL_RenderFillRect(r->sdl_renderer, &food_rect);
}

void renderer_draw_all_food(Renderer* r, const Vec2* food, int count) {
    SDL_SetRenderDrawColor(r->sdl_renderer, COLOR_FOOD_R, COLOR_FOOD_G, COLOR_FOOD_B, 255);
    for (int i = 0; i < count; i++) {
        SDL_Rect food_rect = {
            (int)food[i].x - FOOD_HALF_SIZE,
            (int)food[i].y - FOOD_HALF_SIZE,
            FOOD_HALF_SIZE * 2,
            FOOD_HALF_SIZE * 2
        };
        SDL_RenderFillRect(r->sdl_renderer, &food_rect);
    }
}

void renderer_draw_fish(Renderer* r, const Fish* fish) {
    SDL_SetRenderDrawColor(r->sdl_renderer, COLOR_FISH_R, COLOR_FISH_G, COLOR_FISH_B, 255);
    SDL_Rect fish_rect = {
        (int)fish->state.pos.x - FISH_HALF_W,
        (int)fish->state.pos.y - FISH_HALF_H,
        FISH_HALF_W * 2,
        FISH_HALF_H * 2
    };
    SDL_RenderFillRect(r->sdl_renderer, &fish_rect);
}

void renderer_present(Renderer* r) {
    SDL_RenderPresent(r->sdl_renderer);
}

void renderer_close(Renderer* r) {
    if (r->sdl_renderer != NULL) {
        SDL_DestroyRenderer(r->sdl_renderer);
        r->sdl_renderer = NULL;
    }
    if (r->window != NULL) {
        SDL_DestroyWindow(r->window);
        r->window = NULL;
    }
    SDL_Quit();
}
