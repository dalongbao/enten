#ifndef FISH_H
#define FISH_H
#include <math.h>

#define SCREEN_WIDTH 640
#define SCREEN_HEIGHT 480
#define MAX_FOOD 32

typedef struct {
    float x;
    float y;
} Vec2;

typedef struct {
    float mass;
    float drag_coeff;
    float max_thrust;
    float turn_rate;
    float length;
} FishParams;

typedef struct {
    float phase;
    float frequency;
    float amplitude;
} TailState;

typedef struct {
    Vec2 pos;
    Vec2 vel;
    float angle;
    float angular_vel;
    TailState tail;
} FishState;

typedef struct {
    FishParams params;
    FishState state;
} Fish;

static inline FishParams fish_default_params(void) {
    return (FishParams){
        .mass = 1.0f,
        .drag_coeff = 2.0f,
        .max_thrust = 500.0f,
        .turn_rate = 3.0f,
        .length = 30.0f
    };
}

static inline Fish fish_create(float x, float y) {
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

static inline void fish_update(Fish* fish, float thrust, float turn, float dt, int screen_w, int screen_h) {
    FishState* s = &fish->state;
    const FishParams* p = &fish->params;

    s->tail.amplitude = thrust;
    s->tail.phase += s->tail.frequency * 2.0f * M_PI * dt;
    if (s->tail.phase > 2.0f * M_PI) {
        s->tail.phase -= 2.0f * M_PI;
    }

    s->angular_vel = turn * p->turn_rate;
    s->angle += s->angular_vel * dt;

    float thrust_force = thrust * p->max_thrust;
    float fx = thrust_force * cosf(s->angle);
    float fy = thrust_force * sinf(s->angle);

    float speed = sqrtf(s->vel.x * s->vel.x + s->vel.y * s->vel.y);
    if (speed > 0.0001f) {
        float drag = p->drag_coeff * speed;
        fx -= drag * (s->vel.x / speed);
        fy -= drag * (s->vel.y / speed);
    }

    float ax = fx / p->mass;
    float ay = fy / p->mass;

    s->vel.x += ax * dt;
    s->vel.y += ay * dt;
    s->pos.x += s->vel.x * dt;
    s->pos.y += s->vel.y * dt;

    if (s->pos.x < 0) s->pos.x += screen_w;
    if (s->pos.x >= screen_w) s->pos.x -= screen_w;
    if (s->pos.y < 0) s->pos.y += screen_h;
    if (s->pos.y >= screen_h) s->pos.y -= screen_h;
}

static inline float fish_speed(const Fish* fish) {
    const Vec2* v = &fish->state.vel;
    return sqrtf(v->x * v->x + v->y * v->y);
}

static inline float fish_tail_offset(const Fish* fish) {
    const TailState* t = &fish->state.tail;
    return t->amplitude * sinf(t->phase);
}



#endif
