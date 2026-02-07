#include "fish.h"
#include <stdlib.h>
#include <string.h>

#define BASE_AOE 30.0f
#define AOE_GROWTH_RATE 5.0f
#define TAIL_FREQ_MIN 0.3f
#define TAIL_FREQ_MAX 0.5f
#define ENERGY_COST_BASE 0.0001f
#define ENERGY_RECOVERY_RATE 0.0005f

// Fin-based physics constants
#define FIN_BODY_FREQ_MIN 0.5f
#define FIN_BODY_FREQ_MAX 4.0f
#define FIN_PEC_FREQ_MIN 0.0f
#define FIN_PEC_FREQ_MAX 3.0f
#define FIN_PEC_LEVER_ARM 30.0f

// Particle constants (continuous trail)
#define PARTICLE_LIFETIME 5.0f
#define PARTICLE_SIZE_MIN 2.0f
#define PARTICLE_SIZE_MAX 4.0f
#define PARTICLE_LENGTH_MIN 8.0f
#define PARTICLE_LENGTH_MAX 15.0f
#define PARTICLE_ALPHA_INITIAL 0.7f

// Collision avoidance constants
#define COLLISION_RADIUS 80.0f
#define SEPARATION_FORCE 150.0f

static float randf(float min, float max) {
  return min + (float)rand() / (float)RAND_MAX * (max - min);
}

FishParams goldfish_common_params(void) {
  return (FishParams){.body_length = 180.0f,
                      .fin_area_mult = 1.0f,
                      .thrust_coeff = 120.0f,
                      .drag_coeff = 4.0f,
                      .turn_rate = 1.5f,
                      .mass = 1.0f};
}

FishParams goldfish_comet_params(void) {
  return (FishParams){.body_length = 180.0f,
                      .fin_area_mult = 0.7f,
                      .thrust_coeff = 150.0f,
                      .drag_coeff = 3.0f,
                      .turn_rate = 2.0f,
                      .mass = 0.8f};
}

FishParams goldfish_fancy_params(void) {
  return (FishParams){.body_length = 180.0f,
                      .fin_area_mult = 1.5f,
                      .thrust_coeff = 80.0f,
                      .drag_coeff = 6.0f,
                      .turn_rate = 1.0f,
                      .mass = 1.2f};
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

  // All dimensions as ratios of body_length for unified scaling
  float s = params.body_length; // base scale

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
          .internal = {.hunger = 0.6f,
                       .stress = 0.0f,
                       .social_comfort = 0.5f,
                       .energy = 1.0f},
          // Body segments: 4:3:2 length ratio, widths as fraction of
          // body_length
          .body = {.lengths = {s * 0.44f, s * 0.33f, s * 0.22f},
                   .widths = {s * 0.33f, s * 0.55f, s * 0.44f, 0.0f}},
          // Tail fin: lengths and amplitudes as fractions of body_length
          .tail_fin = {.lengths = {s * 0.78f, s * 0.56f, s * 0.72f},
                       .amplitudes = {s * 0.67f, s * 0.67f, s * 0.89f}},
          // Pectoral fins: lengths and amplitudes as fractions of body_length
          .pectoral_left = {.lengths = {s * 0.33f, s * 0.38f},
                            .amplitudes = {s * 0.13f, s * 0.13f},
                            .pressure = 0.0f},
          .pectoral_right = {.lengths = {s * 0.33f, s * 0.38f},
                             .amplitudes = {s * 0.13f, s * 0.13f},
                             .pressure = 0.0f},
          .particles = {.count = 0, .emit_timer = 0.0f}}};
}

void fish_update(Fish *fish, float speed, float direction, float urgency,
                 float dt, int screen_w, int screen_h) {
  FishState *s = &fish->state;
  const FishParams *p = &fish->params;

  speed = fmaxf(0.0f, fminf(1.0f, speed));
  direction = fmaxf(-1.0f, fminf(1.0f, direction));
  urgency = fmaxf(0.0f, fminf(1.0f, urgency));

  float tail_freq = TAIL_FREQ_MIN + urgency * (TAIL_FREQ_MAX - TAIL_FREQ_MIN);
  s->tail.frequency = tail_freq;
  s->tail.amplitude = speed * 0.8f;
  s->tail.phase += tail_freq * 2.0f * M_PI * dt;
  if (s->tail.phase > 2.0f * M_PI) {
    s->tail.phase -= 2.0f * M_PI;
  }

  float target_curve = direction * 0.4f;
  s->body_curve += (target_curve - s->body_curve) * 5.0f * dt;

  float base_pec = 0.3f - speed * 0.2f;
  s->left_pectoral = base_pec + direction * 0.3f;
  s->right_pectoral = base_pec - direction * 0.3f;

  float effective_turn_rate = p->turn_rate * (0.5f + urgency * 0.5f);
  s->angular_vel = direction * effective_turn_rate;
  s->angle += s->angular_vel * dt;

  float tail_pulse = sinf(s->tail.phase);
  tail_pulse = tail_pulse * tail_pulse;

  float thrust_factor = 0.3f + urgency * 0.7f;
  float thrust = speed * p->thrust_coeff * p->fin_area_mult * thrust_factor;
  thrust *= (0.5f + tail_pulse * 0.5f);

  float cos_a = cosf(s->angle);
  float sin_a = sinf(s->angle);

  float thrust_angle = s->angle - s->body_curve * 0.2f;
  float fx = thrust * cosf(thrust_angle);
  float fy = thrust * sinf(thrust_angle);

  float current_speed = sqrtf(s->vel.x * s->vel.x + s->vel.y * s->vel.y);
  s->current_speed = current_speed;

  float v_forward = s->vel.x * cos_a + s->vel.y * sin_a;
  float v_lateral = -s->vel.x * sin_a + s->vel.y * cos_a;

  float base_drag = p->drag_coeff * p->fin_area_mult * 0.01f;
  float lateral_mult = 10.0f;
  float curve_drag = fabsf(s->body_curve) * 0.5f;

  float drag_forward = (base_drag + curve_drag) * v_forward * fabsf(v_forward);
  float drag_lateral =
      (base_drag * lateral_mult) * v_lateral * fabsf(v_lateral);

  float drag_x = drag_forward * cos_a - drag_lateral * sin_a;
  float drag_y = drag_forward * sin_a + drag_lateral * cos_a;

  fx -= drag_x;
  fy -= drag_y;

  float effective_mass = p->mass * 1.3f;

  float ax = fx / effective_mass;
  float ay = fy / effective_mass;

  s->vel.x += ax * dt;
  s->vel.y += ay * dt;
  s->pos.x += s->vel.x * dt;
  s->pos.y += s->vel.y * dt;

  float movement_intensity = speed * (0.3f + urgency * 0.7f);
  s->internal.energy -= ENERGY_COST_BASE * movement_intensity;
  s->internal.energy += ENERGY_RECOVERY_RATE * (1.0f - speed) * dt;
  s->internal.energy = fmaxf(0.0f, fminf(1.0f, s->internal.energy));

  s->internal.stress *= (1.0f - 0.1f * dt);

  (void)screen_w;
  (void)screen_h;
}

void fish_compute_body(Fish *fish) {
  FishState *s = &fish->state;
  BodyState *b = &s->body;
  TailFinState *tf = &s->tail_fin;

  float cos_a = cosf(s->angle);
  float sin_a = sinf(s->angle);

  float total_len = b->lengths[0] + b->lengths[1] + b->lengths[2];
  float head_offset = total_len * 0.4f;

  b->points[0].x = s->pos.x + cos_a * head_offset;
  b->points[0].y = s->pos.y + sin_a * head_offset;

  float cumulative = 0.0f;
  for (int i = 0; i < NUM_BODY_SEGMENTS; i++) {
    cumulative += b->lengths[i];
    float t = cumulative / total_len;

    float curve_offset = s->body_curve * t * 20.0f;
    float wave_offset =
        sinf(s->tail.phase - t * 2.0f) * s->tail.amplitude * t * 15.0f;

    float local_x = head_offset - cumulative;
    float local_y = curve_offset + wave_offset;

    b->points[i + 1].x = s->pos.x + cos_a * local_x - sin_a * local_y;
    b->points[i + 1].y = s->pos.y + sin_a * local_x + cos_a * local_y;
  }

  // Pectoral fins anchored on sides of body rectangle (segment 1)
  // Calculate body direction and perpendicular at segment 1
  float body_dx = b->points[2].x - b->points[1].x;
  float body_dy = b->points[2].y - b->points[1].y;
  float body_len = sqrtf(body_dx * body_dx + body_dy * body_dy);
  if (body_len < 0.001f)
    body_len = 0.001f;

  // Normal perpendicular to body (pointing left/right from body center)
  float nx = -body_dy / body_len;
  float ny = body_dx / body_len;

  // Body width at segment 1 (the rectangle, widest part)
  float half_width = b->widths[1] / 2.0f;

  // Anchor points on sides of the body
  Vec2 left_base = {b->points[1].x + nx * half_width,
                    b->points[1].y + ny * half_width};
  Vec2 right_base = {b->points[1].x - nx * half_width,
                     b->points[1].y - ny * half_width};

  s->pectoral_left.base = left_base;
  s->pectoral_right.base = right_base;

  // Pressure from angular velocity affects fin bend
  float turn_pressure = s->angular_vel * 0.5f;
  s->pectoral_left.pressure +=
      (turn_pressure - s->pectoral_left.pressure) * 0.1f;
  s->pectoral_right.pressure +=
      (-turn_pressure - s->pectoral_right.pressure) * 0.1f;

  // Left pectoral fin - rays extend outward from left side
  // Base angle points outward, angled slightly forward toward head
  float forward_angle = 1.1f; // ~6° forward
  float left_out_angle = atan2f(ny, nx) - forward_angle;

  for (int ray = 0; ray < NUM_FIN_RAYS; ray++) {
    // Small base offset, waves can cause overlap
    float ray_offset = (ray == 0) ? -0.15f : 0.15f;
    float ray_len = s->pectoral_left.lengths[ray];
    float ray_amp = s->pectoral_left.amplitudes[ray];

    for (int j = 0; j < NUM_FIN_JOINTS; j++) {
      float t = (float)j / (NUM_FIN_JOINTS - 1);
      float dist = t * ray_len;

      // Wave on all joints except base, can cause overlap
      float wave = 0.0f;
      if (j >= 1) {
        float wave_phase = s->tail.phase - t * 1.5f - ray * 0.8f;
        wave = sinf(wave_phase) * ray_amp * 0.03f * t;
      }
      float pressure_bend = s->pectoral_left.pressure * t * 0.3f;

      float angle = left_out_angle + ray_offset + wave + pressure_bend;
      s->pectoral_left.joints[ray][j].x = left_base.x + cosf(angle) * dist;
      s->pectoral_left.joints[ray][j].y = left_base.y + sinf(angle) * dist;
    }
  }

  // Right pectoral fin - rays extend outward from right side, angled forward
  float right_out_angle = atan2f(-ny, -nx) + forward_angle;

  for (int ray = 0; ray < NUM_FIN_RAYS; ray++) {
    // Small base offset, waves can cause overlap
    float ray_offset = (ray == 0) ? 0.15f : -0.15f;
    float ray_len = s->pectoral_right.lengths[ray];
    float ray_amp = s->pectoral_right.amplitudes[ray];

    for (int j = 0; j < NUM_FIN_JOINTS; j++) {
      float t = (float)j / (NUM_FIN_JOINTS - 1);
      float dist = t * ray_len;

      float wave = 0.0f;
      if (j >= 1) {
        float wave_phase = s->tail.phase - t * 1.5f - ray * 0.8f;
        wave = sinf(wave_phase) * ray_amp * 0.03f * t;
      }
      float pressure_bend = s->pectoral_right.pressure * t * 0.3f;

      float angle = right_out_angle + ray_offset - wave + pressure_bend;
      s->pectoral_right.joints[ray][j].x = right_base.x + cosf(angle) * dist;
      s->pectoral_right.joints[ray][j].y = right_base.y + sinf(angle) * dist;
    }
  }

  tf->base = b->points[NUM_BODY_SEGMENTS];
  float tail_base_angle = atan2f(
      b->points[NUM_BODY_SEGMENTS].y - b->points[NUM_BODY_SEGMENTS - 1].y,
      b->points[NUM_BODY_SEGMENTS].x - b->points[NUM_BODY_SEGMENTS - 1].x);

  for (int ray = 0; ray < NUM_TAIL_RAYS; ray++) {
    float ray_len = tf->lengths[ray];
    float ray_amp = tf->amplitudes[ray];
    float ray_phase_offset = ray * 1.5f;

    for (int j = 0; j < NUM_TAIL_JOINTS; j++) {
      float t = (float)j / (NUM_TAIL_JOINTS - 1);

      // Rigidity ramps quadratically, extra rigid at base
      float rigidity = t * t;

      float dist = t * ray_len;
      float wave_phase = s->tail.phase - t * 2.0f - ray_phase_offset;
      float wave = sinf(wave_phase) * ray_amp * rigidity * s->tail.amplitude;

      float local_x = dist;
      float local_y = wave;
      float cos_t = cosf(tail_base_angle);
      float sin_t = sinf(tail_base_angle);
      tf->joints[ray][j].x = tf->base.x + cos_t * local_x - sin_t * local_y;
      tf->joints[ray][j].y = tf->base.y + sin_t * local_x + cos_t * local_y;
    }
  }
}

static inline float food_aoe_radius(float age) {
  return BASE_AOE + age * AOE_GROWTH_RATE;
}

static float ray_circle_intersect(float ray_ox, float ray_oy, float ray_dx,
                                  float ray_dy, float cx, float cy,
                                  float radius) {
  float ocx = cx - ray_ox;
  float ocy = cy - ray_oy;
  float proj = ocx * ray_dx + ocy * ray_dy;
  float closest_x = ray_ox + proj * ray_dx;
  float closest_y = ray_oy + proj * ray_dy;
  float dist_sq =
      (cx - closest_x) * (cx - closest_x) + (cy - closest_y) * (cy - closest_y);
  if (dist_sq > radius * radius)
    return -1.0f;
  float half_chord = sqrtf(radius * radius - dist_sq);
  float t = proj - half_chord;
  if (t < 0) {
    t = proj + half_chord;
    if (t < 0)
      return -1.0f;
  }
  return t;
}

void fish_cast_rays(const Fish *fish, const Vec2 *food_positions,
                    const float *food_ages, int food_count, int num_rays,
                    float arc_radians, float max_distance, float *output) {
  const FishState *s = &fish->state;

  for (int i = 0; i < num_rays; i++) {
    output[i * 2] = 1.0f;
    output[i * 2 + 1] = 0.0f;
  }

  for (int i = 0; i < num_rays; i++) {
    float t = (num_rays > 1) ? (float)i / (num_rays - 1) : 0.5f;
    float ray_angle = s->angle + (t - 0.5f) * arc_radians;
    float ray_dx = cosf(ray_angle);
    float ray_dy = sinf(ray_angle);
    float closest_dist = max_distance;
    float closest_intensity = 0.0f;

    for (int f = 0; f < food_count; f++) {
      float aoe = food_aoe_radius(food_ages[f]);
      float dist =
          ray_circle_intersect(s->pos.x, s->pos.y, ray_dx, ray_dy,
                               food_positions[f].x, food_positions[f].y, aoe);
      if (dist > 0 && dist < closest_dist) {
        closest_dist = dist;
        float dist_to_center =
            sqrtf((s->pos.x + ray_dx * dist - food_positions[f].x) *
                      (s->pos.x + ray_dx * dist - food_positions[f].x) +
                  (s->pos.y + ray_dy * dist - food_positions[f].y) *
                      (s->pos.y + ray_dy * dist - food_positions[f].y));
        closest_intensity = 1.0f - (dist_to_center / aoe);
        if (closest_intensity < 0)
          closest_intensity = 0;
      }
    }
    output[i * 2] = closest_dist / max_distance;
    output[i * 2 + 1] = closest_intensity;
  }
}

void fish_sense_lateral(const Fish *fish, const Vec2 *food_positions,
                        const float *food_ages, int food_count, int num_sensors,
                        float *output) {
  const FishState *s = &fish->state;
  const FishParams *p = &fish->params;

  memset(output, 0, num_sensors * 2 * sizeof(float));

  float cos_a = cosf(s->angle);
  float sin_a = sinf(s->angle);
  float sensor_spacing = p->body_length / (num_sensors / 2 + 1);

  for (int i = 0; i < num_sensors; i++) {
    int side = (i < num_sensors / 2) ? -1 : 1;
    int idx = i % (num_sensors / 2);

    float local_x = p->body_length / 2 - (idx + 1) * sensor_spacing;
    float local_y = side * (p->body_length / 6);

    float world_x = s->pos.x + local_x * cos_a - local_y * sin_a;
    float world_y = s->pos.y + local_x * sin_a + local_y * cos_a;

    float grad_x = 0.0f;
    float grad_y = 0.0f;

    for (int f = 0; f < food_count; f++) {
      float dx = food_positions[f].x - world_x;
      float dy = food_positions[f].y - world_y;
      float dist_sq = dx * dx + dy * dy;
      float dist = sqrtf(dist_sq);
      float aoe = food_aoe_radius(food_ages[f]);

      if (dist < aoe && dist > 0.1f) {
        float strength = (aoe - dist) / aoe;
        strength = strength * strength;
        grad_x += strength * (dx / dist);
        grad_y += strength * (dy / dist);
      }
    }

    float local_grad_x = grad_x * cos_a + grad_y * sin_a;
    float local_grad_y = -grad_x * sin_a + grad_y * cos_a;

    output[i * 2] = local_grad_x;
    output[i * 2 + 1] = local_grad_y;
  }
}

static inline float clamp_unit(float x) {
  if (x < -1.0f)
    return -1.0f;
  if (x > 1.0f)
    return 1.0f;
  return x;
}

void fish_get_proprioception(const Fish *fish, float *output) {
  const FishState *s = &fish->state;
  const FishParams *p = &fish->params;

  float cos_a = cosf(s->angle);
  float sin_a = sinf(s->angle);

  float vel_forward = (s->vel.x * cos_a + s->vel.y * sin_a) / 100.0f;
  output[0] = clamp_unit(vel_forward);

  float vel_lateral = (-s->vel.x * sin_a + s->vel.y * cos_a) / 100.0f;
  output[1] = clamp_unit(vel_lateral);

  float angular_vel_norm = s->angular_vel / p->turn_rate;
  output[2] = clamp_unit(angular_vel_norm);

  output[3] = clamp_unit(s->current_speed / 100.0f);
}

void fish_get_internal_state(const Fish *fish, float *output) {
  const FishInternalState *i = &fish->state.internal;
  output[0] = i->hunger;
  output[1] = i->stress;
  output[2] = i->social_comfort;
  output[3] = i->energy;
}

void fish_emit_particles(Fish *fish, float dt) {
  FishState *s = &fish->state;
  ParticleSystem *ps = &s->particles;

  // Emission rate based on tail activity and speed
  float speed = sqrtf(s->vel.x * s->vel.x + s->vel.y * s->vel.y);
  float emit_rate =
      s->tail.amplitude * s->tail.frequency * 20.0f + speed * 0.1f;
  if (emit_rate < 0.5f)
    emit_rate = 0.5f;

  ps->emit_timer += dt;
  float interval = 1.0f / emit_rate;

  while (ps->emit_timer >= interval && ps->count < MAX_PARTICLES_PER_FISH) {
    // Emit from tail tip (middle ray, last joint)
    Vec2 tail_tip = s->tail_fin.joints[1][NUM_TAIL_JOINTS - 1];

    Particle *p = &ps->particles[ps->count++];
    p->pos = tail_tip;
    p->prev_pos = tail_tip;

    // Velocity follows fish wake with some randomness
    float wake_angle = s->angle + M_PI + randf(-0.3f, 0.3f);
    float wake_speed = speed * 0.2f + randf(5.0f, 15.0f);
    p->vel.x = cosf(wake_angle) * wake_speed;
    p->vel.y = sinf(wake_angle) * wake_speed;

    p->life = 1.0f;
    p->size = randf(PARTICLE_SIZE_MIN, PARTICLE_SIZE_MAX);
    p->length = randf(PARTICLE_LENGTH_MIN, PARTICLE_LENGTH_MAX);
    p->alpha = PARTICLE_ALPHA_INITIAL;
    p->angle = wake_angle;

    ps->emit_timer -= interval;
  }
}

void fish_update_particles(Fish *fish, float dt) {
  ParticleSystem *ps = &fish->state.particles;
  float decay_rate = 1.0f / PARTICLE_LIFETIME;

  for (int i = 0; i < ps->count;) {
    Particle *p = &ps->particles[i];

    p->life -= decay_rate * dt;

    if (p->life <= 0.0f) {
      // Remove by swap with last
      ps->particles[i] = ps->particles[--ps->count];
      continue;
    }

    // Store previous position for brushstroke trail
    p->prev_pos = p->pos;

    p->pos.x += p->vel.x * dt;
    p->pos.y += p->vel.y * dt;

    // Slow down velocity (water resistance)
    p->vel.x *= 0.98f;
    p->vel.y *= 0.98f;

    float vel_mag = sqrtf(p->vel.x * p->vel.x + p->vel.y * p->vel.y);
    if (vel_mag > 0.1f) {
      p->angle = atan2f(p->vel.y, p->vel.x);
    }

    p->alpha = p->life * p->life * PARTICLE_ALPHA_INITIAL;

    p->length *= 0.995f;

    i++;
  }
}

void fish_update_fin(Fish *fish, const FinAction *action, float dt,
                     int screen_w, int screen_h) {
  FishState *s = &fish->state;
  const FishParams *p = &fish->params;

  float body_freq = fmaxf(0.0f, fminf(1.0f, action->body_freq));
  float body_amp = fmaxf(0.0f, fminf(1.0f, action->body_amp));
  float left_pec_freq = fmaxf(0.0f, fminf(1.0f, action->left_pec_freq));
  float left_pec_amp = fmaxf(0.0f, fminf(1.0f, action->left_pec_amp));
  float right_pec_freq = fmaxf(0.0f, fminf(1.0f, action->right_pec_freq));
  float right_pec_amp = fmaxf(0.0f, fminf(1.0f, action->right_pec_amp));

  float actual_body_freq =
      FIN_BODY_FREQ_MIN + body_freq * (FIN_BODY_FREQ_MAX - FIN_BODY_FREQ_MIN);
  float actual_body_amp = body_amp;

  s->tail.frequency = actual_body_freq;
  s->tail.amplitude = actual_body_amp;
  s->tail.phase += actual_body_freq * 2.0f * M_PI * dt;
  if (s->tail.phase > 2.0f * M_PI) {
    s->tail.phase -= 2.0f * M_PI;
  }

  // === THRUST FROM TAIL ===
  // Thrust comes from body wave: thrust = coeff * amplitude * frequency *
  // fin_area
  float fin_area = p->body_length * p->fin_area_mult * 0.1f;
  float thrust =
      p->thrust_coeff * actual_body_amp * actual_body_freq * fin_area;

  // Modulate by tail phase (peak thrust at mid-stroke)
  float tail_pulse = sinf(s->tail.phase);
  tail_pulse = tail_pulse * tail_pulse;
  thrust *= (0.5f + tail_pulse * 0.5f);

  // === TORQUE FROM PECTORAL FINS ===
  float left_pec_actual_freq =
      FIN_PEC_FREQ_MIN + left_pec_freq * (FIN_PEC_FREQ_MAX - FIN_PEC_FREQ_MIN);
  float right_pec_actual_freq =
      FIN_PEC_FREQ_MIN + right_pec_freq * (FIN_PEC_FREQ_MAX - FIN_PEC_FREQ_MIN);

  float left_force = left_pec_amp * left_pec_actual_freq;
  float right_force = right_pec_amp * right_pec_actual_freq;
  float torque = (right_force - left_force) * FIN_PEC_LEVER_ARM;

  s->angular_vel += torque / p->mass * dt;

  // Angular drag to prevent spinning forever
  float angular_drag = 0.95f;
  s->angular_vel *= powf(angular_drag, dt * 60.0f);

  s->angle += s->angular_vel * dt;

  float target_curve = s->angular_vel * 0.3f;
  s->body_curve += (target_curve - s->body_curve) * 5.0f * dt;

  s->left_pectoral = left_pec_amp;
  s->right_pectoral = right_pec_amp;

  // === VELOCITY UPDATE ===
  float cos_a = cosf(s->angle);
  float sin_a = sinf(s->angle);

  float thrust_angle = s->angle - s->body_curve * 0.2f;
  float fx = thrust * cosf(thrust_angle);
  float fy = thrust * sinf(thrust_angle);

  float current_speed = sqrtf(s->vel.x * s->vel.x + s->vel.y * s->vel.y);
  s->current_speed = current_speed;

  float v_forward = s->vel.x * cos_a + s->vel.y * sin_a;
  float v_lateral = -s->vel.x * sin_a + s->vel.y * cos_a;

  // Drag model (lateral drag much higher than forward)
  float base_drag = p->drag_coeff * p->fin_area_mult * 0.01f;
  float lateral_mult = 10.0f;
  float curve_drag = fabsf(s->body_curve) * 0.5f;

  float drag_forward = (base_drag + curve_drag) * v_forward * fabsf(v_forward);
  float drag_lateral =
      (base_drag * lateral_mult) * v_lateral * fabsf(v_lateral);

  float drag_x = drag_forward * cos_a - drag_lateral * sin_a;
  float drag_y = drag_forward * sin_a + drag_lateral * cos_a;

  fx -= drag_x;
  fy -= drag_y;

  float effective_mass = p->mass * 1.3f;
  float ax = fx / effective_mass;
  float ay = fy / effective_mass;

  s->vel.x += ax * dt;
  s->vel.y += ay * dt;
  s->pos.x += s->vel.x * dt;
  s->pos.y += s->vel.y * dt;

  // === INTERNAL STATE UPDATES ===
  float movement_intensity =
      body_amp * body_freq +
      (left_pec_amp * left_pec_freq + right_pec_amp * right_pec_freq) * 0.5f;
  s->internal.energy -= ENERGY_COST_BASE * movement_intensity;
  s->internal.energy += ENERGY_RECOVERY_RATE * (1.0f - movement_intensity) * dt;
  s->internal.energy = fmaxf(0.0f, fminf(1.0f, s->internal.energy));

  s->internal.stress *= (1.0f - 0.1f * dt);

  (void)screen_w;
  (void)screen_h;
}
