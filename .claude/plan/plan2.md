# Goldfish Physics & Equilibrium System

## Overview

Major redesign of fish simulation:
1. **New physics model**: Hybrid control (speed + direction + urgency), fin-animated rendering
2. **Thrust scales with fin area**: Different goldfish varieties have distinct swimming patterns
3. **Multi-factor equilibrium**: Fish maintains homeostasis (hunger, stress, social comfort)
4. **External event integration**: Real-world data affects environment, fish behavior emerges

---

## TODO List

### Phase 1: Physics Redesign
- [x] **1.1** Redesign `FishParams` struct for fin-based thrust scaling - DONE
- [x] **1.2** Implement new hybrid physics in `fish.c` (speed/direction/urgency → fin animation) - DONE
- [x] **1.3** Update simulator.h/c and main.c for new 3-action API - DONE
- [x] **1.4** Update renderer.js/shell.html for 3-action API - DONE (Agent-2)
- [x] **1.5** Update visualize.py for auto fin animation - DONE (Agent-2)
- [x] **1.6** Implement goldfish variety presets (fancy, comet, common) - DONE (in fish.c)
- [x] **1.7** Update fish_env.py for new 3-output action space - DONE
- [ ] **1.8** Update vec_fish_env.py for new 3-output action space - [IN PROGRESS by Agent-1]
- [ ] **1.9** Update policy network for 3 outputs

### Phase 2: Equilibrium System
- [ ] **2.1** Add internal state struct (hunger, stress, social_comfort) - DONE (in fish.h)
- [x] **2.2** Implement equilibrium reward function in fish_env.py - DONE (already implemented)
- [x] **2.3** Implement equilibrium in vec_fish_env.py - DONE (already implemented)
- [ ] **2.4** Add environmental modifiers (external event hooks)

### Phase 3: Multi-Fish (deferred)
- [ ] **3.1** Multi-fish support (after physics is solid)

---

## 1. New Physics Model

### Control Scheme: Hybrid

**Model outputs (3 values):**
```
speed:    [0, 1]     - desired forward speed (0=stop, 1=max)
direction: [-1, 1]   - turn rate (-1=left, +1=right)
urgency:  [0, 1]     - movement intensity (0=lazy drift, 1=burst)
```

**Renderer auto-animates:**
- Tail frequency: `0.3 + urgency * 0.2` Hz (0.3-0.5 Hz range)
- Tail amplitude: `speed * 0.8`
- Body curve: `direction * 0.3`
- Pectoral angle: based on turning + braking

### Physics Equations

```c
// Tail beat frequency (very slow for goldfish)
float freq = 0.3f + urgency * 0.2f;  // 0.3-0.5 Hz
tail_phase += freq * 2π * dt;

// Thrust scales with fin area AND urgency
float thrust = speed * fin_thrust_coeff * (0.3f + urgency * 0.7f);

// Drag also scales with fin area (big fins = more drag)
float drag = base_drag + fin_drag_coeff * speed²;

// Momentum-based movement
velocity += (thrust * heading - drag * velocity) * dt / mass;
position += velocity * dt;

// Turning (body curve + pectoral differential)
angular_vel = direction * turn_rate * (0.5f + urgency * 0.5f);
angle += angular_vel * dt;
```

### Goldfish Variety Presets

| Variety | Fin Area | Thrust Coeff | Drag Coeff | Max Speed | Character |
|---------|----------|--------------|------------|-----------|-----------|
| Fancy (Oranda, Ryukin) | 1.5x | 80 | 6.0 | Slow | Graceful, floaty |
| Common | 1.0x | 120 | 4.0 | Medium | Balanced |
| Comet | 0.7x | 150 | 3.0 | Fast | Agile, zippy |

```c
typedef struct {
    float body_length;      // 80px base
    float fin_area_mult;    // 0.7 - 1.5
    float thrust_coeff;     // scales with fin area
    float drag_coeff;       // scales with fin area
    float turn_rate;
    float mass;
} FishParams;

FishParams goldfish_fancy(void) {
    return (FishParams){
        .body_length = 80.0f,
        .fin_area_mult = 1.5f,
        .thrust_coeff = 80.0f,
        .drag_coeff = 6.0f,
        .turn_rate = 1.0f,
        .mass = 1.2f
    };
}
```

---

## 2. Multi-Factor Equilibrium

### Internal States

```c
typedef struct {
    float hunger;          // [0, 1] - 0=starving, 1=satiated
    float stress;          // [0, 1] - 0=calm, 1=panicked
    float social_comfort;  // [0, 1] - 0=lonely/crowded, 1=comfortable
    float energy;          // [0, 1] - 0=exhausted, 1=rested
} FishState;
```

### State Dynamics

```python
# Hunger: decays over time, restored by eating
hunger -= hunger_decay_rate * dt
hunger += food_restore on eating
hunger = clamp(0, 1)

# Stress: affected by external events + proximity to threats
stress += stress_from_environment * dt
stress -= calm_rate * dt when safe
stress = clamp(0, 1)

# Social comfort: optimal with 1-2 nearby fish, not too close/far
if num_nearby_fish == 0:
    social_comfort -= loneliness_rate * dt
elif nearest_fish_dist < personal_space:
    social_comfort -= crowding_rate * dt
else:
    social_comfort += comfort_rate * dt

# Energy: depleted by movement, restored by resting
energy -= movement_cost * speed * urgency * dt
energy += rest_rate * (1 - speed) * dt
```

### Equilibrium Reward

```python
def compute_reward(state):
    # Target: all states near 0.5-0.7 (comfortable middle)
    hunger_reward = -abs(state.hunger - 0.6) * 0.1
    stress_reward = -state.stress * 0.2  # lower is better
    social_reward = -abs(state.social_comfort - 0.6) * 0.1
    energy_reward = -abs(state.energy - 0.5) * 0.05

    # Survival bonus: not starving or exhausted
    survival = 0.0
    if state.hunger < 0.1:
        survival -= 1.0  # starving penalty
    if state.energy < 0.1:
        survival -= 0.5  # exhausted penalty

    return hunger_reward + stress_reward + social_reward + energy_reward + survival
```

### Behavioral Emergence

With this reward structure, fish should naturally:
- **Meander** when comfortable (low urgency, exploring)
- **Burst** when hungry and food detected (high urgency)
- **Rest** when energy low (near-zero speed)
- **School** for social comfort (stay near but not too close to others)
- **Flee** when stressed (high urgency away from threat)

---

## 3. External Event Integration

### Event → Environment Mapping

```python
class EnvironmentModifier:
    def __init__(self):
        self.food_abundance = 1.0    # multiplier on food spawn rate
        self.threat_level = 0.0      # affects stress
        self.water_quality = 1.0     # affects energy recovery

    def apply_stock_price(self, price_change_pct):
        # Positive = abundance, Negative = scarcity
        self.food_abundance = 1.0 + price_change_pct * 0.5

    def apply_news_sentiment(self, sentiment):  # -1 to +1
        # Negative news = stress, Positive = calm
        self.threat_level = max(0, -sentiment * 0.5)
        self.water_quality = 0.8 + sentiment * 0.2
```

### Usage (Future)

```python
# In main loop
modifier.apply_stock_price(get_sp500_change())
modifier.apply_news_sentiment(get_news_sentiment())
env.set_modifier(modifier)
```

---

## 4. Observation Space

**New observation (per fish):**
```
Raycasts:        32 features (food detection)
Lateral line:    16 features (nearby movement)
Proprioception:   4 features (vel_forward, vel_lateral, angular_vel, speed)
Internal state:   4 features (hunger, stress, social_comfort, energy)
Other fish:       4 features (nearest_dist, nearest_angle, num_nearby, avg_heading_diff)
─────────────────────────────────────────────────────────────────────────
Total:           60 features
```

**New action (per fish):**
```
speed:     [0, 1]
direction: [-1, 1]
urgency:   [0, 1]
─────────────────
Total:     3 values
```

---

## 5. Files to Modify

### Phase 1: Physics
- `src/fish.h` - New FishParams, FishInternalState structs
- `src/fish.c` - New physics (hybrid control, fin-scaled thrust)
- `src/simulator.h` - Add internal state to simulation
- `src/simulator.c` - Update step for new physics
- `src/main.c` - Update WASM exports
- `src/renderer.js` - Auto fin animation from movement
- `training/envs/fish_env.py` - New physics, 3D action
- `training/envs/vec_fish_env.py` - Same
- `training/visualize.py` - Same fin animation
- `training/models/policy.py` - 3 outputs instead of 4

### Phase 2: Equilibrium
- `src/fish.h` - Internal state struct
- `src/simulator.c` - State dynamics
- `training/envs/fish_env.py` - Equilibrium reward, new obs
- `training/envs/vec_fish_env.py` - Same
- `training/configs/default.yaml` - Equilibrium params

---

## Current Status

**Last updated**: Phase 1 almost complete (1.8, 1.9 remaining)

**Active agents**:
- Agent-1: Updating vec_fish_env.py (task 1.8)
- Agent-2: Available

**Completed**:
- fish.h: New FishParams, FishInternalState, GoldfishVariety enum
- fish.c: Hybrid physics (speed/direction/urgency), goldfish presets, auto-fin animation
- simulator.h/c: Updated for new 3-action API
- main.c: Updated WASM exports for 3-action API
- shell.html: Updated to 3-action API (manual controls + brain inference)
- visualize.py: Auto fin animation from speed/direction/urgency
- fish_env.py: Updated for 3-action space
- WASM builds successfully

**Next up** (unclaimed):
- 1.9: Update policy network for 3 outputs
- 2.2-2.4: Equilibrium reward system

**Key decisions made**:
- Hybrid control: speed + direction + urgency (3 outputs)
- Thrust ∝ fin area for variety-specific swimming
- 0.3-0.5 Hz tail frequency (slow, graceful)
- Multi-factor equilibrium (hunger, stress, social, energy)
- External events via environment modifiers
