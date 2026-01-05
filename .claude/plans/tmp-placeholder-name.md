# Fish Physics & Multi-Fish Update

## Instructions for Claude Instances

**READ THIS FIRST**: This is a shared plan for updating the fish simulation.
Multiple Claude instances may work on this in parallel.

### How to use this document:
1. **Check the TODO list below** - find an unclaimed task (no `[IN PROGRESS by
...]` marker)
2. **Claim your task** - edit this file to mark it `[IN PROGRESS by Instance-X]`
3. **Create a feature branch** - `git checkout -b feature/<task-name>`
4. **Implement the task** - follow the specifications in this document
5. **Mark complete** - change `[ ]` to `[x]` and remove the IN PROGRESS marker
6. **Commit and push** - include branch name in commit message

### Coordination rules:
- Only work on ONE task at a time
- If a task depends on another (marked with `depends:`), wait for it to complete
- If you need to modify a file another instance is working on, coordinate or
wait
- When done, check if any dependent tasks are now unblocked

---

## TODO List

### Phase 1: Physics & Size Tuning
- [x] **1.1** Update `src/fish.c` physics constants for graceful movement
- [x] **1.2** Update `training/envs/fish_env.py` physics constants (depends:
1.1)
- [x] **1.3** Update `training/envs/vec_fish_env.py` physics constants (depends:
1.1)
- [x] **1.4** Increase fish size in C code (`fish.c`, `simulator.h`)
- [x] **1.5** Increase fish size in Python envs (depends: 1.4)
- [x] **1.6** Update `training/visualize.py` for larger fish rendering

### Phase 2: Multi-Fish Support
- [ ] **2.1** Update `src/simulator.h` - add MAX_FISH, fish array, multi-fish
structs
- [ ] **2.2** Update `src/simulator.c` - multi-fish step/reset/eat logic
(depends: 2.1)
- [ ] **2.3** Update `src/main.c` - WASM exports for multi-fish (depends: 2.2)
- [ ] **2.4** Update `training/envs/fish_env.py` - multi-fish state &
observations (depends: 2.1) [IN PROGRESS]
- [ ] **2.5** Update `training/envs/vec_fish_env.py` - vectorized multi-fish
(depends: 2.4)
- [ ] **2.6** Update `training/visualize.py` - render multiple fish (depends:
2.4)

### Phase 3: Schooling Behavior
- [ ] **3.1** Add schooling rewards to `fish_env.py` (depends: 2.4)
- [ ] **3.2** Add schooling rewards to `vec_fish_env.py` (depends: 2.5)
- [ ] **3.3** Add other-fish observation features (depends: 3.1)
- [ ] **3.4** Update policy network for new observation size (depends: 3.3)

### Phase 4: Training & Validation
- [ ] **4.1** Rebuild WASM and test in browser (depends: 2.3)
- [ ] **4.2** Run overnight training (10M steps) (depends: 3.4)

---

## Overview
Three changes:
1. Tune physics for slower, graceful goldfish movement
2. Increase fish size for better visibility
3. Support multiple fish (3) in simulation with schooling rewards

## 1. Graceful Physics Tuning

### Current → New Values
| Parameter | Current | New | Rationale |
|-----------|---------|-----|-----------|
| `max_tail_thrust` | 400 | 150 | Slower acceleration |
| `drag_coeff` | 2.0 | 4.0 | More water resistance |
| `tail_freq_base` | 2.0 | 1.0 | Slower, lazier tail beats |
| `tail_freq_scale` | 3.0 | 1.5 | Less frequency variation |
| `pectoral_thrust` | 50 | 20 | Gentler fin propulsion |
| `pectoral_brake_drag` | 3.0 | 5.0 | More effective braking |
| `curve_turn_rate` | 2.5 | 1.5 | Slower turning |

### Files to modify
- `src/fish.c`: `fish_default_params()`
- `training/envs/fish_env.py`: Physics constants
- `training/envs/vec_fish_env.py`: Physics constants

## 2. Larger Fish Size

### Current → New
- `fish.length`: 30 → 80 pixels
- `eat_radius`: 25 → 40 pixels (proportional)

### Files to modify
- `src/fish.c`: `fish_default_params()` length
- `src/simulator.h`: `SIM_EAT_RADIUS`
- `training/envs/fish_env.py`: `eat_radius`
- `training/envs/vec_fish_env.py`: `eat_radius`
- `training/visualize.py`: Rendering constants

## 3. Multiple Fish Support

### Architecture Changes

**Simulation struct** (simulator.h):
```c
#define MAX_FISH 3

typedef struct {
 Fish fish[MAX_FISH];
 int fish_count;
 Food food[MAX_FOOD];
 int food_count;
 // ... rest
} Simulation;
```

**Step function**: Loop over all fish, each gets its own action

**Observations**: Each fish needs its own observation:
- Raycasts from its position (32 features)
- Lateral line (16 features)
- Proprioception (3 features)
- Hunger (1 feature)
- **NEW**: Other fish relative positions (2 fish × 2 coords = 4 features)
- Total: 52 → 56 features per fish

**WASM exports**:
- `set_action(fish_id, tail, curve, left, right)`
- `get_x(fish_id)`, `get_y(fish_id)`, etc.
- `get_fish_count()`

### Python Environment Changes

**Single env** (`fish_env.py`):
- `num_fish` parameter
- Observation space: `(num_fish, OBS_DIM)` or flattened
- Action space: `(num_fish, 4)` or flattened
- Each fish has independent state arrays

**Vectorized env** (`vec_fish_env.py`):
- Shape becomes `(num_envs, num_fish, ...)`
- More complex but same principle

### Policy Network Options

**Option A: Shared policy (recommended)**
- Same network controls all fish
- Input: single fish observation
- Output: single fish action
- Fish don't see each other initially

**Option B: Multi-agent**
- Fish observe each other's positions
- More complex, enables schooling
- Can add later

### Schooling Behavior via Rewards

Train natural schooling through environmental pressure (reward shaping):

**Cohesion reward**: Small bonus for staying within "comfort zone" of other fish
```python
# If distance to nearest fish is 50-150px, small reward
if 50 < dist_to_nearest < 150:
 reward += 0.01
```

**Separation penalty**: Discourage collisions
```python
# If too close to another fish, penalty
if dist_to_nearest < 30:
 reward -= 0.05
```

**Alignment bonus** (optional): Reward similar heading to neighbors
```python
# If heading within 45° of nearest fish
if abs(angle_diff) < 0.8:
 reward += 0.005
```

This lets the network learn schooling naturally rather than hard-coding it.

### Future: Variable Fish Sizes = Network Sizes
- Different policy networks with different parameter counts
- Fish visual size proportional to its brain's parameter count
- e.g., 5K params = small fish, 50K params = large fish
- Interesting to see if bigger brains = better behavior

## Files to Modify

### Phase 1: Physics + Size
- `src/fish.c`
- `src/simulator.h`
- `training/envs/fish_env.py`
- `training/envs/vec_fish_env.py`
- `training/visualize.py`

### Phase 2: Multi-Fish
- `src/simulator.h` - Add MAX_FISH, fish array
- `src/simulator.c` - Loop over fish in step/reset
- `src/main.c` - Update WASM exports
- `training/envs/fish_env.py` - Multi-fish state
- `training/envs/vec_fish_env.py` - Multi-fish vectorized
- `training/visualize.py` - Draw multiple fish

## Implementation Order

1. **Physics tuning** (quick)
2. **Fish size increase** (quick)
3. **Rebuild WASM** and test in browser
4. **Multi-fish C code**
5. **Multi-fish Python envs**
6. **Update visualizer**
7. **Overnight training run**

## Training Plan
- After all changes: run `python -m training.train --total-timesteps 10000000`
- ~10M steps overnight should yield good graceful behavior
- Use MPS device on Mac for GPU acceleration

---

## Current Status

**Last updated**: Phase 1 complete

**Active instances**: None

**Blocked tasks**: None (Phase 2 tasks are ready to start)

**Notes**:
- Phase 1 complete: physics tuned for graceful movement, fish size increased to 80px
- Implementation exceeded plan: added goldfish variety system (common/comet/fancy)
- Phase 2 (multi-fish) is now unblocked and ready to start
- Phase 3 requires Phase 2 multi-fish to be complete first
