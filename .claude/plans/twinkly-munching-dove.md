# Digital Goldfish RL Training System

## Instructions for Claude Instances

**READ THIS FIRST**: This is a shared plan for building a digital goldfish RL system. Multiple Claude instances may work on this in parallel.

### How to use this document:
1. **Check the TODO list below** - find an unclaimed task (no `[IN PROGRESS by ...]` marker)
2. **Claim your task** - edit this file to mark it `[IN PROGRESS by Instance-X]`
3. **Create a feature branch** - `git checkout -b feature/<task-name>`
4. **Implement the task** - follow the specifications in this document
5. **Mark complete** - change `[ ]` to `[x]` and remove the IN PROGRESS marker
6. **Commit and push** - include branch name in commit message

### Coordination rules:
- Only work on ONE task at a time
- If a task depends on another (marked with `depends:`), wait for it to complete
- If you need to modify a file another instance is working on, coordinate or wait
- When done, check if any dependent tasks are now unblocked

---

## TODO List

### Phase 1: Python Training Environment
- [x] **1.1** Create `training/envs/fish_env.py` - single Gymnasium environment
- [x] **1.2** Create `training/envs/vec_fish_env.py` - vectorized NumPy environment
- [x] **1.3** Create `requirements.txt` with dependencies

### Phase 2: Neural Network & Training
- [x] **2.1** Create `training/models/policy.py` - MLP policy network
- [x] **2.2** Create `training/configs/default.yaml` - training hyperparameters
- [x] **2.3** Create `training/train.py` - PPO training script

### Phase 3: C Refactoring
- [x] **3.1** Create `src/fish.c` - extract physics from fish.h, add perception functions
- [x] **3.2** Create `src/simulator.c` - environment state and logic (depends: 3.1)
- [x] **3.3** Create `src/renderer.c` - extract SDL2 rendering from main.c (depends: 3.1)
- [x] **3.4** Update `src/main.c` - entry point only, use new modules (depends: 3.1, 3.2, 3.3)
- [x] **3.5** Update `src/Makefile` - build new file structure (depends: 3.4)

### Phase 4: Training & Validation
- [x] **4.1** Run extended training (2M steps) - Avg 10.2 food/ep, model at `checkpoints/policy_final.pt`
- [ ] **4.2** Validate physics match between Python and C (depends: 3.4, 4.1)

### Phase 5: Browser Integration (LATER)
- [x] **5.1** Create `training/models/export.py` - ONNX export utility (depends: 4.1)
- [x] **5.2** Create `inference/brain.js` - ONNX.js inference wrapper (depends: 5.1)
- [x] **5.3** Update `src/shell.html` - integrate brain.js (depends: 5.2)

---

## Architecture Overview

### Perception System (51 features total)

**Raycasts (32 features)**: 16 rays in 180° frontal arc
- Each ray: `[normalized_distance, food_intensity]`
- Detects food by hitting its AOE (area of effect)

**Lateral Line (16 features)**: 8 sensors (4 per side of fish body)
- Each sensor: `[pressure_x, pressure_y]` in fish-local coords
- Senses pressure gradients from nearby food AOE

**Proprioception (3 features)**:
- `forward_velocity` - speed in heading direction
- `lateral_velocity` - speed perpendicular to heading
- `angular_velocity` - rotation rate

### Action Space (Continuous)
```
thrust ∈ [0, 1]   # forward acceleration
turn ∈ [-1, 1]    # rotation (-1 = left, +1 = right)
```

### MLP Architecture (~6,500 parameters)
```
Input (51)
├── Visual Encoder: raycast(32) → Linear(32) → ReLU → Linear(16) → ReLU → 16
├── Lateral Encoder: lateral(16) → Linear(16) → ReLU → Linear(8) → ReLU → 8
└── Proprioception: proprio(3) → 3

Combined (27) → Linear(64) → ReLU → Linear(64) → ReLU
├── Action Head → Linear(2) → [thrust_mean, turn_mean]
├── Log Std → 2 learnable parameters
└── Value Head → Linear(1) → state value
```

### Food System
- **Size**: Static visual size (small pellet, ~10px)
- **AOE (Area of Effect)**: Detection radius that grows over time
  - `aoe_radius = base_aoe + age * growth_rate`
  - Default: `30 + age * 5` pixels
  - Older food is easier to detect → encourages exploration

---

## File Structure

```
enten/
├── src/                          # C/WASM
│   ├── fish.h                   # Structs and declarations
│   ├── fish.c                   # Physics + perception (NEW)
│   ├── simulator.c              # Environment logic (NEW)
│   ├── renderer.c               # SDL2 display (NEW)
│   ├── main.c                   # Entry point + WASM exports (REFACTOR)
│   ├── shell.html
│   ├── Makefile                 # (UPDATE)
│   └── build/
│
├── training/                     # Python (ALL NEW)
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── fish_env.py          # Single Gymnasium env
│   │   └── vec_fish_env.py      # Vectorized env
│   ├── models/
│   │   ├── __init__.py
│   │   ├── policy.py            # MLP policy
│   │   └── export.py            # ONNX export
│   ├── configs/
│   │   └── default.yaml
│   └── train.py                 # Training script
│
├── inference/                    # Browser (LATER)
│   ├── model.onnx
│   └── brain.js
│
└── requirements.txt
```

---

## Detailed Specifications

### Task 1.1: fish_env.py

Create a Gymnasium-compatible single fish environment.

**Location**: `training/envs/fish_env.py`

**Physics must match `src/fish.h` exactly**:
- mass = 1.0
- drag_coeff = 2.0
- max_thrust = 500.0
- turn_rate = 3.0
- eat_radius = 25.0

**Raycast implementation**:
```python
# For each of 16 rays spread across 180° arc:
ray_angle = fish_angle + (i / 15 - 0.5) * pi
# Cast ray, find nearest food AOE intersection
# Return [distance/max_distance, aoe_intensity]
```

**Lateral line implementation**:
```python
# 8 sensors: positions along fish body
# For each sensor, compute pressure gradient toward nearby food
# Transform to fish-local coordinates
```

**Reward**: +1.0 for each food eaten

**Episode**: 1000 steps, then truncate

---

### Task 1.2: vec_fish_env.py

Vectorized environment running N fish in parallel using NumPy.

**Key**: All operations must be vectorized (no Python loops over envs)

```python
class VecFishEnv:
    def __init__(self, num_envs=64, ...):
        self.pos = np.zeros((num_envs, 2))    # All fish positions
        self.vel = np.zeros((num_envs, 2))    # All fish velocities
        # ... etc

    def step(self, actions):  # actions: (num_envs, 2)
        # Vectorized physics update
        # Vectorized perception
        # Return (obs, rewards, terminated, truncated, infos)
```

---

### Task 2.1: policy.py

PyTorch MLP with separate encoders for visual and lateral inputs.

```python
class FishPolicy(nn.Module):
    def __init__(self, hidden_dim=64, num_rays=16, num_lateral=8):
        # Visual encoder: 32 → 32 → 16
        # Lateral encoder: 16 → 16 → 8
        # Combined: 27 → hidden_dim → hidden_dim
        # Action head: hidden_dim → 2
        # Value head: hidden_dim → 1

    def forward(self, obs):
        # Returns: action_mean, action_log_std, value

    def get_action(self, obs, deterministic=False):
        # Sample action, apply bounds (sigmoid for thrust, tanh for turn)
```

---

### Task 2.3: train.py

PufferLib PPO training script.

```python
# Key hyperparameters (from default.yaml):
num_envs: 64
lr: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_coef: 0.2
ent_coef: 0.01
total_timesteps: 1_000_000

# Training loop using pufferlib.frameworks.cleanrl.PPO
```

---

### Task 3.1: fish.c

Extract inline functions from fish.h to fish.c, add perception.

**New functions to add**:
```c
// Cast rays and populate output buffer
void fish_cast_rays(
    const Fish* fish,
    const Vec2* food_positions,
    const float* food_ages,
    int food_count,
    int num_rays,
    float arc_radians,
    float max_distance,
    float* output  // size: num_rays * 2
);

// Compute lateral line pressure gradients
void fish_sense_lateral(
    const Fish* fish,
    const Vec2* food_positions,
    const float* food_ages,
    int food_count,
    int num_sensors,
    float* output  // size: num_sensors * 2
);
```

---

## Dependencies

**requirements.txt**:
```
torch>=2.0
numpy>=1.24
gymnasium>=0.29
pufferlib>=0.7
pyyaml
tensorboard
onnx
```

---

## Git Workflow

```bash
# For each task:
git checkout main
git pull
git checkout -b feature/task-X.Y-description

# Work on task...

git add .
git commit -m "feat: implement task X.Y - description"
git push -u origin feature/task-X.Y-description

# After review/merge:
git checkout main
git pull
```

---

## Current Status

**Last updated**: Not started

**Active instances**: None

**Blocked tasks**: None (Phase 1 tasks are ready to start)
