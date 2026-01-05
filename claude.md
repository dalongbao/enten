# enten - digital goldfish

C/WebAssembly fish simulation using Emscripten with RL-trained brain.

## Build Commands
- `cd src && make` - Build WASM output to src/build/
- `cd src && make clean` - Remove build artifacts

## Run locally
- `cd src/build && python3 -m http.server 8000` - Serve at localhost:8000

## Training Workflow

### 1. Train new model
```bash
# Full training (10M steps, runs overnight)
PYTHONUNBUFFERED=1 nohup python3 -m training.train > training.log 2>&1 &

# Monitor progress
tail -f training.log

# Quick test run
python3 -m training.train --num-envs 16 --total-timesteps 100000
```

### 2. Export to ONNX
```bash
python3 -c "
from training.models.export import export_to_onnx
export_to_onnx('checkpoints/policy_final.pt', 'inference/model.onnx')
"
```

### 3. Rebuild WASM
```bash
cd src && make clean && make
```

### 4. Deploy
```bash
cp inference/model.onnx src/build/
cd src/build && python3 -m http.server 8000
```

## Project structure
- src/main.c - Entry point, WASM exports
- src/fish.c - Fish physics and perception
- src/fish.h - Fish structs and declarations
- src/simulator.c - Environment state and logic
- src/simulator.h - Simulation constants and structs
- src/shell.html - HTML template with brain.js integration
- src/renderer.js - WebGL fish renderer
- training/ - Python RL training code
  - training/envs/ - Gymnasium environments (FishEnv, VecFishEnv)
  - training/models/ - Policy network and export
  - training/train.py - PPO training script
  - training/configs/ - Training configuration
- inference/ - ONNX model output
- checkpoints/ - Training checkpoints

## Multi-fish Environment
- Default: 3 fish per environment
- Shared policy (same network controls all fish independently)
- Observation: 60 features per fish
  - Raycasts: 32 (16 rays x 2)
  - Lateral line: 16 (8 sensors x 2)
  - Proprioception: 4 (vel_forward, vel_lateral, angular_vel, speed)
  - Internal: 4 (hunger, stress, social_comfort, energy)
  - Social: 4 (nearest_dist, nearest_angle, num_nearby, heading_diff)
- Action: 3 outputs per fish (speed, direction, urgency)
- Schooling rewards: cohesion, separation, alignment

## WASM exports
- set_action(speed, direction, urgency) - Control fish (3 actions, hybrid system)
- get_x(), get_y(), get_angle() - Fish position/orientation
- get_vx(), get_vy() - Fish velocity
- get_food_count() - Number of food items
- get_observation(ptr) - Get 60-feature observation vector
- get_tail_phase(), get_tail_amplitude() - Tail animation state
- get_body_curve(), get_left_pectoral(), get_right_pectoral() - Fin state

## Git Workflow
- Create feature branch for each task: `git checkout -b feature/<task-name>`
- Commit with descriptive messages when feature is complete
- Plans go in `.claude/plans/`
