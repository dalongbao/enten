"""
Vectorized multi-fish environment for fast RL training.

All operations are vectorized using NumPy - no Python loops over environments.
This enables training on hundreds of parallel environments efficiently.

Supports N fish (default 3) per environment with:
- Independent physics and control per fish
- Shared food system (fish compete for food)
- Social dynamics (schooling rewards based on proximity)
- Observation includes other fish positions

NEW HYBRID PHYSICS - Model controls high-level movement:
- speed [0, 1]: Desired forward speed
- direction [-1, 1]: Turn rate (-1=left, +1=right)
- urgency [0, 1]: Movement intensity (affects tail frequency)

Fin animation is computed automatically from movement.
Thrust scales with fin area (different goldfish varieties swim differently).

Observation per fish: raycasts (32) + lateral (16) + proprio (4) + internal (4) + social (4) = 60
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class VecFishEnv:
    """Fully vectorized multi-fish environment running N envs with M fish each."""

    # Perception constants (must match FishEnv)
    NUM_RAYS = 16
    RAY_ARC = np.pi  # 180 degrees
    MAX_RAY_LENGTH = 200.0
    NUM_LATERAL_SENSORS = 8
    OBS_DIM = NUM_RAYS * 2 + NUM_LATERAL_SENSORS * 2 + 4 + 4 + 4  # 60

    # Tail frequency range (very slow for goldfish: 0.3-0.5 Hz)
    TAIL_FREQ_MIN = 0.3
    TAIL_FREQ_MAX = 0.5

    def __init__(self, num_envs: int = 64, config: dict = None):
        self.num_envs = num_envs
        self.config = config or {}

        # Number of fish per environment
        self.num_fish = self.config.get("num_fish", 3)

        # Environment dimensions
        self.width = self.config.get("width", 800)
        self.height = self.config.get("height", 600)

        # Spaces (for compatibility) - flattened for all fish
        self.single_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        # 3 outputs per fish - speed, direction, urgency
        self.single_action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Flattened observation/action spaces for all fish
        total_obs_dim = self.num_fish * self.OBS_DIM
        total_action_dim = self.num_fish * 3

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_envs, total_obs_dim), dtype=np.float32
        )
        # Action space per environment (flattened across fish)
        self.action_space = spaces.Box(
            low=np.tile([0.0, -1.0, 0.0], self.num_fish).astype(np.float32),
            high=np.tile([1.0, 1.0, 1.0], self.num_fish).astype(np.float32),
            dtype=np.float32,
        )

        # Per-fish dims for reshaping
        self.single_obs_dim = self.OBS_DIM
        self.single_action_dim = 3

        # Goldfish variety (can be configured)
        variety = self.config.get("variety", "common")
        self._set_variety(variety)

        # Food parameters
        self.max_food = 32
        self.eat_radius = 40.0
        self.food_aoe_base = 30.0
        self.food_aoe_growth = 5.0

        # Episode settings
        self.max_steps = self.config.get("max_steps", 1000)
        self.dt = 1.0 / 60.0

        # Internal state parameters
        self.energy_cost_base = 0.0001
        self.energy_recovery_rate = 0.0005
        self.hunger_decay_rate = 0.001
        self.hunger_eat_restore = 0.3
        self.stress_decay_rate = 0.1

        # Exploration parameters
        exploration_config = self.config.get("exploration", {})
        self.grid_size = exploration_config.get("grid_size", 100)
        self.visit_bonus = exploration_config.get("visit_bonus", 0.05)
        self.distance_bonus = exploration_config.get("distance_bonus", 0.001)

        # Social/schooling parameters
        self.schooling_min_dist = 50.0
        self.schooling_max_dist = 150.0
        self.collision_dist = 30.0
        self.cohesion_reward = 0.01
        self.separation_penalty = 0.05
        self.alignment_reward = 0.005

        # Pre-compute ray angles (relative to heading)
        t = np.linspace(0, 1, self.NUM_RAYS)
        self.ray_angles_rel = (t - 0.5) * self.RAY_ARC  # [-pi/2, pi/2]

        # Lateral sensor offsets (in fish-local coords)
        self.lateral_along = np.array([(i % 4 - 1.5) / 3.0 * 0.8 * self.body_length
                                        for i in range(self.NUM_LATERAL_SENSORS)])
        self.lateral_side = np.array([-1 if i < 4 else 1
                                       for i in range(self.NUM_LATERAL_SENSORS)])
        self.lateral_perp_dist = self.body_length / 4

        # Initialize RNG
        self.rng = np.random.default_rng()

        # State arrays (initialized in reset)
        self._init_state_arrays()

    def _set_variety(self, variety: str):
        """Set physics parameters based on goldfish variety."""
        if variety == "comet":
            self.body_length = 80.0
            self.fin_area_mult = 0.7
            self.thrust_coeff = 150.0
            self.drag_coeff = 3.0
            self.turn_rate = 2.0
            self.mass = 0.8
        elif variety == "fancy":
            self.body_length = 80.0
            self.fin_area_mult = 1.5
            self.thrust_coeff = 80.0
            self.drag_coeff = 6.0
            self.turn_rate = 1.0
            self.mass = 1.2
        else:  # common (default)
            self.body_length = 80.0
            self.fin_area_mult = 1.0
            self.thrust_coeff = 120.0
            self.drag_coeff = 4.0
            self.turn_rate = 1.5
            self.mass = 1.0

    def _init_state_arrays(self):
        """Initialize all state arrays."""
        n = self.num_envs
        m = self.num_fish

        # Fish state: (num_envs, num_fish, ...)
        self.pos = np.zeros((n, m, 2), dtype=np.float32)
        self.vel = np.zeros((n, m, 2), dtype=np.float32)
        self.angle = np.zeros((n, m), dtype=np.float32)
        self.angular_vel = np.zeros((n, m), dtype=np.float32)
        self.current_speed = np.zeros((n, m), dtype=np.float32)
        self.tail_phase = np.zeros((n, m), dtype=np.float32)
        self.body_curve = np.zeros((n, m), dtype=np.float32)
        self.left_pectoral = np.zeros((n, m), dtype=np.float32)
        self.right_pectoral = np.zeros((n, m), dtype=np.float32)

        # Food: (num_envs, max_food, 3) - x, y, age
        self.food = np.zeros((n, self.max_food, 3), dtype=np.float32)
        self.food_count = np.zeros(n, dtype=np.int32)

        self.steps = np.zeros(n, dtype=np.int32)

        # Internal state per fish: (num_envs, num_fish)
        self.hunger = np.ones((n, m), dtype=np.float32) * 0.6
        self.stress = np.zeros((n, m), dtype=np.float32)
        self.social_comfort = np.ones((n, m), dtype=np.float32) * 0.5
        self.energy = np.ones((n, m), dtype=np.float32)

        # Exploration state per fish
        self.visited_cells = [[set() for _ in range(m)] for _ in range(n)]
        self.prev_pos = np.zeros((n, m, 2), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset all environments."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        n = self.num_envs
        m = self.num_fish

        # Initialize fish positions (ensure they don't overlap)
        self.pos = np.zeros((n, m, 2), dtype=np.float32)
        for env_idx in range(n):
            for fish_idx in range(m):
                while True:
                    pos = self.rng.uniform(
                        [self.body_length, self.body_length],
                        [self.width - self.body_length, self.height - self.body_length]
                    )
                    if fish_idx == 0:
                        break
                    dists = np.linalg.norm(self.pos[env_idx, :fish_idx] - pos, axis=1)
                    if dists.min() > self.body_length * 2:
                        break
                self.pos[env_idx, fish_idx] = pos

        self.vel = np.zeros((n, m, 2), dtype=np.float32)
        self.angle = self.rng.uniform(0, 2 * np.pi, size=(n, m)).astype(np.float32)
        self.angular_vel = np.zeros((n, m), dtype=np.float32)
        self.current_speed = np.zeros((n, m), dtype=np.float32)
        self.tail_phase = np.zeros((n, m), dtype=np.float32)
        self.body_curve = np.zeros((n, m), dtype=np.float32)
        self.left_pectoral = np.zeros((n, m), dtype=np.float32)
        self.right_pectoral = np.zeros((n, m), dtype=np.float32)

        self.food = np.zeros((n, self.max_food, 3), dtype=np.float32)
        self.food_count = np.zeros(n, dtype=np.int32)
        self.steps = np.zeros(n, dtype=np.int32)

        # Spawn initial food
        initial_food = self.config.get("initial_food", 5)
        self._spawn_food_all(initial_food)

        # Initialize internal state
        self.hunger = np.ones((n, m), dtype=np.float32) * 0.6
        self.stress = np.zeros((n, m), dtype=np.float32)
        self.social_comfort = np.ones((n, m), dtype=np.float32) * 0.5
        self.energy = np.ones((n, m), dtype=np.float32)

        # Initialize exploration
        self.visited_cells = [[set() for _ in range(m)] for _ in range(n)]
        self.prev_pos = self.pos.copy()

        return self._get_obs(), {}

    def reset_envs(self, env_mask: np.ndarray):
        """Reset only specified environments."""
        indices = np.where(env_mask)[0]
        n_reset = len(indices)

        if n_reset == 0:
            return

        m = self.num_fish

        # Reset fish positions
        for env_idx in indices:
            for fish_idx in range(m):
                while True:
                    pos = self.rng.uniform(
                        [self.body_length, self.body_length],
                        [self.width - self.body_length, self.height - self.body_length]
                    )
                    if fish_idx == 0:
                        break
                    dists = np.linalg.norm(self.pos[env_idx, :fish_idx] - pos, axis=1)
                    if dists.min() > self.body_length * 2:
                        break
                self.pos[env_idx, fish_idx] = pos

        self.vel[indices] = 0.0
        self.angle[indices] = self.rng.uniform(0, 2 * np.pi, size=(n_reset, m)).astype(np.float32)
        self.angular_vel[indices] = 0.0
        self.current_speed[indices] = 0.0
        self.tail_phase[indices] = 0.0
        self.body_curve[indices] = 0.0
        self.left_pectoral[indices] = 0.0
        self.right_pectoral[indices] = 0.0
        self.food[indices] = 0.0
        self.food_count[indices] = 0
        self.steps[indices] = 0

        # Spawn initial food for reset envs
        initial_food = self.config.get("initial_food", 5)
        for idx in indices:
            self._spawn_food_single(idx, initial_food)

        # Reset internal state
        self.hunger[indices] = 0.6
        self.stress[indices] = 0.0
        self.social_comfort[indices] = 0.5
        self.energy[indices] = 1.0

        # Reset exploration
        for idx in indices:
            self.visited_cells[idx] = [set() for _ in range(m)]
        self.prev_pos[indices] = self.pos[indices].copy()

    def step(self, actions: np.ndarray):
        """Step all environments with given actions."""
        actions = np.asarray(actions, dtype=np.float32)
        # Reshape to (num_envs, num_fish, 3)
        actions = actions.reshape(self.num_envs, self.num_fish, 3)

        speed = np.clip(actions[:, :, 0], 0.0, 1.0)
        direction = np.clip(actions[:, :, 1], -1.0, 1.0)
        urgency = np.clip(actions[:, :, 2], 0.0, 1.0)

        # Vectorized physics update
        self._physics_step_vec(speed, direction, urgency)

        # Age food
        for i in range(self.num_envs):
            if self.food_count[i] > 0:
                self.food[i, :self.food_count[i], 2] += self.dt

        # Check eating (returns per-fish food counts)
        food_eaten = self._check_eating_vec()

        # Compute equilibrium rewards per fish
        rewards = self._compute_equilibrium_vec(food_eaten)

        # Add social rewards
        rewards += self._compute_social_vec()

        # Add exploration reward
        rewards += self._compute_exploration_vec()

        # Aggregate rewards across fish (sum)
        total_rewards = rewards.sum(axis=1)

        # Spawn new food occasionally
        spawn_rate = self.config.get("food_spawn_rate", 0.02)
        spawn_mask = self.rng.random(self.num_envs) < spawn_rate
        for i in np.where(spawn_mask)[0]:
            self._spawn_food_single(i, 1)

        self.steps += 1

        # Check for truncation
        truncated = self.steps >= self.max_steps
        terminated = np.zeros(self.num_envs, dtype=bool)

        # Auto-reset truncated environments
        done = truncated | terminated
        if done.any():
            self.reset_envs(done)

        return self._get_obs(), total_rewards, terminated, truncated, {}

    def _physics_step_vec(self, speed: np.ndarray, direction: np.ndarray, urgency: np.ndarray):
        """Vectorized hybrid physics update for all environments and fish."""
        # speed, direction, urgency: (num_envs, num_fish)

        # === TAIL ANIMATION ===
        tail_freq = self.TAIL_FREQ_MIN + urgency * (self.TAIL_FREQ_MAX - self.TAIL_FREQ_MIN)
        self.tail_phase += tail_freq * 2.0 * np.pi * self.dt
        self.tail_phase = self.tail_phase % (2.0 * np.pi)

        # === BODY CURVE ===
        target_curve = direction * 0.4
        self.body_curve += (target_curve - self.body_curve) * 5.0 * self.dt

        # === PECTORAL FINS ===
        base_pec = 0.3 - speed * 0.2
        self.left_pectoral = base_pec + direction * 0.3
        self.right_pectoral = base_pec - direction * 0.3

        # === ROTATION ===
        effective_turn_rate = self.turn_rate * (0.5 + urgency * 0.5)
        self.angular_vel = direction * effective_turn_rate
        self.angle += self.angular_vel * self.dt

        # === THRUST ===
        tail_pulse = np.sin(self.tail_phase) ** 2
        thrust_factor = 0.3 + urgency * 0.7
        thrust = speed * self.thrust_coeff * self.fin_area_mult * thrust_factor
        thrust *= (0.5 + tail_pulse * 0.5)

        thrust_angle = self.angle - self.body_curve * 0.2
        cos_thrust = np.cos(thrust_angle)
        sin_thrust = np.sin(thrust_angle)
        fx = thrust * cos_thrust
        fy = thrust * sin_thrust

        # === DRAG ===
        self.current_speed = np.linalg.norm(self.vel, axis=2)
        effective_drag = self.drag_coeff * self.fin_area_mult + np.abs(self.body_curve) * 2.0

        drag_mask = self.current_speed > 0.0001
        drag_force = np.zeros_like(self.current_speed)
        drag_force[drag_mask] = effective_drag[drag_mask] * self.current_speed[drag_mask]

        # Apply drag in velocity direction
        vel_norm = np.zeros_like(self.vel)
        vel_norm[drag_mask] = self.vel[drag_mask] / self.current_speed[drag_mask, np.newaxis]
        fx -= drag_force * vel_norm[:, :, 0]
        fy -= drag_force * vel_norm[:, :, 1]

        # === INTEGRATION ===
        ax = fx / self.mass
        ay = fy / self.mass
        self.vel[:, :, 0] += ax * self.dt
        self.vel[:, :, 1] += ay * self.dt
        self.pos[:, :, 0] += self.vel[:, :, 0] * self.dt
        self.pos[:, :, 1] += self.vel[:, :, 1] * self.dt

        # === INTERNAL STATE UPDATES ===
        movement_intensity = speed * (0.3 + urgency * 0.7)
        self.energy -= self.energy_cost_base * movement_intensity
        self.energy += self.energy_recovery_rate * (1.0 - speed) * self.dt
        self.energy = np.clip(self.energy, 0.0, 1.0)

        self.hunger -= self.hunger_decay_rate * self.dt
        self.hunger = np.maximum(0.0, self.hunger)

        self.stress *= (1.0 - self.stress_decay_rate * self.dt)

        # === WRAPAROUND ===
        self.pos[:, :, 0] = self.pos[:, :, 0] % self.width
        self.pos[:, :, 1] = self.pos[:, :, 1] % self.height

    def _get_obs(self) -> np.ndarray:
        """Get observations for all environments (flattened per env)."""
        n = self.num_envs
        m = self.num_fish
        obs = np.zeros((n, m, self.OBS_DIM), dtype=np.float32)

        for env_idx in range(n):
            for fish_idx in range(m):
                # Raycasts (32 features)
                ray_obs = self._cast_rays_single(env_idx, fish_idx)
                obs[env_idx, fish_idx, :self.NUM_RAYS * 2] = ray_obs

                # Lateral line (16 features)
                lateral_start = self.NUM_RAYS * 2
                lateral_obs = self._sense_lateral_single(env_idx, fish_idx)
                obs[env_idx, fish_idx, lateral_start:lateral_start + self.NUM_LATERAL_SENSORS * 2] = lateral_obs

        # Proprioception (4 features) - vectorized
        proprio_start = self.NUM_RAYS * 2 + self.NUM_LATERAL_SENSORS * 2
        cos_angle = np.cos(self.angle)
        sin_angle = np.sin(self.angle)
        heading = np.stack([cos_angle, sin_angle], axis=2)  # (n, m, 2)
        perpendicular = np.stack([-sin_angle, cos_angle], axis=2)

        vel_forward = np.sum(self.vel * heading, axis=2) / 100.0
        vel_lateral = np.sum(self.vel * perpendicular, axis=2) / 100.0
        angular_vel_norm = self.angular_vel / self.turn_rate
        speed_norm = self.current_speed / 100.0

        obs[:, :, proprio_start] = np.clip(vel_forward, -1, 1)
        obs[:, :, proprio_start + 1] = np.clip(vel_lateral, -1, 1)
        obs[:, :, proprio_start + 2] = np.clip(angular_vel_norm, -1, 1)
        obs[:, :, proprio_start + 3] = np.clip(speed_norm, -1, 1)

        # Internal state (4 features)
        internal_start = proprio_start + 4
        obs[:, :, internal_start] = self.hunger
        obs[:, :, internal_start + 1] = self.stress
        obs[:, :, internal_start + 2] = self.social_comfort
        obs[:, :, internal_start + 3] = self.energy

        # Social features (4 features)
        social_start = internal_start + 4
        social_obs = self._compute_social_obs()
        obs[:, :, social_start:social_start + 4] = social_obs

        # Flatten to (num_envs, num_fish * OBS_DIM)
        return obs.reshape(n, m * self.OBS_DIM)

    def _compute_social_obs(self) -> np.ndarray:
        """Compute social observation features for all fish."""
        n = self.num_envs
        m = self.num_fish
        social = np.zeros((n, m, 4), dtype=np.float32)

        if m == 1:
            social[:, :, 0] = 1.0  # Max distance (normalized)
            return social

        for env_idx in range(n):
            for fish_idx in range(m):
                my_pos = self.pos[env_idx, fish_idx]
                my_angle = self.angle[env_idx, fish_idx]
                my_heading = np.array([np.cos(my_angle), np.sin(my_angle)])

                dists = []
                angles_to_others = []

                for other_idx in range(m):
                    if other_idx == fish_idx:
                        continue

                    diff = self.pos[env_idx, other_idx] - my_pos
                    # Wraparound
                    if diff[0] > self.width / 2:
                        diff[0] -= self.width
                    elif diff[0] < -self.width / 2:
                        diff[0] += self.width
                    if diff[1] > self.height / 2:
                        diff[1] -= self.height
                    elif diff[1] < -self.height / 2:
                        diff[1] += self.height

                    dist = np.linalg.norm(diff)
                    dists.append(dist)

                    if dist > 0.1:
                        angle_to_other = np.arctan2(diff[1], diff[0]) - my_angle
                        while angle_to_other > np.pi:
                            angle_to_other -= 2 * np.pi
                        while angle_to_other < -np.pi:
                            angle_to_other += 2 * np.pi
                    else:
                        angle_to_other = 0.0
                    angles_to_others.append(angle_to_other)

                dists = np.array(dists)
                angles_to_others = np.array(angles_to_others)

                # Feature 1: Normalized distance to nearest
                nearest_idx = np.argmin(dists)
                nearest_dist = dists[nearest_idx]
                social[env_idx, fish_idx, 0] = np.clip(nearest_dist / self.schooling_max_dist, 0.0, 1.0)

                # Feature 2: Angle to nearest
                social[env_idx, fish_idx, 1] = angles_to_others[nearest_idx] / np.pi

                # Feature 3: Number nearby
                num_nearby = np.sum(dists < self.schooling_max_dist)
                social[env_idx, fish_idx, 2] = num_nearby / (m - 1)

                # Feature 4: Average heading difference
                heading_diffs = []
                other_indices = [j for j in range(m) if j != fish_idx]
                for j_idx, j in enumerate(other_indices):
                    if dists[j_idx] < self.schooling_max_dist:
                        other_heading = np.array([
                            np.cos(self.angle[env_idx, j]),
                            np.sin(self.angle[env_idx, j])
                        ])
                        heading_diffs.append(np.dot(my_heading, other_heading))

                if heading_diffs:
                    social[env_idx, fish_idx, 3] = np.mean(heading_diffs)

        return social

    def _compute_social_vec(self) -> np.ndarray:
        """Compute social/schooling rewards for all fish with dynamic comfort zones."""
        n = self.num_envs
        m = self.num_fish
        rewards = np.zeros((n, m), dtype=np.float32)

        if m == 1:
            return rewards

        for env_idx in range(n):
            for fish_idx in range(m):
                # Dynamic comfort zones based on stress level
                # Stressed = tight school (30-100px), Calm = loose school (50-200px)
                stress = self.stress[env_idx, fish_idx]
                min_dist = 30 + (1 - stress) * 20   # 30-50px
                max_dist = 100 + (1 - stress) * 100  # 100-200px

                dists = []
                for other_idx in range(m):
                    if other_idx == fish_idx:
                        continue
                    diff = self.pos[env_idx, other_idx] - self.pos[env_idx, fish_idx]
                    if diff[0] > self.width / 2:
                        diff[0] -= self.width
                    elif diff[0] < -self.width / 2:
                        diff[0] += self.width
                    if diff[1] > self.height / 2:
                        diff[1] -= self.height
                    elif diff[1] < -self.height / 2:
                        diff[1] += self.height
                    dists.append(np.linalg.norm(diff))

                nearest_dist = min(dists)

                # Separation: strong penalty for collision
                if nearest_dist < self.collision_dist:
                    rewards[env_idx, fish_idx] -= self.separation_penalty
                elif nearest_dist < min_dist:
                    # Graduated penalty for being too close
                    rewards[env_idx, fish_idx] -= 0.02 * (min_dist - nearest_dist) / min_dist

                # Cohesion: reward for being in comfort zone (peak at center)
                elif min_dist <= nearest_dist <= max_dist:
                    zone_center = (min_dist + max_dist) / 2
                    distance_from_center = abs(nearest_dist - zone_center)
                    zone_half_width = (max_dist - min_dist) / 2
                    rewards[env_idx, fish_idx] += self.cohesion_reward * (1 - distance_from_center / zone_half_width)

                # Alignment reward
                nearest_idx = dists.index(nearest_dist)
                other_indices = [j for j in range(m) if j != fish_idx]
                nearest_fish = other_indices[nearest_idx]
                my_heading = np.array([
                    np.cos(self.angle[env_idx, fish_idx]),
                    np.sin(self.angle[env_idx, fish_idx])
                ])
                other_heading = np.array([
                    np.cos(self.angle[env_idx, nearest_fish]),
                    np.sin(self.angle[env_idx, nearest_fish])
                ])
                heading_dot = np.dot(my_heading, other_heading)
                if heading_dot > 0.7:
                    rewards[env_idx, fish_idx] += self.alignment_reward * (heading_dot - 0.7) / 0.3

                # Update social comfort based on dynamic zones
                in_comfort_zone = sum(1 for d in dists if min_dist < d < max_dist)
                too_close = sum(1 for d in dists if d < self.collision_dist)

                if too_close > 0:
                    self.social_comfort[env_idx, fish_idx] -= 0.05 * self.dt
                elif in_comfort_zone > 0:
                    self.social_comfort[env_idx, fish_idx] += 0.02 * self.dt
                else:
                    self.social_comfort[env_idx, fish_idx] -= 0.01 * self.dt

                self.social_comfort[env_idx, fish_idx] = np.clip(
                    self.social_comfort[env_idx, fish_idx], 0.0, 1.0
                )

        return rewards

    def _cast_rays_single(self, env_idx: int, fish_idx: int) -> np.ndarray:
        """Cast rays for a single fish in a single environment."""
        rays = np.zeros(self.NUM_RAYS * 2, dtype=np.float32)

        if self.food_count[env_idx] == 0:
            rays[0::2] = 1.0  # Max distance
            return rays

        pos = self.pos[env_idx, fish_idx]
        angle = self.angle[env_idx, fish_idx]
        food_positions = self.food[env_idx, :self.food_count[env_idx], :2]
        food_ages = self.food[env_idx, :self.food_count[env_idx], 2]
        food_aoes = self.food_aoe_base + food_ages * self.food_aoe_growth

        for ray_idx in range(self.NUM_RAYS):
            ray_angle = angle + self.ray_angles_rel[ray_idx]
            ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)])

            min_dist = self.MAX_RAY_LENGTH
            max_intensity = 0.0

            for food_idx in range(self.food_count[env_idx]):
                food_pos = food_positions[food_idx]
                food_aoe = food_aoes[food_idx]

                to_food = food_pos - pos
                if to_food[0] > self.width / 2:
                    to_food[0] -= self.width
                elif to_food[0] < -self.width / 2:
                    to_food[0] += self.width
                if to_food[1] > self.height / 2:
                    to_food[1] -= self.height
                elif to_food[1] < -self.height / 2:
                    to_food[1] += self.height

                proj = np.dot(to_food, ray_dir)
                if proj > 0 and proj < self.MAX_RAY_LENGTH:
                    closest = ray_dir * proj
                    dist_to_center = np.linalg.norm(to_food - closest)

                    if dist_to_center < food_aoe and proj < min_dist:
                        min_dist = proj
                        max_intensity = 1.0 - (dist_to_center / food_aoe)

            rays[ray_idx * 2] = min_dist / self.MAX_RAY_LENGTH
            rays[ray_idx * 2 + 1] = max_intensity

        return rays

    def _sense_lateral_single(self, env_idx: int, fish_idx: int) -> np.ndarray:
        """Sense lateral line for a single fish."""
        lateral = np.zeros(self.NUM_LATERAL_SENSORS * 2, dtype=np.float32)

        if self.food_count[env_idx] == 0:
            return lateral

        pos = self.pos[env_idx, fish_idx]
        angle = self.angle[env_idx, fish_idx]
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        heading = np.array([cos_a, sin_a])
        perpendicular = np.array([-sin_a, cos_a])

        food_positions = self.food[env_idx, :self.food_count[env_idx], :2]
        food_ages = self.food[env_idx, :self.food_count[env_idx], 2]
        food_aoes = self.food_aoe_base + food_ages * self.food_aoe_growth

        for sensor_idx in range(self.NUM_LATERAL_SENSORS):
            sensor_pos = (
                pos
                + heading * self.lateral_along[sensor_idx]
                + perpendicular * self.lateral_side[sensor_idx] * self.lateral_perp_dist
            )

            pressure = np.zeros(2, dtype=np.float32)

            for food_idx in range(self.food_count[env_idx]):
                food_pos = food_positions[food_idx]
                food_aoe = food_aoes[food_idx]

                to_food = food_pos - sensor_pos
                if to_food[0] > self.width / 2:
                    to_food[0] -= self.width
                elif to_food[0] < -self.width / 2:
                    to_food[0] += self.width
                if to_food[1] > self.height / 2:
                    to_food[1] -= self.height
                elif to_food[1] < -self.height / 2:
                    to_food[1] += self.height

                dist = np.linalg.norm(to_food)
                sensing_range = food_aoe * 2

                if dist < sensing_range and dist > 0.1:
                    intensity = (sensing_range - dist) / sensing_range
                    pressure += (to_food / dist) * intensity

            pressure_x = np.dot(pressure, heading)
            pressure_y = np.dot(pressure, perpendicular)

            lateral[sensor_idx * 2] = np.clip(pressure_x / 2.0, -1, 1)
            lateral[sensor_idx * 2 + 1] = np.clip(pressure_y / 2.0, -1, 1)

        return lateral

    def _check_eating_vec(self) -> np.ndarray:
        """Check eating for all environments, return per-fish food counts eaten."""
        eaten = np.zeros((self.num_envs, self.num_fish), dtype=np.int32)

        for env_idx in range(self.num_envs):
            i = 0
            while i < self.food_count[env_idx]:
                food_pos = self.food[env_idx, i, :2]
                food_eaten = False

                # Check all fish - first one wins
                for fish_idx in range(self.num_fish):
                    diff = self.pos[env_idx, fish_idx] - food_pos
                    if diff[0] > self.width / 2:
                        diff[0] -= self.width
                    elif diff[0] < -self.width / 2:
                        diff[0] += self.width
                    if diff[1] > self.height / 2:
                        diff[1] -= self.height
                    elif diff[1] < -self.height / 2:
                        diff[1] += self.height

                    dist = np.linalg.norm(diff)

                    if dist < self.eat_radius:
                        eaten[env_idx, fish_idx] += 1
                        self.hunger[env_idx, fish_idx] = min(
                            1.0, self.hunger[env_idx, fish_idx] + self.hunger_eat_restore
                        )
                        # Swap with last
                        self.food[env_idx, i] = self.food[env_idx, self.food_count[env_idx] - 1]
                        self.food_count[env_idx] -= 1
                        food_eaten = True
                        break

                if not food_eaten:
                    i += 1

        return eaten

    def _compute_equilibrium_vec(self, food_eaten: np.ndarray) -> np.ndarray:
        """Compute equilibrium rewards for all fish."""
        rewards = food_eaten.astype(np.float32) * 1.0

        # Target: all states near comfortable middle
        hunger_reward = -np.abs(self.hunger - 0.6) * 0.1
        stress_reward = -self.stress * 0.2
        social_reward = -np.abs(self.social_comfort - 0.6) * 0.1
        energy_reward = -np.abs(self.energy - 0.5) * 0.05

        rewards += hunger_reward + stress_reward + social_reward + energy_reward

        # Survival penalties
        rewards -= (self.hunger < 0.1).astype(np.float32) * 0.5
        rewards -= (self.energy < 0.1).astype(np.float32) * 0.3

        return rewards

    def _compute_exploration_vec(self) -> np.ndarray:
        """Compute exploration rewards for all fish."""
        n = self.num_envs
        m = self.num_fish
        rewards = np.zeros((n, m), dtype=np.float32)

        for env_idx in range(n):
            for fish_idx in range(m):
                grid_x = int(self.pos[env_idx, fish_idx, 0] / self.grid_size)
                grid_y = int(self.pos[env_idx, fish_idx, 1] / self.grid_size)
                cell = (grid_x, grid_y)

                if cell not in self.visited_cells[env_idx][fish_idx]:
                    self.visited_cells[env_idx][fish_idx].add(cell)
                    rewards[env_idx, fish_idx] += self.visit_bonus

                diff = self.pos[env_idx, fish_idx] - self.prev_pos[env_idx, fish_idx]
                if diff[0] > self.width / 2:
                    diff[0] -= self.width
                elif diff[0] < -self.width / 2:
                    diff[0] += self.width
                if diff[1] > self.height / 2:
                    diff[1] -= self.height
                elif diff[1] < -self.height / 2:
                    diff[1] += self.height

                distance = np.linalg.norm(diff)
                rewards[env_idx, fish_idx] += distance * self.distance_bonus

        self.prev_pos = self.pos.copy()
        return rewards

    def _spawn_food_all(self, count: int):
        """Spawn food for all environments."""
        for i in range(self.num_envs):
            self._spawn_food_single(i, count)

    def _spawn_food_single(self, env_idx: int, count: int):
        """Spawn food for a single environment."""
        for _ in range(count):
            if self.food_count[env_idx] >= self.max_food:
                break
            x = self.rng.uniform(0, self.width)
            y = self.rng.uniform(0, self.height)
            idx = self.food_count[env_idx]
            self.food[env_idx, idx] = [x, y, 0.0]
            self.food_count[env_idx] += 1
