"""
Vectorized fish environment for fast RL training.

All operations are vectorized using NumPy - no Python loops over environments.
This enables training on hundreds of parallel environments efficiently.

Observation: raycasts (32) + lateral (16) + proprio (3) + hunger (1) = 52
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class VecFishEnv:
    """Fully vectorized fish environment running N fish in parallel."""

    # Perception constants (must match FishEnv)
    NUM_RAYS = 16
    RAY_ARC = np.pi  # 180 degrees
    MAX_RAY_LENGTH = 200.0
    NUM_LATERAL_SENSORS = 8
    OBS_DIM = NUM_RAYS * 2 + NUM_LATERAL_SENSORS * 2 + 3 + 1  # 52

    def __init__(self, num_envs: int = 64, config: dict = None):
        self.num_envs = num_envs
        self.config = config or {}

        # Environment dimensions
        self.width = self.config.get("width", 800)
        self.height = self.config.get("height", 600)

        # Spaces (for compatibility)
        self.single_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_envs, self.OBS_DIM), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.tile([0.0, -1.0], (num_envs, 1)).astype(np.float32),
            high=np.tile([1.0, 1.0], (num_envs, 1)).astype(np.float32),
            dtype=np.float32,
        )

        # Physics parameters (must match src/fish.h)
        self.mass = 1.0
        self.drag_coeff = 2.0
        self.max_thrust = 500.0
        self.turn_rate = 3.0
        self.dt = 1.0 / 60.0

        # Food parameters
        self.max_food = 32
        self.eat_radius = 25.0
        self.food_aoe_base = 30.0
        self.food_aoe_growth = 5.0

        # Episode settings
        self.max_steps = self.config.get("max_steps", 1000)

        # Hunger parameters
        hunger_config = self.config.get("hunger", {})
        self.hunger_initial = hunger_config.get("initial", 1.0)
        self.hunger_decay_rate = hunger_config.get("decay_rate", 0.001)
        self.hunger_eat_restore = hunger_config.get("eat_restore", 0.3)
        self.hunger_penalty_scale = hunger_config.get("penalty_scale", 0.01)

        # Exploration parameters
        exploration_config = self.config.get("exploration", {})
        self.grid_size = exploration_config.get("grid_size", 100)
        self.visit_bonus = exploration_config.get("visit_bonus", 0.05)
        self.distance_bonus = exploration_config.get("distance_bonus", 0.001)

        # Pre-compute ray angles (relative to heading)
        t = np.linspace(0, 1, self.NUM_RAYS)
        self.ray_angles_rel = (t - 0.5) * self.RAY_ARC  # [-pi/2, pi/2]

        # Lateral sensor offsets (in fish-local coords)
        # 4 sensors on each side, spread along body
        fish_length = 30.0
        self.lateral_along = np.array([(i % 4 - 1.5) / 3.0 * 0.8 * fish_length
                                        for i in range(self.NUM_LATERAL_SENSORS)])
        self.lateral_side = np.array([-1 if i < 4 else 1
                                       for i in range(self.NUM_LATERAL_SENSORS)])
        self.lateral_perp_dist = fish_length / 4

        # Initialize RNG
        self.rng = np.random.default_rng()

        # State arrays (initialized in reset)
        self._init_state_arrays()

    def _init_state_arrays(self):
        """Initialize all state arrays."""
        n = self.num_envs
        self.pos = np.zeros((n, 2), dtype=np.float32)
        self.vel = np.zeros((n, 2), dtype=np.float32)
        self.angle = np.zeros(n, dtype=np.float32)
        self.angular_vel = np.zeros(n, dtype=np.float32)

        # Food: (num_envs, max_food, 3) - x, y, age
        self.food = np.zeros((n, self.max_food, 3), dtype=np.float32)
        self.food_count = np.zeros(n, dtype=np.int32)

        self.steps = np.zeros(n, dtype=np.int32)

        # Hunger state
        self.hunger = np.ones(n, dtype=np.float32) * self.hunger_initial

        # Exploration state
        self.visited_cells = [set() for _ in range(n)]
        self.prev_pos = np.zeros((n, 2), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset all environments."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        n = self.num_envs
        self.pos = self.rng.uniform(
            [0, 0], [self.width, self.height], size=(n, 2)
        ).astype(np.float32)
        self.vel = np.zeros((n, 2), dtype=np.float32)
        self.angle = self.rng.uniform(0, 2 * np.pi, size=n).astype(np.float32)
        self.angular_vel = np.zeros(n, dtype=np.float32)

        self.food = np.zeros((n, self.max_food, 3), dtype=np.float32)
        self.food_count = np.zeros(n, dtype=np.int32)
        self.steps = np.zeros(n, dtype=np.int32)

        # Spawn initial food
        initial_food = self.config.get("initial_food", 5)
        self._spawn_food_all(initial_food)

        # Initialize hunger
        self.hunger = np.ones(n, dtype=np.float32) * self.hunger_initial

        # Initialize exploration
        self.visited_cells = [set() for _ in range(n)]
        self.prev_pos = self.pos.copy()

        return self._get_obs(), {}

    def reset_envs(self, env_mask: np.ndarray):
        """Reset only specified environments."""
        indices = np.where(env_mask)[0]
        n_reset = len(indices)

        if n_reset == 0:
            return

        self.pos[indices] = self.rng.uniform(
            [0, 0], [self.width, self.height], size=(n_reset, 2)
        ).astype(np.float32)
        self.vel[indices] = 0.0
        self.angle[indices] = self.rng.uniform(0, 2 * np.pi, size=n_reset).astype(np.float32)
        self.angular_vel[indices] = 0.0
        self.food[indices] = 0.0
        self.food_count[indices] = 0
        self.steps[indices] = 0

        # Spawn initial food for reset envs
        initial_food = self.config.get("initial_food", 5)
        for idx in indices:
            self._spawn_food_single(idx, initial_food)

        # Reset hunger
        self.hunger[indices] = self.hunger_initial

        # Reset exploration
        for idx in indices:
            self.visited_cells[idx] = set()
        self.prev_pos[indices] = self.pos[indices].copy()

    def step(self, actions: np.ndarray):
        """Step all environments with given actions."""
        actions = np.asarray(actions, dtype=np.float32)
        thrust = np.clip(actions[:, 0], 0.0, 1.0)
        turn = np.clip(actions[:, 1], -1.0, 1.0)

        # Vectorized physics update
        self._physics_step_vec(thrust, turn)

        # Age food
        for i in range(self.num_envs):
            if self.food_count[i] > 0:
                self.food[i, :self.food_count[i], 2] += self.dt

        # Check eating (returns rewards, also restores hunger)
        rewards = self._check_eating_vec()

        # Add exploration reward
        rewards += self._compute_exploration_vec()

        # Apply hunger decay and penalty
        self.hunger -= self.hunger_decay_rate
        self.hunger = np.maximum(0.0, self.hunger)
        hunger_penalty = self.hunger_penalty_scale * (1.0 - np.minimum(1.0, self.hunger))
        rewards -= hunger_penalty

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

        return self._get_obs(), rewards, terminated, truncated, {}

    def _physics_step_vec(self, thrust: np.ndarray, turn: np.ndarray):
        """Vectorized physics update for all environments."""
        # Angular update
        self.angular_vel = turn * self.turn_rate
        self.angle += self.angular_vel * self.dt

        # Thrust force in heading direction
        thrust_force = thrust * self.max_thrust
        cos_angle = np.cos(self.angle)
        sin_angle = np.sin(self.angle)
        fx = thrust_force * cos_angle
        fy = thrust_force * sin_angle

        # Drag force
        speed = np.linalg.norm(self.vel, axis=1)
        drag_mask = speed > 0.0001
        drag = np.zeros(self.num_envs, dtype=np.float32)
        drag[drag_mask] = self.drag_coeff * speed[drag_mask]

        # Apply drag (opposing velocity direction)
        fx[drag_mask] -= drag[drag_mask] * (self.vel[drag_mask, 0] / speed[drag_mask])
        fy[drag_mask] -= drag[drag_mask] * (self.vel[drag_mask, 1] / speed[drag_mask])

        # Integration
        ax = fx / self.mass
        ay = fy / self.mass
        self.vel[:, 0] += ax * self.dt
        self.vel[:, 1] += ay * self.dt
        self.pos[:, 0] += self.vel[:, 0] * self.dt
        self.pos[:, 1] += self.vel[:, 1] * self.dt

        # Wraparound
        self.pos[:, 0] = self.pos[:, 0] % self.width
        self.pos[:, 1] = self.pos[:, 1] % self.height

    def _get_obs(self) -> np.ndarray:
        """Get observations for all environments."""
        obs = np.zeros((self.num_envs, self.OBS_DIM), dtype=np.float32)

        # Raycasts (per-env, partially vectorized)
        ray_obs = self._cast_rays_vec()
        obs[:, :self.NUM_RAYS * 2] = ray_obs

        # Lateral line
        lateral_obs = self._sense_lateral_vec()
        lateral_start = self.NUM_RAYS * 2
        obs[:, lateral_start:lateral_start + self.NUM_LATERAL_SENSORS * 2] = lateral_obs

        # Proprioception (fully vectorized)
        cos_angle = np.cos(self.angle)
        sin_angle = np.sin(self.angle)
        heading = np.stack([cos_angle, sin_angle], axis=1)
        perpendicular = np.stack([-sin_angle, cos_angle], axis=1)

        vel_forward = np.sum(self.vel * heading, axis=1) / 100.0
        vel_lateral = np.sum(self.vel * perpendicular, axis=1) / 100.0
        angular_vel_norm = self.angular_vel / self.turn_rate

        obs[:, -4] = np.clip(vel_forward, -1, 1)
        obs[:, -3] = np.clip(vel_lateral, -1, 1)
        obs[:, -2] = np.clip(angular_vel_norm, -1, 1)

        # Hunger (1 feature)
        obs[:, -1] = np.clip(self.hunger, 0.0, 1.0)

        return obs

    def _cast_rays_vec(self) -> np.ndarray:
        """Cast rays for all environments."""
        rays = np.zeros((self.num_envs, self.NUM_RAYS * 2), dtype=np.float32)

        for env_idx in range(self.num_envs):
            if self.food_count[env_idx] == 0:
                # No food - all rays return max distance, zero intensity
                rays[env_idx, 0::2] = 1.0  # normalized distance
                continue

            pos = self.pos[env_idx]
            angle = self.angle[env_idx]
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

                    # Vector to food (with wraparound consideration)
                    to_food = food_pos - pos
                    # Handle wraparound
                    if to_food[0] > self.width / 2:
                        to_food[0] -= self.width
                    elif to_food[0] < -self.width / 2:
                        to_food[0] += self.width
                    if to_food[1] > self.height / 2:
                        to_food[1] -= self.height
                    elif to_food[1] < -self.height / 2:
                        to_food[1] += self.height

                    # Project onto ray
                    proj = np.dot(to_food, ray_dir)
                    if proj > 0 and proj < self.MAX_RAY_LENGTH:
                        closest = ray_dir * proj
                        dist_to_center = np.linalg.norm(to_food - closest)

                        if dist_to_center < food_aoe and proj < min_dist:
                            min_dist = proj
                            max_intensity = 1.0 - (dist_to_center / food_aoe)

                rays[env_idx, ray_idx * 2] = min_dist / self.MAX_RAY_LENGTH
                rays[env_idx, ray_idx * 2 + 1] = max_intensity

        return rays

    def _sense_lateral_vec(self) -> np.ndarray:
        """Sense lateral line for all environments."""
        lateral = np.zeros((self.num_envs, self.NUM_LATERAL_SENSORS * 2), dtype=np.float32)

        for env_idx in range(self.num_envs):
            if self.food_count[env_idx] == 0:
                continue

            pos = self.pos[env_idx]
            angle = self.angle[env_idx]
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
                    # Wraparound
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

                # Transform to fish-local coords
                pressure_x = np.dot(pressure, heading)
                pressure_y = np.dot(pressure, perpendicular)

                lateral[env_idx, sensor_idx * 2] = np.clip(pressure_x / 2.0, -1, 1)
                lateral[env_idx, sensor_idx * 2 + 1] = np.clip(pressure_y / 2.0, -1, 1)

        return lateral

    def _check_eating_vec(self) -> np.ndarray:
        """Check eating for all environments, return rewards."""
        rewards = np.zeros(self.num_envs, dtype=np.float32)

        for env_idx in range(self.num_envs):
            pos = self.pos[env_idx]
            i = 0
            while i < self.food_count[env_idx]:
                food_pos = self.food[env_idx, i, :2]

                # Distance with wraparound
                diff = pos - food_pos
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
                    rewards[env_idx] += 1.0
                    # Restore hunger (no cap - overeating allowed like real goldfish)
                    self.hunger[env_idx] += self.hunger_eat_restore
                    # Swap with last
                    self.food[env_idx, i] = self.food[env_idx, self.food_count[env_idx] - 1]
                    self.food_count[env_idx] -= 1
                else:
                    i += 1

        return rewards

    def _compute_exploration_vec(self) -> np.ndarray:
        """Compute exploration rewards for all environments."""
        rewards = np.zeros(self.num_envs, dtype=np.float32)

        for env_idx in range(self.num_envs):
            # Grid cell visit bonus
            grid_x = int(self.pos[env_idx, 0] / self.grid_size)
            grid_y = int(self.pos[env_idx, 1] / self.grid_size)
            cell = (grid_x, grid_y)

            if cell not in self.visited_cells[env_idx]:
                self.visited_cells[env_idx].add(cell)
                rewards[env_idx] += self.visit_bonus

            # Distance traveled bonus (with wraparound handling)
            diff = self.pos[env_idx] - self.prev_pos[env_idx]

            # Handle wraparound
            if diff[0] > self.width / 2:
                diff[0] -= self.width
            elif diff[0] < -self.width / 2:
                diff[0] += self.width
            if diff[1] > self.height / 2:
                diff[1] -= self.height
            elif diff[1] < -self.height / 2:
                diff[1] += self.height

            distance = np.linalg.norm(diff)
            rewards[env_idx] += distance * self.distance_bonus

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
