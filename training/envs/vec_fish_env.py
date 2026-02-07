"""Vectorized multi-fish environment (per-env API)."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class VecFishEnv:
    NUM_RAYS = 16
    RAY_ARC = np.pi * 1.5  # 270 degrees
    MAX_RAY_LENGTH = 450.0
    NUM_LATERAL_SENSORS = 8
    OBS_DIM = NUM_RAYS * 2 + NUM_LATERAL_SENSORS * 2 + 4 + 4 + 4  # 60

    # Fin-based physics constants
    FIN_BODY_FREQ_MIN = 0.5
    FIN_BODY_FREQ_MAX = 4.0
    FIN_PEC_FREQ_MIN = 0.0
    FIN_PEC_FREQ_MAX = 3.0
    FIN_PEC_LEVER_ARM = 30.0

    def __init__(self, num_envs: int = 64, config: dict = None):
        self.num_envs = num_envs
        self.config = config or {}

        self.num_fish = self.config.get("num_fish", 3)
        self.width = self.config.get("width", 800)
        self.height = self.config.get("height", 600)

        self.single_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        total_obs_dim = self.num_fish * self.OBS_DIM
        total_action_dim = self.num_fish * 6

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_envs, total_obs_dim), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], self.num_fish).astype(np.float32),
            high=np.tile([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], self.num_fish).astype(np.float32),
            dtype=np.float32,
        )

        self.single_obs_dim = self.OBS_DIM
        self.single_action_dim = 6

        variety = self.config.get("variety", "common")
        self._set_variety(variety)

        self.max_food = 32
        self.eat_radius = 40.0
        self.food_aoe_base = 30.0
        self.food_aoe_growth = 5.0

        self.max_steps = self.config.get("max_steps", 1000)
        self.dt = 1.0 / 60.0

        self.energy_cost_base = 0.0001
        self.energy_recovery_rate = 0.0005
        self.hunger_decay_rate = 0.001
        self.hunger_eat_restore = 0.3
        self.stress_decay_rate = 0.1

        exploration_config = self.config.get("exploration", {})
        self.grid_size = exploration_config.get("grid_size", 100)
        self.visit_bonus = exploration_config.get("visit_bonus", 0.05)
        self.distance_bonus = exploration_config.get("distance_bonus", 0.001)

        self.schooling_min_dist = 50.0
        self.schooling_max_dist = 150.0
        self.collision_dist = 30.0
        self.cohesion_reward = 0.01
        self.separation_penalty = 0.05
        self.alignment_reward = 0.005

        t = np.linspace(0, 1, self.NUM_RAYS)
        self.ray_angles_rel = (t - 0.5) * self.RAY_ARC

        self.lateral_along = np.array([(i % 4 - 1.5) / 3.0 * 0.8 * self.body_length
                                        for i in range(self.NUM_LATERAL_SENSORS)])
        self.lateral_side = np.array([-1 if i < 4 else 1
                                       for i in range(self.NUM_LATERAL_SENSORS)])
        self.lateral_perp_dist = self.body_length / 4

        self.rng = np.random.default_rng()
        self._init_state_arrays()

    def _set_variety(self, variety: str):
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
        n = self.num_envs
        m = self.num_fish

        self.pos = np.zeros((n, m, 2), dtype=np.float32)
        self.vel = np.zeros((n, m, 2), dtype=np.float32)
        self.angle = np.zeros((n, m), dtype=np.float32)
        self.angular_vel = np.zeros((n, m), dtype=np.float32)
        self.current_speed = np.zeros((n, m), dtype=np.float32)
        self.tail_phase = np.zeros((n, m), dtype=np.float32)
        self.body_curve = np.zeros((n, m), dtype=np.float32)
        self.left_pectoral = np.zeros((n, m), dtype=np.float32)
        self.right_pectoral = np.zeros((n, m), dtype=np.float32)

        self.food = np.zeros((n, self.max_food, 3), dtype=np.float32)
        self.food_count = np.zeros(n, dtype=np.int32)

        self.steps = np.zeros(n, dtype=np.int32)

        self.hunger = np.ones((n, m), dtype=np.float32) * 0.6
        self.stress = np.zeros((n, m), dtype=np.float32)
        self.social_comfort = np.ones((n, m), dtype=np.float32) * 0.5
        self.energy = np.ones((n, m), dtype=np.float32)

        self.visited_cells = [[set() for _ in range(m)] for _ in range(n)]
        self.prev_pos = np.zeros((n, m, 2), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        n = self.num_envs
        m = self.num_fish

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

        initial_food = self.config.get("initial_food", 5)
        self._spawn_food_all(initial_food)

        self.hunger = np.ones((n, m), dtype=np.float32) * 0.6
        self.stress = np.zeros((n, m), dtype=np.float32)
        self.social_comfort = np.ones((n, m), dtype=np.float32) * 0.5
        self.energy = np.ones((n, m), dtype=np.float32)

        self.visited_cells = [[set() for _ in range(m)] for _ in range(n)]
        self.prev_pos = self.pos.copy()

        return self._get_obs(), {}

    def reset_envs(self, env_mask: np.ndarray):
        indices = np.where(env_mask)[0]
        n_reset = len(indices)

        if n_reset == 0:
            return

        m = self.num_fish

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

        initial_food = self.config.get("initial_food", 5)
        for idx in indices:
            self._spawn_food_single(idx, initial_food)

        self.hunger[indices] = 0.6
        self.stress[indices] = 0.0
        self.social_comfort[indices] = 0.5
        self.energy[indices] = 1.0

        for idx in indices:
            self.visited_cells[idx] = [set() for _ in range(m)]
        self.prev_pos[indices] = self.pos[indices].copy()

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32)
        actions = actions.reshape(self.num_envs, self.num_fish, 6)

        body_freq = np.clip(actions[:, :, 0], 0.0, 1.0)
        body_amp = np.clip(actions[:, :, 1], 0.0, 1.0)
        left_pec_freq = np.clip(actions[:, :, 2], 0.0, 1.0)
        left_pec_amp = np.clip(actions[:, :, 3], 0.0, 1.0)
        right_pec_freq = np.clip(actions[:, :, 4], 0.0, 1.0)
        right_pec_amp = np.clip(actions[:, :, 5], 0.0, 1.0)

        self._physics_step_fin(body_freq, body_amp, left_pec_freq, left_pec_amp, right_pec_freq, right_pec_amp)

        for i in range(self.num_envs):
            if self.food_count[i] > 0:
                self.food[i, :self.food_count[i], 2] += self.dt

        food_eaten = self._check_eating_vec()
        rewards = self._compute_equilibrium_vec(food_eaten)
        rewards += self._compute_social_vec()
        rewards += self._compute_exploration_vec()
        total_rewards = rewards.sum(axis=1)

        spawn_rate = self.config.get("food_spawn_rate", 0.02)
        spawn_mask = self.rng.random(self.num_envs) < spawn_rate
        for i in np.where(spawn_mask)[0]:
            self._spawn_food_single(i, 1)

        self.steps += 1

        truncated = self.steps >= self.max_steps
        terminated = np.zeros(self.num_envs, dtype=bool)

        done = truncated | terminated
        if done.any():
            self.reset_envs(done)

        return self._get_obs(), total_rewards, terminated, truncated, {}

    def _physics_step_fin(self, body_freq: np.ndarray, body_amp: np.ndarray,
                          left_pec_freq: np.ndarray, left_pec_amp: np.ndarray,
                          right_pec_freq: np.ndarray, right_pec_amp: np.ndarray):
        actual_body_freq = self.FIN_BODY_FREQ_MIN + body_freq * (self.FIN_BODY_FREQ_MAX - self.FIN_BODY_FREQ_MIN)
        actual_body_amp = body_amp

        self.tail_phase += actual_body_freq * 2.0 * np.pi * self.dt
        self.tail_phase = self.tail_phase % (2.0 * np.pi)

        fin_area = self.body_length * self.fin_area_mult * 0.1
        thrust = self.thrust_coeff * actual_body_amp * actual_body_freq * fin_area

        tail_pulse = np.sin(self.tail_phase) ** 2
        thrust *= (0.5 + tail_pulse * 0.5)

        left_pec_actual_freq = self.FIN_PEC_FREQ_MIN + left_pec_freq * (self.FIN_PEC_FREQ_MAX - self.FIN_PEC_FREQ_MIN)
        right_pec_actual_freq = self.FIN_PEC_FREQ_MIN + right_pec_freq * (self.FIN_PEC_FREQ_MAX - self.FIN_PEC_FREQ_MIN)

        left_force = left_pec_amp * left_pec_actual_freq
        right_force = right_pec_amp * right_pec_actual_freq
        torque = (right_force - left_force) * self.FIN_PEC_LEVER_ARM

        self.angular_vel += torque / self.mass * self.dt

        # Angular drag to prevent spinning forever
        angular_drag = 0.95
        self.angular_vel *= np.power(angular_drag, self.dt * 60.0)

        self.angle += self.angular_vel * self.dt

        target_curve = self.angular_vel * 0.3
        self.body_curve += (target_curve - self.body_curve) * 5.0 * self.dt

        self.left_pectoral = left_pec_amp
        self.right_pectoral = right_pec_amp

        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)

        thrust_angle = self.angle - self.body_curve * 0.2
        cos_thrust = np.cos(thrust_angle)
        sin_thrust = np.sin(thrust_angle)
        fx = thrust * cos_thrust
        fy = thrust * sin_thrust

        self.current_speed = np.linalg.norm(self.vel, axis=2)

        v_forward = self.vel[:, :, 0] * cos_a + self.vel[:, :, 1] * sin_a
        v_lateral = -self.vel[:, :, 0] * sin_a + self.vel[:, :, 1] * cos_a

        # Drag model (lateral drag much higher than forward)
        base_drag = self.drag_coeff * self.fin_area_mult * 0.01
        lateral_mult = 10.0
        curve_drag = np.abs(self.body_curve) * 0.5

        drag_forward = (base_drag + curve_drag) * v_forward * np.abs(v_forward)
        drag_lateral = (base_drag * lateral_mult) * v_lateral * np.abs(v_lateral)

        drag_x = drag_forward * cos_a - drag_lateral * sin_a
        drag_y = drag_forward * sin_a + drag_lateral * cos_a

        fx -= drag_x
        fy -= drag_y

        effective_mass = self.mass * 1.3
        ax = fx / effective_mass
        ay = fy / effective_mass
        self.vel[:, :, 0] += ax * self.dt
        self.vel[:, :, 1] += ay * self.dt
        self.pos[:, :, 0] += self.vel[:, :, 0] * self.dt
        self.pos[:, :, 1] += self.vel[:, :, 1] * self.dt

        movement_intensity = body_amp * body_freq + (left_pec_amp * left_pec_freq + right_pec_amp * right_pec_freq) * 0.5
        self.energy -= self.energy_cost_base * movement_intensity
        self.energy += self.energy_recovery_rate * (1.0 - movement_intensity) * self.dt
        self.energy = np.clip(self.energy, 0.0, 1.0)

        self.hunger -= self.hunger_decay_rate * self.dt
        self.hunger = np.maximum(0.0, self.hunger)

        self.stress *= (1.0 - self.stress_decay_rate * self.dt)

        collision_radius = 100.0
        separation_force = 300.0
        for env_idx in range(self.num_envs):
            for i in range(self.num_fish):
                sep_x, sep_y = 0.0, 0.0
                for j in range(self.num_fish):
                    if i == j:
                        continue
                    dx = self.pos[env_idx, i, 0] - self.pos[env_idx, j, 0]
                    dy = self.pos[env_idx, i, 1] - self.pos[env_idx, j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < collision_radius and dist > 0.1:
                        strength = ((collision_radius - dist) / collision_radius) ** 2
                        sep_x += (dx / dist) * strength * separation_force
                        sep_y += (dy / dist) * strength * separation_force
                self.vel[env_idx, i, 0] += sep_x * self.dt
                self.vel[env_idx, i, 1] += sep_y * self.dt

        # Screen wrapping (infinite canvas)
        self.pos[:, :, 0] = self.pos[:, :, 0] % self.width
        self.pos[:, :, 1] = self.pos[:, :, 1] % self.height

    def _get_obs(self) -> np.ndarray:
        n = self.num_envs
        m = self.num_fish
        obs = np.zeros((n, m, self.OBS_DIM), dtype=np.float32)

        for env_idx in range(n):
            for fish_idx in range(m):
                ray_obs = self._cast_rays_single(env_idx, fish_idx)
                obs[env_idx, fish_idx, :self.NUM_RAYS * 2] = ray_obs

                lateral_start = self.NUM_RAYS * 2
                lateral_obs = self._sense_lateral_single(env_idx, fish_idx)
                obs[env_idx, fish_idx, lateral_start:lateral_start + self.NUM_LATERAL_SENSORS * 2] = lateral_obs

        proprio_start = self.NUM_RAYS * 2 + self.NUM_LATERAL_SENSORS * 2
        cos_angle = np.cos(self.angle)
        sin_angle = np.sin(self.angle)
        heading = np.stack([cos_angle, sin_angle], axis=2)
        perpendicular = np.stack([-sin_angle, cos_angle], axis=2)

        vel_forward = np.sum(self.vel * heading, axis=2) / 100.0
        vel_lateral = np.sum(self.vel * perpendicular, axis=2) / 100.0
        angular_vel_norm = self.angular_vel / self.turn_rate
        speed_norm = self.current_speed / 100.0

        obs[:, :, proprio_start] = np.clip(vel_forward, -1, 1)
        obs[:, :, proprio_start + 1] = np.clip(vel_lateral, -1, 1)
        obs[:, :, proprio_start + 2] = np.clip(angular_vel_norm, -1, 1)
        obs[:, :, proprio_start + 3] = np.clip(speed_norm, -1, 1)

        internal_start = proprio_start + 4
        obs[:, :, internal_start] = self.hunger
        obs[:, :, internal_start + 1] = self.stress
        obs[:, :, internal_start + 2] = self.social_comfort
        obs[:, :, internal_start + 3] = self.energy

        social_start = internal_start + 4
        social_obs = self._compute_social_obs()
        obs[:, :, social_start:social_start + 4] = social_obs

        return obs.reshape(n, m * self.OBS_DIM)

    def _compute_social_obs(self) -> np.ndarray:
        n = self.num_envs
        m = self.num_fish
        social = np.zeros((n, m, 4), dtype=np.float32)

        if m == 1:
            social[:, :, 0] = 1.0
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

                nearest_idx = np.argmin(dists)
                nearest_dist = dists[nearest_idx]
                social[env_idx, fish_idx, 0] = np.clip(nearest_dist / self.schooling_max_dist, 0.0, 1.0)
                social[env_idx, fish_idx, 1] = angles_to_others[nearest_idx] / np.pi

                num_nearby = np.sum(dists < self.schooling_max_dist)
                social[env_idx, fish_idx, 2] = num_nearby / (m - 1)

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
        n = self.num_envs
        m = self.num_fish
        rewards = np.zeros((n, m), dtype=np.float32)

        if m == 1:
            return rewards

        for env_idx in range(n):
            for fish_idx in range(m):
                stress = self.stress[env_idx, fish_idx]
                min_dist = 30 + (1 - stress) * 20
                max_dist = 100 + (1 - stress) * 100

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

                if nearest_dist < self.collision_dist:
                    rewards[env_idx, fish_idx] -= self.separation_penalty
                elif nearest_dist < min_dist:
                    rewards[env_idx, fish_idx] -= 0.02 * (min_dist - nearest_dist) / min_dist
                elif min_dist <= nearest_dist <= max_dist:
                    zone_center = (min_dist + max_dist) / 2
                    distance_from_center = abs(nearest_dist - zone_center)
                    zone_half_width = (max_dist - min_dist) / 2
                    rewards[env_idx, fish_idx] += self.cohesion_reward * (1 - distance_from_center / zone_half_width)

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
        rays = np.zeros(self.NUM_RAYS * 2, dtype=np.float32)

        if self.food_count[env_idx] == 0:
            rays[0::2] = 1.0
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
        eaten = np.zeros((self.num_envs, self.num_fish), dtype=np.int32)

        for env_idx in range(self.num_envs):
            i = 0
            while i < self.food_count[env_idx]:
                food_pos = self.food[env_idx, i, :2]
                food_eaten = False

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
                        self.food[env_idx, i] = self.food[env_idx, self.food_count[env_idx] - 1]
                        self.food_count[env_idx] -= 1
                        food_eaten = True
                        break

                if not food_eaten:
                    i += 1

        return eaten

    def _compute_equilibrium_vec(self, food_eaten: np.ndarray) -> np.ndarray:
        rewards = food_eaten.astype(np.float32) * 1.0

        hunger_reward = -np.abs(self.hunger - 0.6) * 0.1
        stress_reward = -self.stress * 0.2
        social_reward = -np.abs(self.social_comfort - 0.6) * 0.1
        energy_reward = -np.abs(self.energy - 0.5) * 0.05

        rewards += hunger_reward + stress_reward + social_reward + energy_reward

        rewards -= (self.hunger < 0.1).astype(np.float32) * 0.5
        rewards -= (self.energy < 0.1).astype(np.float32) * 0.3

        return rewards

    def _compute_exploration_vec(self) -> np.ndarray:
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
        for i in range(self.num_envs):
            self._spawn_food_single(i, count)

    def _spawn_food_single(self, env_idx: int, count: int):
        for _ in range(count):
            if self.food_count[env_idx] >= self.max_food:
                break
            x = self.rng.uniform(0, self.width)
            y = self.rng.uniform(0, self.height)
            idx = self.food_count[env_idx]
            self.food[env_idx, idx] = [x, y, 0.0]
            self.food_count[env_idx] += 1
