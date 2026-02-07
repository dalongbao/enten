"""PufferLib-style vectorized multi-fish environment (per-agent API)."""

import numpy as np
from gymnasium import spaces


class PufferFishEnv:
    NUM_RAYS = 16
    RAY_ARC = np.pi * 1.5  # 270 degrees
    MAX_RAY_LENGTH = 450.0
    NUM_LATERAL_SENSORS = 8
    OBS_DIM = NUM_RAYS * 2 + NUM_LATERAL_SENSORS * 2 + 4 + 4 + 4

    # Fin-based physics constants
    FIN_BODY_FREQ_MIN = 0.5
    FIN_BODY_FREQ_MAX = 4.0
    FIN_PEC_FREQ_MIN = 0.0
    FIN_PEC_FREQ_MAX = 3.0
    FIN_PEC_LEVER_ARM = 30.0

    def __init__(self, num_envs: int = 64, num_fish: int = 3, config: dict = None):
        self.num_envs = num_envs
        self.num_fish = num_fish
        self.config = config or {}

        self.width = self.config.get("width", 800)
        self.height = self.config.get("height", 600)

        self.single_obs_dim = self.OBS_DIM
        self.single_action_dim = 6

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.single_obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        variety = self.config.get("variety", "common")
        self._set_variety(variety)

        self.max_food = self.config.get("max_food", 128)
        self.eat_radius = 40.0
        self.food_aoe_base = 30.0
        self.food_aoe_growth = 5.0

        self.max_steps = self.config.get("max_steps", 2000)
        self.dt = 1.0 / 60.0

        self.energy_cost_base = 0.0001
        self.energy_recovery_rate = 0.0005
        self.hunger_decay_rate = 0.01
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

        self.lateral_along = np.array([
            (i % 4 - 1.5) / 3.0 * 0.8 * self.body_length
            for i in range(self.NUM_LATERAL_SENSORS)
        ])
        self.lateral_side = np.array([
            -1 if i < 4 else 1 for i in range(self.NUM_LATERAL_SENSORS)
        ])
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
        else:  # common
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

        self.episode_returns = np.zeros((n, m), dtype=np.float32)
        self.episode_lengths = np.zeros((n, m), dtype=np.int32)

    def reset(self, seed=None):
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

        self._spawn_food_all(self._get_initial_food())

        self.hunger = np.ones((n, m), dtype=np.float32) * 0.6
        self.stress = np.zeros((n, m), dtype=np.float32)
        self.social_comfort = np.ones((n, m), dtype=np.float32) * 0.5
        self.energy = np.ones((n, m), dtype=np.float32)

        self.visited_cells = [[set() for _ in range(m)] for _ in range(n)]
        self.prev_pos = self.pos.copy()

        self.episode_returns = np.zeros((n, m), dtype=np.float32)
        self.episode_lengths = np.zeros((n, m), dtype=np.int32)

        return self._get_obs()

    def step(self, actions):
        n = self.num_envs
        m = self.num_fish

        actions = np.asarray(actions, dtype=np.float32).reshape(n, m, 6)

        body_freq = np.clip(actions[:, :, 0], 0.0, 1.0)
        body_amp = np.clip(actions[:, :, 1], 0.0, 1.0)
        left_pec_freq = np.clip(actions[:, :, 2], 0.0, 1.0)
        left_pec_amp = np.clip(actions[:, :, 3], 0.0, 1.0)
        right_pec_freq = np.clip(actions[:, :, 4], 0.0, 1.0)
        right_pec_amp = np.clip(actions[:, :, 5], 0.0, 1.0)

        self._physics_step_fin(body_freq, body_amp, left_pec_freq, left_pec_amp, right_pec_freq, right_pec_amp)

        for i in range(n):
            if self.food_count[i] > 0:
                self.food[i, :self.food_count[i], 2] += self.dt

        food_eaten = self._check_eating()

        rewards = self._compute_equilibrium_reward(food_eaten)
        rewards += self._compute_social_reward()
        rewards += self._compute_exploration_reward()

        spawn_rate = self.config.get("food_spawn_rate", 0.02)
        spawn_mask = self.rng.random(n) < spawn_rate
        for i in np.where(spawn_mask)[0]:
            self._spawn_food_single(i, 1)

        self.steps += 1

        self.episode_returns += rewards
        self.episode_lengths += 1

        env_truncated = self.steps >= self.max_steps
        env_terminated = np.zeros(n, dtype=bool)
        env_done = env_truncated | env_terminated

        fish_dones = np.broadcast_to(env_done[:, None], (n, m))

        infos = {}
        if env_done.any():
            done_returns = self.episode_returns[env_done].flatten()
            done_lengths = self.episode_lengths[env_done].flatten()
            infos["episode_return"] = done_returns
            infos["episode_length"] = done_lengths
            self._reset_envs(env_done)

        obs = self._get_obs()
        rewards_flat = rewards.flatten()
        dones_flat = fish_dones.flatten()

        return obs, rewards_flat, dones_flat, infos

    def _reset_envs(self, env_mask):
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

        for idx in indices:
            self._spawn_food_single(idx, self._get_initial_food())

        self.hunger[indices] = 0.6
        self.stress[indices] = 0.0
        self.social_comfort[indices] = 0.5
        self.energy[indices] = 1.0

        for idx in indices:
            self.visited_cells[idx] = [set() for _ in range(m)]
        self.prev_pos[indices] = self.pos[indices].copy()

        self.episode_returns[indices] = 0.0
        self.episode_lengths[indices] = 0

    def _physics_step_fin(self, body_freq, body_amp, left_pec_freq, left_pec_amp, right_pec_freq, right_pec_amp):
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

        # Collision avoidance between fish (vectorized)
        collision_radius = 100.0
        separation_force = 300.0
        n, m = self.num_envs, self.num_fish
        pos_i = self.pos[:, :, np.newaxis, :]  # (n, m, 1, 2)
        pos_j = self.pos[:, np.newaxis, :, :]  # (n, 1, m, 2)
        diff = pos_i - pos_j  # (n, m, m, 2)
        dist = np.linalg.norm(diff, axis=3)  # (n, m, m)
        dist = np.where(dist < 0.1, 1e6, dist)  # avoid div by zero, ignore self
        mask = (dist < collision_radius) & (dist < 1e5)
        strength = np.where(mask, ((collision_radius - dist) / collision_radius) ** 2, 0)
        sep = (diff / dist[:, :, :, np.newaxis]) * strength[:, :, :, np.newaxis] * separation_force
        sep = np.sum(sep, axis=2)  # (n, m, 2)
        self.vel += sep * self.dt

        # Screen wrapping (infinite canvas)
        self.pos[:, :, 0] = self.pos[:, :, 0] % self.width
        self.pos[:, :, 1] = self.pos[:, :, 1] % self.height

    def _get_obs(self):
        n = self.num_envs
        m = self.num_fish
        obs = np.zeros((n, m, self.OBS_DIM), dtype=np.float32)

        obs[:, :, :self.NUM_RAYS * 2] = self._cast_rays_batch()
        lateral_start = self.NUM_RAYS * 2
        obs[:, :, lateral_start:lateral_start + self.NUM_LATERAL_SENSORS * 2] = self._sense_lateral_batch()

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

        return obs.reshape(n * m, self.OBS_DIM)

    def _compute_social_obs(self):
        n, m = self.num_envs, self.num_fish
        social = np.zeros((n, m, 4), dtype=np.float32)

        if m == 1:
            social[:, :, 0] = 1.0
            return social

        pos_i = self.pos[:, :, np.newaxis, :]  # (n, m, 1, 2)
        pos_j = self.pos[:, np.newaxis, :, :]  # (n, 1, m, 2)
        diff = pos_j - pos_i  # (n, m, m, 2)

        diff[:, :, :, 0] = np.where(diff[:, :, :, 0] > self.width/2, diff[:, :, :, 0] - self.width, diff[:, :, :, 0])
        diff[:, :, :, 0] = np.where(diff[:, :, :, 0] < -self.width/2, diff[:, :, :, 0] + self.width, diff[:, :, :, 0])
        diff[:, :, :, 1] = np.where(diff[:, :, :, 1] > self.height/2, diff[:, :, :, 1] - self.height, diff[:, :, :, 1])
        diff[:, :, :, 1] = np.where(diff[:, :, :, 1] < -self.height/2, diff[:, :, :, 1] + self.height, diff[:, :, :, 1])

        dists = np.linalg.norm(diff, axis=3)  # (n, m, m)
        eye_mask = np.eye(m, dtype=bool)[np.newaxis, :, :]
        dists = np.where(eye_mask, 1e6, dists)

        nearest_idx = np.argmin(dists, axis=2)  # (n, m)
        nearest_dist = np.take_along_axis(dists, nearest_idx[:, :, np.newaxis], axis=2).squeeze(-1)
        social[:, :, 0] = np.clip(nearest_dist / self.schooling_max_dist, 0.0, 1.0)

        angles_to = np.arctan2(diff[:, :, :, 1], diff[:, :, :, 0]) - self.angle[:, :, np.newaxis]
        angles_to = (angles_to + np.pi) % (2 * np.pi) - np.pi
        nearest_angle = np.take_along_axis(angles_to, nearest_idx[:, :, np.newaxis], axis=2).squeeze(-1)
        social[:, :, 1] = nearest_angle / np.pi

        nearby = (dists < self.schooling_max_dist) & ~eye_mask
        social[:, :, 2] = np.sum(nearby, axis=2) / (m - 1)

        cos_a, sin_a = np.cos(self.angle), np.sin(self.angle)
        heading = np.stack([cos_a, sin_a], axis=-1)  # (n, m, 2)
        heading_i = heading[:, :, np.newaxis, :]
        heading_j = heading[:, np.newaxis, :, :]
        heading_dot = np.sum(heading_i * heading_j, axis=3)  # (n, m, m)
        heading_dot = np.where(nearby, heading_dot, 0)
        nearby_count = np.sum(nearby, axis=2, keepdims=True).clip(1, None)
        social[:, :, 3] = np.sum(heading_dot, axis=2) / nearby_count.squeeze(-1)

        return social

    def _compute_social_reward(self):
        n, m = self.num_envs, self.num_fish
        rewards = np.zeros((n, m), dtype=np.float32)

        if m == 1:
            return rewards

        pos_i = self.pos[:, :, np.newaxis, :]
        pos_j = self.pos[:, np.newaxis, :, :]
        diff = pos_j - pos_i

        diff[:, :, :, 0] = np.where(diff[:, :, :, 0] > self.width/2, diff[:, :, :, 0] - self.width, diff[:, :, :, 0])
        diff[:, :, :, 0] = np.where(diff[:, :, :, 0] < -self.width/2, diff[:, :, :, 0] + self.width, diff[:, :, :, 0])
        diff[:, :, :, 1] = np.where(diff[:, :, :, 1] > self.height/2, diff[:, :, :, 1] - self.height, diff[:, :, :, 1])
        diff[:, :, :, 1] = np.where(diff[:, :, :, 1] < -self.height/2, diff[:, :, :, 1] + self.height, diff[:, :, :, 1])

        dists = np.linalg.norm(diff, axis=3)
        eye_mask = np.eye(m, dtype=bool)[np.newaxis, :, :]
        dists = np.where(eye_mask, 1e6, dists)

        nearest_dist = np.min(dists, axis=2)
        nearest_idx = np.argmin(dists, axis=2)

        min_d = 30 + (1 - self.stress) * 20
        max_d = 100 + (1 - self.stress) * 100

        collision = nearest_dist < self.collision_dist
        too_close = (nearest_dist >= self.collision_dist) & (nearest_dist < min_d)
        in_zone = (nearest_dist >= min_d) & (nearest_dist <= max_d)

        rewards -= np.where(collision, self.separation_penalty, 0)
        rewards -= np.where(too_close, 0.02 * (min_d - nearest_dist) / min_d, 0)

        zone_center = (min_d + max_d) / 2
        zone_half = (max_d - min_d) / 2
        rewards += np.where(in_zone, self.cohesion_reward * (1 - np.abs(nearest_dist - zone_center) / zone_half), 0)

        cos_a, sin_a = np.cos(self.angle), np.sin(self.angle)
        heading = np.stack([cos_a, sin_a], axis=-1)
        heading_i = heading[:, :, np.newaxis, :]
        heading_j = heading[:, np.newaxis, :, :]
        heading_dot_all = np.sum(heading_i * heading_j, axis=3)
        nearest_heading_dot = np.take_along_axis(heading_dot_all, nearest_idx[:, :, np.newaxis], axis=2).squeeze(-1)
        aligned = nearest_heading_dot > 0.7
        rewards += np.where(aligned, self.alignment_reward * (nearest_heading_dot - 0.7) / 0.3, 0)

        too_close_any = np.any(dists < self.collision_dist, axis=2)
        in_comfort = np.any((dists > min_d[:, :, np.newaxis]) & (dists < max_d[:, :, np.newaxis]) & ~eye_mask, axis=2)

        self.social_comfort -= np.where(too_close_any, 0.05 * self.dt, 0)
        self.social_comfort += np.where(~too_close_any & in_comfort, 0.02 * self.dt, 0)
        self.social_comfort -= np.where(~too_close_any & ~in_comfort, 0.01 * self.dt, 0)
        self.social_comfort = np.clip(self.social_comfort, 0.0, 1.0)

        return rewards

    def _cast_rays_batch(self):
        n, m, r = self.num_envs, self.num_fish, self.NUM_RAYS
        rays = np.zeros((n, m, r * 2), dtype=np.float32)
        rays[:, :, 0::2] = 1.0

        food_mask = np.arange(self.max_food)[np.newaxis, :] < self.food_count[:, np.newaxis]
        if not food_mask.any():
            return rays

        angles = self.angle[:, :, np.newaxis] + self.ray_angles_rel
        ray_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (n, m, r, 2)

        fp = self.food[:, :, :2]  # (n, max_food, 2)
        fa = self.food_aoe_base + self.food[:, :, 2] * self.food_aoe_growth  # (n, max_food)

        to_food = fp[:, np.newaxis, :, :] - self.pos[:, :, np.newaxis, :]  # (n, m, max_food, 2)
        to_food[:, :, :, 0] = np.where(to_food[:, :, :, 0] > self.width/2, to_food[:, :, :, 0] - self.width, to_food[:, :, :, 0])
        to_food[:, :, :, 0] = np.where(to_food[:, :, :, 0] < -self.width/2, to_food[:, :, :, 0] + self.width, to_food[:, :, :, 0])
        to_food[:, :, :, 1] = np.where(to_food[:, :, :, 1] > self.height/2, to_food[:, :, :, 1] - self.height, to_food[:, :, :, 1])
        to_food[:, :, :, 1] = np.where(to_food[:, :, :, 1] < -self.height/2, to_food[:, :, :, 1] + self.height, to_food[:, :, :, 1])

        proj = np.einsum('nmfd,nmrd->nmrf', to_food, ray_dirs)  # (n, m, r, max_food)
        closest = ray_dirs[:, :, :, np.newaxis, :] * proj[:, :, :, :, np.newaxis]  # (n, m, r, max_food, 2)
        dist_to_center = np.linalg.norm(to_food[:, :, np.newaxis, :, :] - closest, axis=4)  # (n, m, r, max_food)

        valid = (proj > 0) & (proj < self.MAX_RAY_LENGTH) & food_mask[:, np.newaxis, np.newaxis, :]
        hit = valid & (dist_to_center < fa[:, np.newaxis, np.newaxis, :])

        proj_masked = np.where(hit, proj, self.MAX_RAY_LENGTH + 1)
        best_idx = np.argmin(proj_masked, axis=3)  # (n, m, r)
        best_proj = np.take_along_axis(proj, best_idx[:, :, :, np.newaxis], axis=3).squeeze(-1)
        best_dist = np.take_along_axis(dist_to_center, best_idx[:, :, :, np.newaxis], axis=3).squeeze(-1)
        best_aoe = np.take_along_axis(fa[:, np.newaxis, np.newaxis, :], best_idx[:, :, :, np.newaxis], axis=3).squeeze(-1)
        any_hit = np.any(hit, axis=3)

        rays[:, :, 0::2] = np.where(any_hit, best_proj / self.MAX_RAY_LENGTH, 1.0)
        rays[:, :, 1::2] = np.where(any_hit, 1.0 - best_dist / best_aoe, 0.0)

        return rays

    def _sense_lateral_batch(self):
        n, m, s = self.num_envs, self.num_fish, self.NUM_LATERAL_SENSORS
        lateral = np.zeros((n, m, s * 2), dtype=np.float32)

        food_mask = np.arange(self.max_food)[np.newaxis, :] < self.food_count[:, np.newaxis]
        if not food_mask.any():
            return lateral

        cos_a, sin_a = np.cos(self.angle), np.sin(self.angle)
        heading = np.stack([cos_a, sin_a], axis=-1)  # (n, m, 2)
        perp = np.stack([-sin_a, cos_a], axis=-1)

        along = self.lateral_along[np.newaxis, np.newaxis, :, np.newaxis]  # (1, 1, s, 1)
        side = self.lateral_side[np.newaxis, np.newaxis, :, np.newaxis] * self.lateral_perp_dist

        sensor_pos = (self.pos[:, :, np.newaxis, :] +
                      heading[:, :, np.newaxis, :] * along +
                      perp[:, :, np.newaxis, :] * side)  # (n, m, s, 2)

        fp = self.food[:, :, :2]  # (n, max_food, 2)
        fa = self.food_aoe_base + self.food[:, :, 2] * self.food_aoe_growth  # (n, max_food)

        to_food = fp[:, np.newaxis, np.newaxis, :, :] - sensor_pos[:, :, :, np.newaxis, :]  # (n, m, s, max_food, 2)
        to_food[:, :, :, :, 0] = np.where(to_food[:, :, :, :, 0] > self.width/2, to_food[:, :, :, :, 0] - self.width, to_food[:, :, :, :, 0])
        to_food[:, :, :, :, 0] = np.where(to_food[:, :, :, :, 0] < -self.width/2, to_food[:, :, :, :, 0] + self.width, to_food[:, :, :, :, 0])
        to_food[:, :, :, :, 1] = np.where(to_food[:, :, :, :, 1] > self.height/2, to_food[:, :, :, :, 1] - self.height, to_food[:, :, :, :, 1])
        to_food[:, :, :, :, 1] = np.where(to_food[:, :, :, :, 1] < -self.height/2, to_food[:, :, :, :, 1] + self.height, to_food[:, :, :, :, 1])

        dist = np.linalg.norm(to_food, axis=4)  # (n, m, s, max_food)
        sensing_range = fa[:, np.newaxis, np.newaxis, :] * 2
        valid = (dist > 0.1) & (dist < sensing_range) & food_mask[:, np.newaxis, np.newaxis, :]
        intensity = np.where(valid, (sensing_range - dist) / sensing_range, 0)

        dist_safe = np.maximum(dist, 0.1)
        direction = to_food / dist_safe[:, :, :, :, np.newaxis]
        pressure = np.sum(direction * intensity[:, :, :, :, np.newaxis], axis=3)  # (n, m, s, 2)

        pressure_x = np.sum(pressure * heading[:, :, np.newaxis, :], axis=3)
        pressure_y = np.sum(pressure * perp[:, :, np.newaxis, :], axis=3)

        lateral[:, :, 0::2] = np.clip(pressure_x / 2.0, -1, 1)
        lateral[:, :, 1::2] = np.clip(pressure_y / 2.0, -1, 1)

        return lateral

    def _cast_rays_single(self, env_idx, fish_idx):
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
                if 0 < proj < self.MAX_RAY_LENGTH:
                    closest = ray_dir * proj
                    dist_to_center = np.linalg.norm(to_food - closest)

                    if dist_to_center < food_aoe and proj < min_dist:
                        min_dist = proj
                        max_intensity = 1.0 - (dist_to_center / food_aoe)

            rays[ray_idx * 2] = min_dist / self.MAX_RAY_LENGTH
            rays[ray_idx * 2 + 1] = max_intensity

        return rays

    def _sense_lateral_single(self, env_idx, fish_idx):
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

                if 0.1 < dist < sensing_range:
                    intensity = (sensing_range - dist) / sensing_range
                    pressure += (to_food / dist) * intensity

            pressure_x = np.dot(pressure, heading)
            pressure_y = np.dot(pressure, perpendicular)

            lateral[sensor_idx * 2] = np.clip(pressure_x / 2.0, -1, 1)
            lateral[sensor_idx * 2 + 1] = np.clip(pressure_y / 2.0, -1, 1)

        return lateral

    def _check_eating(self):
        n, m = self.num_envs, self.num_fish
        eaten = np.zeros((n, m), dtype=np.int32)

        for env_idx in range(n):
            fc = self.food_count[env_idx]
            if fc == 0:
                continue

            fp = self.food[env_idx, :fc, :2]
            fish_pos = self.pos[env_idx]

            diff = fish_pos[:, np.newaxis, :] - fp[np.newaxis, :, :]
            diff[:, :, 0] = np.where(diff[:, :, 0] > self.width/2, diff[:, :, 0] - self.width, diff[:, :, 0])
            diff[:, :, 0] = np.where(diff[:, :, 0] < -self.width/2, diff[:, :, 0] + self.width, diff[:, :, 0])
            diff[:, :, 1] = np.where(diff[:, :, 1] > self.height/2, diff[:, :, 1] - self.height, diff[:, :, 1])
            diff[:, :, 1] = np.where(diff[:, :, 1] < -self.height/2, diff[:, :, 1] + self.height, diff[:, :, 1])

            dists = np.linalg.norm(diff, axis=2)
            can_eat = dists < self.eat_radius

            to_remove = []
            for food_idx in range(fc):
                eaters = np.where(can_eat[:, food_idx])[0]
                if len(eaters) > 0:
                    fish_idx = eaters[0]
                    eaten[env_idx, fish_idx] += 1
                    self.hunger[env_idx, fish_idx] = min(1.0, self.hunger[env_idx, fish_idx] + self.hunger_eat_restore)
                    to_remove.append(food_idx)
                    can_eat[:, food_idx] = False

            for food_idx in sorted(to_remove, reverse=True):
                self.food[env_idx, food_idx] = self.food[env_idx, self.food_count[env_idx] - 1]
                self.food_count[env_idx] -= 1

        return eaten

    def _compute_equilibrium_reward(self, food_eaten):
        rewards = food_eaten.astype(np.float32) * 1.0

        hunger_reward = -np.abs(self.hunger - 0.6) * 0.1
        stress_reward = -self.stress * 0.2
        social_reward = -np.abs(self.social_comfort - 0.6) * 0.1
        energy_reward = -np.abs(self.energy - 0.5) * 0.05

        rewards += hunger_reward + stress_reward + social_reward + energy_reward
        rewards -= (self.hunger < 0.1).astype(np.float32) * 0.5
        rewards -= (self.energy < 0.1).astype(np.float32) * 0.3

        return rewards

    def _compute_exploration_reward(self):
        n, m = self.num_envs, self.num_fish
        rewards = np.zeros((n, m), dtype=np.float32)

        grid_x = (self.pos[:, :, 0] / self.grid_size).astype(np.int32)
        grid_y = (self.pos[:, :, 1] / self.grid_size).astype(np.int32)

        for env_idx in range(n):
            for fish_idx in range(m):
                cell = (grid_x[env_idx, fish_idx], grid_y[env_idx, fish_idx])
                if cell not in self.visited_cells[env_idx][fish_idx]:
                    self.visited_cells[env_idx][fish_idx].add(cell)
                    rewards[env_idx, fish_idx] += self.visit_bonus

        diff = self.pos - self.prev_pos
        diff[:, :, 0] = np.where(diff[:, :, 0] > self.width/2, diff[:, :, 0] - self.width, diff[:, :, 0])
        diff[:, :, 0] = np.where(diff[:, :, 0] < -self.width/2, diff[:, :, 0] + self.width, diff[:, :, 0])
        diff[:, :, 1] = np.where(diff[:, :, 1] > self.height/2, diff[:, :, 1] - self.height, diff[:, :, 1])
        diff[:, :, 1] = np.where(diff[:, :, 1] < -self.height/2, diff[:, :, 1] + self.height, diff[:, :, 1])

        rewards += np.linalg.norm(diff, axis=2) * self.distance_bonus
        self.prev_pos = self.pos.copy()
        return rewards

    def _spawn_food_all(self, count):
        for i in range(self.num_envs):
            self._spawn_food_single(i, count)

    def _spawn_food_single(self, env_idx, count):
        for _ in range(count):
            if self.food_count[env_idx] >= self.max_food:
                break
            x = self.rng.uniform(0, self.width)
            y = self.rng.uniform(0, self.height)
            idx = self.food_count[env_idx]
            self.food[env_idx, idx] = [x, y, 0.0]
            self.food_count[env_idx] += 1

    def _get_initial_food(self):
        max_start = self.config.get("food_decay_max", 100)
        warmup = self.config.get("food_decay_warmup", 500_000)
        rate = self.config.get("food_decay_rate", 0.00001)
        floor = self.config.get("food_decay_floor", 10)
        step = getattr(self, 'global_step', 0)
        if step < warmup:
            return max_start
        return int((max_start - floor) / (np.exp(rate * (step - warmup)) + 1) + floor)

    def render(self, env_idx=0):
        import pygame
        if not hasattr(self, '_screen') or not pygame.get_init():
            pygame.init()
            self._screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('PufferFish Training')
            self._clock = pygame.time.Clock()
            self._font = pygame.font.Font(None, 24)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self._screen.fill((20, 40, 60))

        for i in range(self.food_count[env_idx]):
            fx, fy = self.food[env_idx, i, :2]
            age = self.food[env_idx, i, 2]
            radius = int(self.food_aoe_base + age * self.food_aoe_growth)
            pygame.draw.circle(self._screen, (80, 200, 80), (int(fx), int(fy)), radius, 1)
            pygame.draw.circle(self._screen, (120, 255, 120), (int(fx), int(fy)), 5)

        for fish_idx in range(self.num_fish):
            x, y = self.pos[env_idx, fish_idx]
            angle = self.angle[env_idx, fish_idx]

            cos_a, sin_a = np.cos(angle), np.sin(angle)
            head = (int(x + cos_a * 30), int(y + sin_a * 30))
            tail = (int(x - cos_a * 30), int(y - sin_a * 30))

            colors = [(232, 85, 48), (100, 180, 255), (255, 200, 50)]
            color = colors[fish_idx % len(colors)]
            pygame.draw.line(self._screen, color, tail, head, 4)
            pygame.draw.circle(self._screen, color, head, 8)

            vx, vy = self.vel[env_idx, fish_idx]
            speed = np.sqrt(vx*vx + vy*vy)
            hunger = self.hunger[env_idx, fish_idx]
            txt = self._font.render(f'{speed:.0f} h:{hunger:.2f}', True, (255, 255, 255))
            self._screen.blit(txt, (int(x) - 20, int(y) - 25))

        step = getattr(self, 'global_step', 0)
        food_target = self._get_initial_food()
        info = self._font.render(f'Step: {step:,}  Food: {self.food_count[env_idx]}/{food_target}', True, (255, 255, 255))
        self._screen.blit(info, (10, 10))

        pygame.display.flip()
        self._clock.tick(self.config.get("render_fps", 30))
