"""Multi-fish Gymnasium environment for RL training."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FishEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    NUM_RAYS = 16
    RAY_ARC = np.pi * 1.5  # 270 degrees
    MAX_RAY_LENGTH = 450.0
    NUM_LATERAL_SENSORS = 8

    # Observation: raycasts (32) + lateral (16) + proprio (4) + internal (4) + social (4) = 60
    OBS_DIM = NUM_RAYS * 2 + NUM_LATERAL_SENSORS * 2 + 4 + 4 + 4

    # Fin-based physics constants
    FIN_BODY_FREQ_MIN = 0.5
    FIN_BODY_FREQ_MAX = 4.0
    FIN_PEC_FREQ_MIN = 0.0
    FIN_PEC_FREQ_MAX = 3.0
    FIN_PEC_LEVER_ARM = 30.0

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        self.config = config or {}
        self.render_mode = render_mode

        self.num_fish = self.config.get("num_fish", 3)

        self.width = self.config.get("width", 800)
        self.height = self.config.get("height", 600)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_fish * self.OBS_DIM,), dtype=np.float32
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

        self.schooling_min_dist = 50.0   # Minimum comfortable distance
        self.schooling_max_dist = 150.0  # Maximum comfortable distance
        self.collision_dist = 30.0       # Too close - penalty
        self.cohesion_reward = 0.01      # Reward for being in comfort zone
        self.separation_penalty = 0.05   # Penalty for collision
        self.alignment_reward = 0.005    # Reward for similar heading

        self.pos = None          # (num_fish, 2)
        self.vel = None          # (num_fish, 2)
        self.angle = None        # (num_fish,)
        self.angular_vel = None  # (num_fish,)
        self.current_speed = None  # (num_fish,)
        self.tail_phase = None   # (num_fish,)
        self.body_curve = None   # (num_fish,)
        self.left_pectoral = None   # (num_fish,)
        self.right_pectoral = None  # (num_fish,)
        self.food = None
        self.food_count = 0
        self.steps = 0

        self.hunger = None       # (num_fish,)
        self.stress = None       # (num_fish,)
        self.social_comfort = None  # (num_fish,)
        self.energy = None       # (num_fish,)

        self.visited_cells = None  # list of sets, one per fish
        self.prev_pos = None     # (num_fish, 2)

    def _set_variety(self, variety: str):
        if variety == "comet":
            # Small fins, fast & agile
            self.body_length = 80.0
            self.fin_area_mult = 0.7
            self.thrust_coeff = 150.0
            self.drag_coeff = 3.0
            self.turn_rate = 2.0
            self.mass = 0.8
        elif variety == "fancy":
            # Large flowing fins, slow & graceful
            self.body_length = 80.0
            self.fin_area_mult = 1.5
            self.thrust_coeff = 80.0
            self.drag_coeff = 6.0
            self.turn_rate = 1.0
            self.mass = 1.2
        else:  # common (default)
            # Balanced
            self.body_length = 80.0
            self.fin_area_mult = 1.0
            self.thrust_coeff = 120.0
            self.drag_coeff = 4.0
            self.turn_rate = 1.5
            self.mass = 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        n = self.num_fish

        self.pos = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            while True:
                pos = self.np_random.uniform(
                    [self.body_length, self.body_length],
                    [self.width - self.body_length, self.height - self.body_length]
                )
                if i == 0:
                    break
                dists = np.linalg.norm(self.pos[:i] - pos, axis=1)
                if dists.min() > self.body_length * 2:
                    break
            self.pos[i] = pos

        self.vel = np.zeros((n, 2), dtype=np.float32)
        self.angle = self.np_random.uniform(0, 2 * np.pi, size=n).astype(np.float32)
        self.angular_vel = np.zeros(n, dtype=np.float32)
        self.current_speed = np.zeros(n, dtype=np.float32)
        self.tail_phase = np.zeros(n, dtype=np.float32)
        self.body_curve = np.zeros(n, dtype=np.float32)
        self.left_pectoral = np.zeros(n, dtype=np.float32)
        self.right_pectoral = np.zeros(n, dtype=np.float32)

        self.food = np.zeros((self.max_food, 3), dtype=np.float32)
        self.food_count = 0

        initial_food = self.config.get("initial_food", 5)
        self._spawn_food(initial_food)

        self.steps = 0

        self.hunger = np.ones(n, dtype=np.float32) * 0.6  # Start slightly hungry
        self.stress = np.zeros(n, dtype=np.float32)  # Start calm
        self.social_comfort = np.ones(n, dtype=np.float32) * 0.5  # Neutral
        self.energy = np.ones(n, dtype=np.float32)  # Start rested

        self.visited_cells = [set() for _ in range(n)]
        self.prev_pos = self.pos.copy()

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        actions = action.reshape(self.num_fish, 6)

        for i in range(self.num_fish):
            body_freq = float(np.clip(actions[i, 0], 0.0, 1.0))
            body_amp = float(np.clip(actions[i, 1], 0.0, 1.0))
            left_pec_freq = float(np.clip(actions[i, 2], 0.0, 1.0))
            left_pec_amp = float(np.clip(actions[i, 3], 0.0, 1.0))
            right_pec_freq = float(np.clip(actions[i, 4], 0.0, 1.0))
            right_pec_amp = float(np.clip(actions[i, 5], 0.0, 1.0))
            self._physics_step_fin(i, body_freq, body_amp, left_pec_freq, left_pec_amp, right_pec_freq, right_pec_amp)

        collision_radius = 100.0
        separation_force = 300.0
        for i in range(self.num_fish):
            sep_x, sep_y = 0.0, 0.0
            for j in range(self.num_fish):
                if i == j:
                    continue
                dx = self.pos[i, 0] - self.pos[j, 0]
                dy = self.pos[i, 1] - self.pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < collision_radius and dist > 0.1:
                    strength = ((collision_radius - dist) / collision_radius) ** 2
                    sep_x += (dx / dist) * strength * separation_force
                    sep_y += (dy / dist) * strength * separation_force
            self.vel[i, 0] += sep_x * self.dt
            self.vel[i, 1] += sep_y * self.dt

        # Screen wrapping (infinite canvas)
        self.pos[:, 0] = self.pos[:, 0] % self.width
        self.pos[:, 1] = self.pos[:, 1] % self.height

        if self.food_count > 0:
            self.food[: self.food_count, 2] += self.dt

        food_eaten = self._check_eating()

        # Compute equilibrium reward per fish
        rewards = self._compute_equilibrium_reward(food_eaten)

        rewards += self._compute_social_reward()

        rewards += self._compute_exploration_reward()

        total_reward = float(rewards.sum())

        spawn_rate = self.config.get("food_spawn_rate", 0.02)
        if self.np_random.random() < spawn_rate:
            self._spawn_food(1)

        self.steps += 1
        truncated = self.steps >= self.max_steps
        terminated = False

        return self._get_obs(), total_reward, terminated, truncated, {}

    def _physics_step_fin(self, fish_idx: int, body_freq: float, body_amp: float,
                          left_pec_freq: float, left_pec_amp: float,
                          right_pec_freq: float, right_pec_amp: float):
        i = fish_idx

        actual_body_freq = self.FIN_BODY_FREQ_MIN + body_freq * (self.FIN_BODY_FREQ_MAX - self.FIN_BODY_FREQ_MIN)
        actual_body_amp = body_amp

        self.tail_phase[i] += actual_body_freq * 2.0 * np.pi * self.dt
        if self.tail_phase[i] > 2.0 * np.pi:
            self.tail_phase[i] -= 2.0 * np.pi

        fin_area = self.body_length * self.fin_area_mult * 0.1
        thrust = self.thrust_coeff * actual_body_amp * actual_body_freq * fin_area

        tail_pulse = np.sin(self.tail_phase[i]) ** 2
        thrust *= (0.5 + tail_pulse * 0.5)

        left_pec_actual_freq = self.FIN_PEC_FREQ_MIN + left_pec_freq * (self.FIN_PEC_FREQ_MAX - self.FIN_PEC_FREQ_MIN)
        right_pec_actual_freq = self.FIN_PEC_FREQ_MIN + right_pec_freq * (self.FIN_PEC_FREQ_MAX - self.FIN_PEC_FREQ_MIN)

        left_force = left_pec_amp * left_pec_actual_freq
        right_force = right_pec_amp * right_pec_actual_freq
        torque = (right_force - left_force) * self.FIN_PEC_LEVER_ARM

        self.angular_vel[i] += torque / self.mass * self.dt

        # Angular drag to prevent spinning forever
        angular_drag = 0.95
        self.angular_vel[i] *= angular_drag ** (self.dt * 60.0)

        self.angle[i] += self.angular_vel[i] * self.dt

        target_curve = self.angular_vel[i] * 0.3
        self.body_curve[i] += (target_curve - self.body_curve[i]) * 5.0 * self.dt

        self.left_pectoral[i] = left_pec_amp
        self.right_pectoral[i] = right_pec_amp

        cos_a = np.cos(self.angle[i])
        sin_a = np.sin(self.angle[i])

        thrust_angle = self.angle[i] - self.body_curve[i] * 0.2
        fx = thrust * np.cos(thrust_angle)
        fy = thrust * np.sin(thrust_angle)

        self.current_speed[i] = np.linalg.norm(self.vel[i])

        v_forward = self.vel[i, 0] * cos_a + self.vel[i, 1] * sin_a
        v_lateral = -self.vel[i, 0] * sin_a + self.vel[i, 1] * cos_a

        # Drag model (lateral drag much higher than forward)
        base_drag = self.drag_coeff * self.fin_area_mult * 0.01
        lateral_mult = 10.0
        curve_drag = abs(self.body_curve[i]) * 0.5

        drag_forward = (base_drag + curve_drag) * v_forward * abs(v_forward)
        drag_lateral = (base_drag * lateral_mult) * v_lateral * abs(v_lateral)

        drag_x = drag_forward * cos_a - drag_lateral * sin_a
        drag_y = drag_forward * sin_a + drag_lateral * cos_a

        fx -= drag_x
        fy -= drag_y

        effective_mass = self.mass * 1.3
        ax = fx / effective_mass
        ay = fy / effective_mass

        self.vel[i, 0] += ax * self.dt
        self.vel[i, 1] += ay * self.dt
        self.pos[i, 0] += self.vel[i, 0] * self.dt
        self.pos[i, 1] += self.vel[i, 1] * self.dt

        movement_intensity = body_amp * body_freq + (left_pec_amp * left_pec_freq + right_pec_amp * right_pec_freq) * 0.5
        self.energy[i] -= self.energy_cost_base * movement_intensity
        self.energy[i] += self.energy_recovery_rate * (1.0 - movement_intensity) * self.dt
        self.energy[i] = np.clip(self.energy[i], 0.0, 1.0)

        # Hunger: slowly decays
        self.hunger[i] -= self.hunger_decay_rate * self.dt
        self.hunger[i] = max(0.0, self.hunger[i])

        # Stress: slowly decays when calm
        self.stress[i] *= (1.0 - self.stress_decay_rate * self.dt)

        # Boundary handling done after all fish updated in step()

    def _get_obs(self):
        obs = np.zeros((self.num_fish, self.OBS_DIM), dtype=np.float32)

        for i in range(self.num_fish):
            ray_obs = self._cast_rays_single(i)
            obs[i, : self.NUM_RAYS * 2] = ray_obs

            lateral_start = self.NUM_RAYS * 2
            lateral_obs = self._sense_lateral_line_single(i)
            obs[i, lateral_start : lateral_start + self.NUM_LATERAL_SENSORS * 2] = lateral_obs

            proprio_start = lateral_start + self.NUM_LATERAL_SENSORS * 2
            heading = np.array([np.cos(self.angle[i]), np.sin(self.angle[i])])
            perpendicular = np.array([-np.sin(self.angle[i]), np.cos(self.angle[i])])

            vel_forward = np.dot(self.vel[i], heading) / 100.0
            vel_lateral = np.dot(self.vel[i], perpendicular) / 100.0
            angular_vel_norm = self.angular_vel[i] / self.turn_rate
            speed_norm = self.current_speed[i] / 100.0

            obs[i, proprio_start] = np.clip(vel_forward, -1, 1)
            obs[i, proprio_start + 1] = np.clip(vel_lateral, -1, 1)
            obs[i, proprio_start + 2] = np.clip(angular_vel_norm, -1, 1)
            obs[i, proprio_start + 3] = np.clip(speed_norm, -1, 1)

            internal_start = proprio_start + 4
            obs[i, internal_start] = self.hunger[i]
            obs[i, internal_start + 1] = self.stress[i]
            obs[i, internal_start + 2] = self.social_comfort[i]
            obs[i, internal_start + 3] = self.energy[i]

            social_start = internal_start + 4
            social_obs = self._compute_other_fish_obs(i)
            obs[i, social_start : social_start + 4] = social_obs

        return obs.flatten()

    def _compute_other_fish_obs(self, fish_idx: int) -> np.ndarray:
        i = fish_idx
        n = self.num_fish

        if n == 1:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        other_indices = [j for j in range(n) if j != i]
        dists = []
        angles_to_others = []

        my_pos = self.pos[i]
        my_angle = self.angle[i]
        my_heading = np.array([np.cos(my_angle), np.sin(my_angle)])

        for j in other_indices:
            diff = self.pos[j] - my_pos
            # Handle wraparound
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
        nearest_dist_norm = np.clip(nearest_dist / self.schooling_max_dist, 0.0, 1.0)

        nearest_angle_norm = angles_to_others[nearest_idx] / np.pi

        num_nearby = np.sum(dists < self.schooling_max_dist)
        num_nearby_norm = num_nearby / (n - 1)

        heading_diffs = []
        for j_idx, j in enumerate(other_indices):
            if dists[j_idx] < self.schooling_max_dist:
                other_heading = np.array([np.cos(self.angle[j]), np.sin(self.angle[j])])
                # Dot product gives cos of angle between headings
                heading_diff = np.dot(my_heading, other_heading)
                heading_diffs.append(heading_diff)

        if heading_diffs:
            avg_heading_diff = np.mean(heading_diffs)
        else:
            avg_heading_diff = 0.0

        return np.array([
            nearest_dist_norm,
            nearest_angle_norm,
            num_nearby_norm,
            avg_heading_diff  # Already in [-1, 1] from dot product
        ], dtype=np.float32)

    def _compute_social_reward(self) -> np.ndarray:
        rewards = np.zeros(self.num_fish, dtype=np.float32)

        if self.num_fish == 1:
            return rewards

        for i in range(self.num_fish):
            # Dynamic comfort zones based on stress level
            # Stressed = tight school (30-100px), Calm = loose school (50-200px)
            stress = self.stress[i]
            min_dist = 30 + (1 - stress) * 20   # 30-50px
            max_dist = 100 + (1 - stress) * 100  # 100-200px

            dists = []
            for j in range(self.num_fish):
                if j == i:
                    continue
                diff = self.pos[j] - self.pos[i]
                # Handle wraparound
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
                rewards[i] -= self.separation_penalty
            elif nearest_dist < min_dist:
                rewards[i] -= 0.02 * (min_dist - nearest_dist) / min_dist

            elif min_dist <= nearest_dist <= max_dist:
                zone_center = (min_dist + max_dist) / 2
                distance_from_center = abs(nearest_dist - zone_center)
                zone_half_width = (max_dist - min_dist) / 2
                rewards[i] += self.cohesion_reward * (1 - distance_from_center / zone_half_width)

            nearest_idx = dists.index(nearest_dist)
            other_indices = [j for j in range(self.num_fish) if j != i]
            nearest_fish = other_indices[nearest_idx]
            my_heading = np.array([np.cos(self.angle[i]), np.sin(self.angle[i])])
            other_heading = np.array([np.cos(self.angle[nearest_fish]), np.sin(self.angle[nearest_fish])])
            heading_dot = np.dot(my_heading, other_heading)
            if heading_dot > 0.7:  # Within ~45 degrees
                rewards[i] += self.alignment_reward * (heading_dot - 0.7) / 0.3

            in_comfort_zone = sum(1 for d in dists if min_dist < d < max_dist)
            too_close = sum(1 for d in dists if d < self.collision_dist)

            if too_close > 0:
                self.social_comfort[i] -= 0.05 * self.dt
            elif in_comfort_zone > 0:
                self.social_comfort[i] += 0.02 * self.dt
            else:
                self.social_comfort[i] -= 0.01 * self.dt

            self.social_comfort[i] = np.clip(self.social_comfort[i], 0.0, 1.0)

        return rewards

    def _cast_rays_single(self, fish_idx: int) -> np.ndarray:
        rays = np.zeros(self.NUM_RAYS * 2, dtype=np.float32)
        pos = self.pos[fish_idx]
        angle = self.angle[fish_idx]

        for i in range(self.NUM_RAYS):
            t = i / (self.NUM_RAYS - 1) if self.NUM_RAYS > 1 else 0.5
            ray_angle = angle + (t - 0.5) * self.RAY_ARC
            ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)])

            min_dist = self.MAX_RAY_LENGTH
            max_intensity = 0.0

            for j in range(self.food_count):
                food_pos = self.food[j, :2]
                food_age = self.food[j, 2]
                food_aoe = self.food_aoe_base + food_age * self.food_aoe_growth

                to_food = food_pos - pos

                for dx in [-self.width, 0, self.width]:
                    for dy in [-self.height, 0, self.height]:
                        wrapped_to_food = to_food + np.array([dx, dy])
                        proj = np.dot(wrapped_to_food, ray_dir)

                        if proj > 0 and proj < self.MAX_RAY_LENGTH:
                            closest = ray_dir * proj
                            dist_to_center = np.linalg.norm(wrapped_to_food - closest)

                            if dist_to_center < food_aoe:
                                if proj < min_dist:
                                    min_dist = proj
                                    max_intensity = 1.0 - (dist_to_center / food_aoe)

            rays[i * 2] = min_dist / self.MAX_RAY_LENGTH
            rays[i * 2 + 1] = max_intensity

        return rays

    def _sense_lateral_line_single(self, fish_idx: int) -> np.ndarray:
        lateral = np.zeros(self.NUM_LATERAL_SENSORS * 2, dtype=np.float32)
        pos = self.pos[fish_idx]
        angle = self.angle[fish_idx]

        heading = np.array([np.cos(angle), np.sin(angle)])
        perpendicular = np.array([-np.sin(angle), np.cos(angle)])

        for i in range(self.NUM_LATERAL_SENSORS):
            side = -1 if i < 4 else 1
            along = (i % 4 - 1.5) / 3.0 * 0.8 * self.body_length

            sensor_pos = pos + heading * along + perpendicular * side * (self.body_length / 4)

            pressure = np.zeros(2, dtype=np.float32)

            for j in range(self.food_count):
                food_pos = self.food[j, :2]
                food_age = self.food[j, 2]
                food_aoe = self.food_aoe_base + food_age * self.food_aoe_growth

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

            pressure_local_x = np.dot(pressure, heading)
            pressure_local_y = np.dot(pressure, perpendicular)

            lateral[i * 2] = np.clip(pressure_local_x / 2.0, -1, 1)
            lateral[i * 2 + 1] = np.clip(pressure_local_y / 2.0, -1, 1)

        return lateral

    def _check_eating(self) -> np.ndarray:
        eaten = np.zeros(self.num_fish, dtype=np.int32)

        i = 0
        while i < self.food_count:
            food_pos = self.food[i, :2]
            food_eaten = False

            for fish_idx in range(self.num_fish):
                diff = self.pos[fish_idx] - food_pos
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
                    eaten[fish_idx] += 1
                    self.hunger[fish_idx] = min(1.0, self.hunger[fish_idx] + self.hunger_eat_restore)
                    # Remove food by swapping with last
                    self.food[i] = self.food[self.food_count - 1]
                    self.food_count -= 1
                    food_eaten = True
                    break  # Only one fish can eat each food item

            if not food_eaten:
                i += 1

        return eaten

    def _compute_equilibrium_reward(self, food_eaten: np.ndarray) -> np.ndarray:
        rewards = np.zeros(self.num_fish, dtype=np.float32)

        rewards += food_eaten.astype(np.float32) * 1.0

        # Target: all states near comfortable middle (0.5-0.7)
        hunger_reward = -np.abs(self.hunger - 0.6) * 0.1
        stress_reward = -self.stress * 0.2  # lower is better
        social_reward = -np.abs(self.social_comfort - 0.6) * 0.1
        energy_reward = -np.abs(self.energy - 0.5) * 0.05

        rewards += hunger_reward + stress_reward + social_reward + energy_reward

        rewards -= (self.hunger < 0.1).astype(np.float32) * 0.5  # starving penalty
        rewards -= (self.energy < 0.1).astype(np.float32) * 0.3  # exhausted penalty

        return rewards

    def _compute_exploration_reward(self) -> np.ndarray:
        rewards = np.zeros(self.num_fish, dtype=np.float32)

        for i in range(self.num_fish):
            grid_x = int(self.pos[i, 0] / self.grid_size)
            grid_y = int(self.pos[i, 1] / self.grid_size)
            cell = (grid_x, grid_y)

            if cell not in self.visited_cells[i]:
                self.visited_cells[i].add(cell)
                rewards[i] += self.visit_bonus

            diff = self.pos[i] - self.prev_pos[i]
            if diff[0] > self.width / 2:
                diff[0] -= self.width
            elif diff[0] < -self.width / 2:
                diff[0] += self.width
            if diff[1] > self.height / 2:
                diff[1] -= self.height
            elif diff[1] < -self.height / 2:
                diff[1] += self.height

            distance = np.linalg.norm(diff)
            rewards[i] += distance * self.distance_bonus

        self.prev_pos = self.pos.copy()
        return rewards

    def _spawn_food(self, count: int):
        for _ in range(count):
            if self.food_count >= self.max_food:
                break
            x = self.np_random.uniform(0, self.width)
            y = self.np_random.uniform(0, self.height)
            self.food[self.food_count] = [x, y, 0.0]
            self.food_count += 1

    def render(self):
        if self.render_mode == "rgb_array":
            import warnings
            warnings.warn("RGB rendering not implemented yet")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return None

    def close(self):
        pass
