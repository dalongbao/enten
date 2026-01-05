"""
Single fish Gymnasium environment for RL training.

Physics matches src/fish.h exactly:
- mass = 1.0
- drag_coeff = 2.0
- max_thrust = 500.0
- turn_rate = 3.0
- eat_radius = 25.0

Perception:
- 16 raycasts in 180° frontal arc → 32 features
- 8 lateral line sensors (4 per side) → 16 features
- 3 proprioceptive features (forward vel, lateral vel, angular vel)
- Total: 51 features
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FishEnv(gym.Env):
    """Single fish environment with raycast + lateral line perception."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Perception constants
    NUM_RAYS = 16
    RAY_ARC = np.pi  # 180 degrees
    MAX_RAY_LENGTH = 200.0
    NUM_LATERAL_SENSORS = 8

    # Observation: raycasts (32) + lateral (16) + proprio (3) = 51
    OBS_DIM = NUM_RAYS * 2 + NUM_LATERAL_SENSORS * 2 + 3

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        self.config = config or {}
        self.render_mode = render_mode

        # Environment dimensions
        self.width = self.config.get("width", 800)
        self.height = self.config.get("height", 600)

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),  # thrust, turn
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Physics parameters (must match src/fish.h)
        self.mass = 1.0
        self.drag_coeff = 2.0
        self.max_thrust = 500.0
        self.turn_rate = 3.0
        self.dt = 1.0 / 60.0  # 60 FPS

        # Food parameters
        self.max_food = 32
        self.eat_radius = 25.0
        self.food_aoe_base = 30.0  # Base AOE radius
        self.food_aoe_growth = 5.0  # AOE growth per second

        # Episode settings
        self.max_steps = self.config.get("max_steps", 1000)

        # State (initialized in reset)
        self.pos = None
        self.vel = None
        self.angle = None
        self.angular_vel = None
        self.food = None  # (max_food, 3): x, y, age
        self.food_count = 0
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize fish at center with random angle
        self.pos = np.array(
            [self.width / 2.0, self.height / 2.0], dtype=np.float32
        )
        self.vel = np.zeros(2, dtype=np.float32)
        self.angle = self.np_random.uniform(0, 2 * np.pi)
        self.angular_vel = 0.0

        # Initialize food array
        self.food = np.zeros((self.max_food, 3), dtype=np.float32)
        self.food_count = 0

        # Spawn initial food
        initial_food = self.config.get("initial_food", 5)
        self._spawn_food(initial_food)

        self.steps = 0

        return self._get_obs(), {}

    def step(self, action):
        thrust = float(np.clip(action[0], 0.0, 1.0))
        turn = float(np.clip(action[1], -1.0, 1.0))

        # Physics update (matches fish.h exactly)
        self._physics_step(thrust, turn)

        # Age food (increases AOE)
        if self.food_count > 0:
            self.food[: self.food_count, 2] += self.dt

        # Check for food eating
        reward = self._check_eating()

        # Spawn new food occasionally
        spawn_rate = self.config.get("food_spawn_rate", 0.02)
        if self.np_random.random() < spawn_rate:
            self._spawn_food(1)

        self.steps += 1
        truncated = self.steps >= self.max_steps
        terminated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def _physics_step(self, thrust: float, turn: float):
        """Physics update matching src/fish.h exactly."""
        # Angular update
        self.angular_vel = turn * self.turn_rate
        self.angle += self.angular_vel * self.dt

        # Thrust force in heading direction
        thrust_force = thrust * self.max_thrust
        fx = thrust_force * np.cos(self.angle)
        fy = thrust_force * np.sin(self.angle)

        # Drag force opposing velocity
        speed = np.linalg.norm(self.vel)
        if speed > 0.0001:
            drag = self.drag_coeff * speed
            fx -= drag * (self.vel[0] / speed)
            fy -= drag * (self.vel[1] / speed)

        # Integration (F = ma, a = F/m)
        ax = fx / self.mass
        ay = fy / self.mass
        self.vel[0] += ax * self.dt
        self.vel[1] += ay * self.dt
        self.pos[0] += self.vel[0] * self.dt
        self.pos[1] += self.vel[1] * self.dt

        # Wraparound boundaries
        if self.pos[0] < 0:
            self.pos[0] += self.width
        if self.pos[0] >= self.width:
            self.pos[0] -= self.width
        if self.pos[1] < 0:
            self.pos[1] += self.height
        if self.pos[1] >= self.height:
            self.pos[1] -= self.height

    def _get_obs(self):
        """Construct observation vector: raycasts + lateral line + proprioception."""
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        # Raycasts (32 features)
        ray_obs = self._cast_rays()
        obs[: self.NUM_RAYS * 2] = ray_obs

        # Lateral line (16 features)
        lateral_start = self.NUM_RAYS * 2
        lateral_obs = self._sense_lateral_line()
        obs[lateral_start : lateral_start + self.NUM_LATERAL_SENSORS * 2] = lateral_obs

        # Proprioception (3 features)
        heading = np.array([np.cos(self.angle), np.sin(self.angle)])
        perpendicular = np.array([-np.sin(self.angle), np.cos(self.angle)])

        # Velocity in fish-relative coordinates, normalized
        vel_forward = np.dot(self.vel, heading) / 100.0
        vel_lateral = np.dot(self.vel, perpendicular) / 100.0
        angular_vel_norm = self.angular_vel / self.turn_rate

        obs[-3] = np.clip(vel_forward, -1, 1)
        obs[-2] = np.clip(vel_lateral, -1, 1)
        obs[-1] = np.clip(angular_vel_norm, -1, 1)

        return obs

    def _cast_rays(self):
        """Cast rays in frontal arc, detect food AOE intersections."""
        rays = np.zeros(self.NUM_RAYS * 2, dtype=np.float32)

        for i in range(self.NUM_RAYS):
            # Ray angle: spread across RAY_ARC centered on heading
            t = i / (self.NUM_RAYS - 1) if self.NUM_RAYS > 1 else 0.5
            ray_angle = self.angle + (t - 0.5) * self.RAY_ARC
            ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)])

            min_dist = self.MAX_RAY_LENGTH
            max_intensity = 0.0

            for j in range(self.food_count):
                food_pos = self.food[j, :2]
                food_age = self.food[j, 2]
                food_aoe = self.food_aoe_base + food_age * self.food_aoe_growth

                # Vector from fish to food
                to_food = food_pos - self.pos

                # Handle wraparound: check if food is closer via wrap
                for dx in [-self.width, 0, self.width]:
                    for dy in [-self.height, 0, self.height]:
                        wrapped_to_food = to_food + np.array([dx, dy])

                        # Project food onto ray direction
                        proj = np.dot(wrapped_to_food, ray_dir)

                        if proj > 0 and proj < self.MAX_RAY_LENGTH:
                            # Closest point on ray to food center
                            closest = ray_dir * proj
                            dist_to_center = np.linalg.norm(wrapped_to_food - closest)

                            # Check if ray passes through AOE
                            if dist_to_center < food_aoe:
                                if proj < min_dist:
                                    min_dist = proj
                                    # Intensity: how deep into AOE (0 at edge, 1 at center)
                                    max_intensity = 1.0 - (dist_to_center / food_aoe)

            rays[i * 2] = min_dist / self.MAX_RAY_LENGTH  # Normalized distance
            rays[i * 2 + 1] = max_intensity  # Food intensity

        return rays

    def _sense_lateral_line(self):
        """Sense pressure gradients from nearby food using lateral line."""
        lateral = np.zeros(self.NUM_LATERAL_SENSORS * 2, dtype=np.float32)

        # Fish coordinate system
        heading = np.array([np.cos(self.angle), np.sin(self.angle)])
        perpendicular = np.array([-np.sin(self.angle), np.cos(self.angle)])

        # Sensor positions: 4 on each side, spread along body
        # Layout: sensors 0-3 on left side, 4-7 on right side
        fish_length = 30.0  # Visual length of fish

        for i in range(self.NUM_LATERAL_SENSORS):
            # Side: left (-1) for first 4, right (+1) for last 4
            side = -1 if i < 4 else 1
            # Position along body: spread from -0.4 to +0.4 of length
            along = (i % 4 - 1.5) / 3.0 * 0.8 * fish_length

            sensor_pos = (
                self.pos + heading * along + perpendicular * side * (fish_length / 4)
            )

            # Accumulate pressure gradient from nearby food
            pressure = np.zeros(2, dtype=np.float32)

            for j in range(self.food_count):
                food_pos = self.food[j, :2]
                food_age = self.food[j, 2]
                food_aoe = self.food_aoe_base + food_age * self.food_aoe_growth

                # Vector from sensor to food (with wraparound)
                to_food = food_pos - sensor_pos

                # Handle wraparound
                if to_food[0] > self.width / 2:
                    to_food[0] -= self.width
                elif to_food[0] < -self.width / 2:
                    to_food[0] += self.width
                if to_food[1] > self.height / 2:
                    to_food[1] -= self.height
                elif to_food[1] < -self.height / 2:
                    to_food[1] += self.height

                dist = np.linalg.norm(to_food)

                # Lateral line detects within 2x AOE range
                sensing_range = food_aoe * 2
                if dist < sensing_range and dist > 0.1:
                    # Pressure gradient points toward food
                    # Intensity falls off with distance
                    intensity = (sensing_range - dist) / sensing_range
                    pressure += (to_food / dist) * intensity

            # Transform pressure to fish-local coordinates
            pressure_local_x = np.dot(pressure, heading)
            pressure_local_y = np.dot(pressure, perpendicular)

            lateral[i * 2] = np.clip(pressure_local_x / 2.0, -1, 1)
            lateral[i * 2 + 1] = np.clip(pressure_local_y / 2.0, -1, 1)

        return lateral

    def _check_eating(self):
        """Check if fish eats any food. Returns reward."""
        reward = 0.0
        i = 0
        while i < self.food_count:
            food_pos = self.food[i, :2]

            # Distance with wraparound
            diff = self.pos - food_pos
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
                reward += 1.0
                # Remove food by swapping with last
                self.food[i] = self.food[self.food_count - 1]
                self.food_count -= 1
            else:
                i += 1

        return reward

    def _spawn_food(self, count: int):
        """Spawn new food items at random positions."""
        for _ in range(count):
            if self.food_count >= self.max_food:
                break
            x = self.np_random.uniform(0, self.width)
            y = self.np_random.uniform(0, self.height)
            self.food[self.food_count] = [x, y, 0.0]  # x, y, age=0
            self.food_count += 1

    def render(self):
        """Render the environment (placeholder for now)."""
        if self.render_mode == "rgb_array":
            # Return a simple visualization
            import warnings

            warnings.warn("RGB rendering not implemented yet")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return None

    def close(self):
        pass
