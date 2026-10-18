"""
orca_env.py
------------
Custom Gym-style environment (reset/step API, no external `gymnasium`
dependency since it isn't installable in this sandbox) integrating:
  vehicle dynamics + sensors + ocean environment + particle-filter
  localization + vehicle/environment health + safety shield.

Observation (12-d):
  [est_x_norm, est_y_norm, depth_norm, heading_sin, heading_cos,
   uncertainty_norm, battery_frac, thruster_wear, nav_residual,
   depth_margin, gradient_richness, risk]

Action (3-d, continuous, in [-1, 1]):
  [speed_cmd, yaw_rate_cmd, depth_rate_cmd]  (scaled internally to physical units)

Reward:
  r = w1*progress - w2*uncertainty - w3*risk_exposure - w4*energy
      - w5*health_penalty - idle_penalty (+ terminal bonus for reaching goal)
"""

import numpy as np
from .vehicle_dynamics import OrcaVehicleParams, VehicleState, OrcaDynamics
from .sensors import SensorParams, NavigationSensors
from .ocean_environment import OceanEnvironment
from .localization import ParticleFilterLocalizer
from .health_monitor import VehicleHealth, EnvironmentHealth
from .safety_shield import SafetyShield


class OrcaNavEnv:
    def __init__(self, seed=0, mission_range_m=5000.0, dt=10.0, max_steps=450,
                 reward_weights=None, use_safety_shield=True, fixed_goal=None):
        self.rng = np.random.default_rng(seed)
        self.params = OrcaVehicleParams()
        self.dyn = OrcaDynamics(self.params, dt=dt)
        self.sensor_params = SensorParams()
        self.ocean = OceanEnvironment(size_m=mission_range_m * 2.2, grid_res=200, seed=seed)
        self.localizer = ParticleFilterLocalizer(self.ocean, n_particles=120, rng=self.rng)
        self.env_health = EnvironmentHealth(self.ocean)
        self.shield = SafetyShield(self.params)
        self.use_shield = use_safety_shield
        self.dt = dt
        self.max_steps = max_steps
        self.mission_range_m = mission_range_m

        # fixed_goal: optional (x, y) tuple in meters, relative to the start
        # position (0,0). If set, every reset() uses this exact goal instead
        # of a randomized one - use this to stress-test specific scenarios
        # (e.g. due-east run, steep diagonal, short-range, long-range).
        self.fixed_goal = fixed_goal

        self.w = reward_weights or dict(progress=1.0, uncertainty=0.0025,
                                 risk=0.4,           # was 2.0 - drastically reduce so
                                                      # per-step risk can't outweigh
                                                      # completing the mission
                                 energy=0.02, health=3.0,
                                 idle=0.5, goal_bonus=1500.0)  # was 200 - make finishing
                                                                # dominate any safe-pocket
                                                                # strategy

        self.action_dim = 3
        self.obs_dim = 12
        self.t = 0.0
        self.step_count = 0
        self.log = []

    def reset(self):
        self.step_count = 0
        self.t = 0.0
        self.log = []
        start_angle = self.rng.uniform(0, 2 * np.pi)
        self.true_state = VehicleState(x=0.0, y=0.0, depth=self.rng.uniform(80, 200),
                                        heading=start_angle, speed=0.0)

        if self.fixed_goal is not None:
            self.goal = np.array(self.fixed_goal, dtype=float)
        else:
            goal_angle = start_angle + self.rng.uniform(-0.6, 0.6)
            self.goal = np.array([
                self.mission_range_m * np.cos(goal_angle),
                self.mission_range_m * np.sin(goal_angle),
            ])

        self.sensors = NavigationSensors(self.sensor_params, dt=self.dt, rng=self.rng)
        self.sensors.reset()
        self.vehicle_health = VehicleHealth(self.params)
        self.localizer.reset(self.true_state.x, self.true_state.y)

        self.est_x, self.est_y = self.true_state.x, self.true_state.y
        self.uncertainty = self.localizer.init_std ** 2 * 2
        self.prev_dist_to_goal = np.hypot(self.goal[0] - self.est_x, self.goal[1] - self.est_y)

        return self._get_obs(), {}

    def _get_obs(self):
        depth_norm = self.true_state.depth / self.params.depth_rating
        unc_norm = np.clip(self.uncertainty / 20000.0, 0, 3)
        env_ctx = self.env_health.sample(self.est_x, self.est_y, self.t)
        vh = self.vehicle_health.as_vector()
        obs = np.array([
            self.est_x / self.mission_range_m,
            self.est_y / self.mission_range_m,
            depth_norm,
            np.sin(self.true_state.heading),
            np.cos(self.true_state.heading),
            unc_norm,
            vh[0], vh[1], vh[2], vh[3],
            env_ctx["gradient_richness"],
            env_ctx["risk"],
        ], dtype=np.float32)
        return obs

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        cmd_speed = (action[0] + 1) / 2 * self.params.max_speed
        cmd_yaw_rate = action[1] * self.params.max_yaw_rate
        cmd_depth_rate = action[2] * self.params.max_depth_rate

        if self.use_shield:
            cmd_speed, cmd_yaw_rate, cmd_depth_rate, overridden = self.shield.filter_action(
                cmd_speed, cmd_yaw_rate, cmd_depth_rate,
                self.true_state.depth, self.vehicle_health.battery_frac(), self.uncertainty)
        else:
            overridden = False

        current_u, current_v = self.ocean.current_at(self.true_state.x, self.true_state.y, self.t)
        prev_pos = np.array([self.true_state.x, self.true_state.y])
        self.true_state, energy_wh, power_w = self.dyn.step(
            self.true_state, cmd_speed, cmd_yaw_rate, cmd_depth_rate,
            current_vec=(current_u, current_v, 0.0))
        dist_step = np.hypot(self.true_state.x - prev_pos[0], self.true_state.y - prev_pos[1])

        self.vehicle_health.update(energy_wh, cmd_speed, self.true_state.speed,
                                    self.true_state.depth, dvl_dropout=False)

        # ---- sensor + localization update ----
        reading = self.sensors.measure(self.true_state, dist_step)
        dx = reading["dvl_speed"] * np.cos(reading["heading"]) * self.dt
        dy = reading["dvl_speed"] * np.sin(reading["heading"]) * self.dt
        process_noise = 0.5 + 0.5 * dist_step / 50.0
        self.localizer.predict(dx, dy, process_noise)

        true_bottom_depth = self.ocean.depth_at(self.true_state.x, self.true_state.y)
        bathy_reading = self.sensors.bathymetry_reading(true_bottom_depth)
        self.localizer.update_with_bathymetry(bathy_reading, self.sensor_params.bathy_noise_std)

        self.est_x, self.est_y, self.uncertainty, _ = self.localizer.estimate()

        # ---- reward ----
        dist_to_goal = np.hypot(self.goal[0] - self.est_x, self.goal[1] - self.est_y)
        progress = self.prev_dist_to_goal - dist_to_goal
        self.prev_dist_to_goal = dist_to_goal

        env_ctx = self.env_health.sample(self.est_x, self.est_y, self.t)
        risk = env_ctx["risk"]
        health_penalty = 1.0 if (overridden or self.vehicle_health.is_critical()) else 0.0

        # Idle penalty: without this, a policy that simply stops moving can
        # "farm" a locally-decent reward by avoiding risk/energy/uncertainty
        # costs altogether rather than actually reaching the goal. This term
        # makes standing still strictly worse than making progress.
        idle_penalty = self.w["idle"] if self.true_state.speed < 0.2 else 0.0

        reward = (self.w["progress"] * progress
                  - self.w["uncertainty"] * self.uncertainty / 100.0
                  - self.w["risk"] * risk
                  - self.w["energy"] * (energy_wh / 100.0)
                  - self.w["health"] * health_penalty
                  - idle_penalty)

        self.t += self.dt
        self.step_count += 1
        done = False
        reached_goal = bool(dist_to_goal < 150.0)
        if reached_goal:
            reward += self.w["goal_bonus"]
            done = True
        if self.vehicle_health.battery_frac() <= 0.0:
            reward -= 50.0
            done = True
        if self.step_count >= self.max_steps:
            done = True

        info = {
            "true_x": float(self.true_state.x), "true_y": float(self.true_state.y),
            "est_x": float(self.est_x), "est_y": float(self.est_y),
            "depth": float(self.true_state.depth), "uncertainty": float(self.uncertainty),
            "battery_frac": float(self.vehicle_health.battery_frac()),
            "risk": float(risk), "reached_goal": reached_goal,
            "overridden": bool(overridden), "power_w": float(power_w),
            "goal_x": float(self.goal[0]), "goal_y": float(self.goal[1]),
            "t": float(self.t),
        }
        self.log.append(info)
        return self._get_obs(), reward, done, info
