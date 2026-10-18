"""
health_monitor.py
-------------------
Tracks vehicle-health and environment-health indicators every timestep.
Feeds both the RL observation space and the independent safety shield.
"""

import numpy as np


class VehicleHealth:
    def __init__(self, params):
        self.p = params
        self.battery_wh = params.battery_capacity_wh
        self.thruster_wear = 0.0          # 0 (new) -> 1 (failed)
        self.nav_residual_ema = 0.0       # sensor cross-check residual (drift indicator)
        self.depth_margin_frac = 1.0      # fraction of depth rating remaining

    def update(self, energy_used_wh, commanded_speed, achieved_speed, depth, dvl_dropout):
        self.battery_wh = max(0.0, self.battery_wh - energy_used_wh)
        # Thruster wear accrues slowly with usage, faster near max thrust
        load_frac = achieved_speed / max(self.p.max_speed, 1e-6)
        self.thruster_wear = min(1.0, self.thruster_wear + 1e-6 * load_frac)
        # Navigation residual: gap between commanded and achieved speed,
        # exponential moving average - a rising residual signals possible
        # thruster or sensor degradation.
        residual = abs(commanded_speed - achieved_speed)
        self.nav_residual_ema = 0.95 * self.nav_residual_ema + 0.05 * residual
        if dvl_dropout:
            self.nav_residual_ema += 0.02
        self.depth_margin_frac = 1.0 - (depth / self.p.depth_rating)

    def battery_frac(self):
        return self.battery_wh / self.p.battery_capacity_wh

    def is_critical(self):
        return (self.battery_frac() < 0.05) or (self.depth_margin_frac < 0.02)

    def as_vector(self):
        return np.array([
            self.battery_frac(),
            self.thruster_wear,
            self.nav_residual_ema,
            self.depth_margin_frac,
        ])


class EnvironmentHealth:
    """Aggregates environmental context signals into the observation."""

    def __init__(self, ocean_env):
        self.env = ocean_env

    def sample(self, x, y, t):
        u, v = self.env.current_at(x, y, t)
        current_strength = float(np.hypot(u, v))
        gradient_richness = self.env.gradient_at(x, y)
        risk = self.env.risk_at(x, y)
        return {
            "current_strength": current_strength,
            "current_u": u, "current_v": v,
            "gradient_richness": gradient_richness,
            "risk": risk,
        }
