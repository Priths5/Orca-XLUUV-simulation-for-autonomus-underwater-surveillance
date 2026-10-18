"""
observation.py
--------------
Builds the policy input. Two layouts, chosen by the policy file:

  "legacy12"     the original 12-d observation of sim/orca_env.py, same
                 order, same normalisation (mission range 5000 m, depth
                 rating 3400 m, uncertainty / 20000):
      [est_x_norm, est_y_norm, depth_norm, heading_sin, heading_cos,
       uncertainty_norm, battery_frac, thruster_wear, nav_residual,
       depth_margin, gradient_richness, risk]

  "obstacle_v1"  the same 12 features first (mission scale 150 m for the
                 Stonefish worlds), then what a goal-directed, obstacle-
                 avoiding policy needs and the original observation lacks:
      [12] sin(bearing to goal - heading)   (+ = goal to starboard)
      [13] cos(bearing to goal - heading)
      [14] distance to goal / 150, clipped to [0, 1.5]
      [15..19] Profiler sector clearance / 30 m, port -> starboard
               ([-80,-48] [-48,-16] [-16,16] [16,48] [48,80] deg)
      [20] port flank clearance / 30 m      (hull surface to remembered
      [21] starboard flank clearance / 30 m  Profiler returns, ObstacleMemory)
      [22] (depth - mission depth) / 5, clipped to [-1, 1]

Every number comes from the navigator (estimated position, never ground
truth in REALISTIC mode), the vehicle-health monitor, the sonar-derived
environment context, and the mission's goal.
"""

import math

import numpy as np

from .frames import bearing_to
from .perception import N_SECTORS, PROFILER_RANGE

LAYOUTS = {
    "legacy12": dict(obs_dim=12, mission_scale=5000.0),
    "obstacle_v1": dict(obs_dim=12 + 3 + N_SECTORS + 2 + 1, mission_scale=150.0),
}
DEPTH_RATING = 3400.0


class ObservationBuilder:
    def __init__(self, layout="obstacle_v1", mission_scale=None):
        if layout not in LAYOUTS:
            raise ValueError(f"unknown observation layout {layout}")
        self.layout = layout
        self.obs_dim = LAYOUTS[layout]["obs_dim"]
        self.mission_scale = mission_scale or LAYOUTS[layout]["mission_scale"]

    def build(self, est_x, est_y, origin, depth, heading, uncertainty, vehicle_health_vec,
              env_ctx, goal=None, clearances=None, flank=None, mission_depth=None):
        """vehicle_health_vec = VehicleHealth.as_vector(); env_ctx = EnvironmentHealth.sample()."""
        vh = vehicle_health_vec
        base = [
            (est_x - origin[0]) / self.mission_scale,
            (est_y - origin[1]) / self.mission_scale,
            depth / DEPTH_RATING,
            math.sin(heading),
            math.cos(heading),
            float(np.clip(uncertainty / 20000.0, 0, 3)),
            vh[0], vh[1], vh[2], vh[3],
            env_ctx["gradient_richness"],
            env_ctx["risk"],
        ]
        if self.layout == "legacy12":
            return np.array(base, dtype=np.float32)

        if goal is None:
            rel, dist = 0.0, 0.0
        else:
            rel, dist = bearing_to(est_x, est_y, heading, goal[0], goal[1])
        if clearances is None:
            clearances = np.full(N_SECTORS, PROFILER_RANGE)
        derr = 0.0 if mission_depth is None else float(np.clip((depth - mission_depth) / 5.0, -1, 1))
        extra = [math.sin(rel), math.cos(rel), float(np.clip(dist / self.mission_scale, 0, 1.5))]
        extra += list(np.asarray(clearances, dtype=float) / PROFILER_RANGE)
        if flank is None:
            flank = (PROFILER_RANGE, PROFILER_RANGE)
        extra += [flank[0] / PROFILER_RANGE, flank[1] / PROFILER_RANGE]
        extra += [derr]
        return np.array(base + extra, dtype=np.float32)


def action_to_command(action, params):
    """The original orca_env.py action scaling, with the vehicle limits in
    `params` (OrcaVehicleParams for the legacy policy, StonefishOrcaParams for
    policies trained against the identified Stonefish Orca)."""
    a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    return ((a[0] + 1) / 2 * params.max_speed,
            a[1] * params.max_yaw_rate,
            a[2] * params.max_depth_rate)
