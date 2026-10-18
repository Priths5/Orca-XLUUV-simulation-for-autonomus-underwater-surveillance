"""
obstacle_baseline.py
---------------------
Classical, hand-designed planner for the obstacle task - the counterpart of
baseline_planner.py for sim/obstacle_env.py. It reads ONLY the policy's
observation (layout "obstacle_v1"): goal bearing/distance, Profiler sector
clearances, flank clearances, depth error. Potential-field style: attraction
to the goal bearing, repulsion from close sectors, slow down when the bow is
crowded.

Used (a) as the comparison baseline when evaluating the RL policy and
(b) as the demonstrator for the behaviour-cloning warm start in
train_obstacle.py (ES then improves on it).
"""

import math

import numpy as np

SECTOR_CENTRES = np.radians([-64.0, -32.0, 0.0, 32.0, 64.0])


class ObstaclePotentialFieldPlanner:
    def __init__(self, trigger=28.0, gain=2.5, v_max=0.7, v_min=0.12, max_speed=1.1):
        self.trigger = trigger
        self.gain = gain
        self.v_max = v_max
        self.v_min = v_min
        self.max_speed = max_speed

    def act(self, obs):
        rel = math.atan2(obs[12], obs[13])
        dist = obs[14] * 150.0
        cl = np.asarray(obs[15:20], dtype=float) * 30.0
        port, stbd = obs[20] * 30.0, obs[21] * 30.0
        side = 1.0 if (cl[3] + 0.5 * cl[4]) >= (cl[1] + 0.5 * cl[0]) else -1.0
        rep = 0.0
        for a, c in zip(SECTOR_CENTRES, cl):
            if c < self.trigger:
                w = ((self.trigger - c) / self.trigger) ** 1.5
                w *= 1.8 if abs(a) < 0.4 else (1.0 if abs(a) < 0.8 else 0.5)
                rep += (-np.sign(a) if abs(a) > 1e-6 else side) * w * self.gain
        desired = math.atan2(math.sin(rel + rep), math.cos(rel + rep))
        yaw = float(np.clip(desired / 0.3, -1, 1))
        if port < 4 and yaw < 0:          # something along the port flank: don't swing into it
            yaw *= 0.2
        if stbd < 4 and yaw > 0:
            yaw *= 0.2
        front = float(cl[1:4].min())
        spd = float(np.interp(front, [7, 14, 28], [self.v_min, 0.3, self.v_max]))
        if abs(desired) > 1.0 or dist < 15.0:
            spd = min(spd, 0.35)
        return np.array([spd / self.max_speed * 2 - 1, yaw, float(np.clip(-obs[22] * 3, -1, 1))])
