"""
safety_shield.py
------------------
Intercepts every RL-commanded action and enforces hard physical/health
limits regardless of what the learned policy proposes. This is the
"Layer 4" described in the technical proposal: physical realism is
guaranteed structurally, not just reward-shaped.
"""

import numpy as np


class SafetyShield:
    def __init__(self, params):
        self.p = params

    def filter_action(self, cmd_speed, cmd_yaw_rate, cmd_depth_rate,
                       depth, battery_frac, uncertainty):
        overridden = False

        # Hard depth-rating margin: never allow commanding deeper than 98% of rating
        max_safe_depth = 0.98 * self.p.depth_rating
        if depth >= max_safe_depth and cmd_depth_rate > 0:
            cmd_depth_rate = -0.05
            overridden = True

        # Battery reserve: if below 8% reserve, force minimum-energy transit
        # (low speed, no unnecessary maneuvering) - "limp home" mode.
        if battery_frac < 0.08:
            cmd_speed = min(cmd_speed, self.p.min_cruise_speed)
            cmd_yaw_rate *= 0.3
            overridden = True

        # If localization uncertainty has grown extremely large, force a
        # conservative, minimum-drift heading hold rather than allow the
        # (possibly poorly-informed) policy to command large maneuvers.
        if uncertainty > 5000.0:  # trace of covariance, m^2 scale
            cmd_yaw_rate = float(np.clip(cmd_yaw_rate, -0.1 * self.p.max_yaw_rate,
                                          0.1 * self.p.max_yaw_rate))
            overridden = True

        return cmd_speed, cmd_yaw_rate, cmd_depth_rate, overridden


class ObstacleAwareSafetyShield(SafetyShield):
    """
    The original shield (depth rating, battery reserve, localisation
    uncertainty - unchanged, applied first) plus the hard limits a vehicle
    moving among obstacles needs. Inputs are sensor-derived only:

      clearances : Profiler sector clearances (m from the bow), port->stbd
      altitude   : DVL altitude above the seabed (m), None if no bottom lock
      depth_error: depth - mission depth (m)

    Rules (each sets overridden=True, which the reward penalises exactly
    like the original shield's overrides):
      * something within `stop_distance` ahead  -> crawl and turn hard
        toward the more open side
      * something within `slow_distance` ahead  -> cap speed
      * DVL altitude below `min_altitude`        -> no further descent
      * outside the mission depth band           -> push back into it
    """

    def __init__(self, params, stop_distance=7.0, slow_distance=12.0,
                 min_altitude=3.0, depth_band=3.0):
        super().__init__(params)
        self.stop_distance = stop_distance
        self.slow_distance = slow_distance
        self.min_altitude = min_altitude
        self.depth_band = depth_band

    def filter_action(self, cmd_speed, cmd_yaw_rate, cmd_depth_rate,
                       depth, battery_frac, uncertainty,
                       clearances=None, altitude=None, depth_error=None):
        cmd_speed, cmd_yaw_rate, cmd_depth_rate, overridden = super().filter_action(
            cmd_speed, cmd_yaw_rate, cmd_depth_rate, depth, battery_frac, uncertainty)

        if clearances is not None and len(clearances) >= 3:
            c = np.asarray(clearances, dtype=float)
            mid = len(c) // 2
            front = float(np.min(c[mid - 1:mid + 2])) if len(c) >= 5 else float(c[mid])
            dead_ahead = float(c[mid])
            if dead_ahead < self.stop_distance or front < 0.6 * self.stop_distance:
                port_open = float(np.mean(c[:mid]))
                stbd_open = float(np.mean(c[mid + 1:]))
                turn = 1.0 if stbd_open >= port_open else -1.0
                cmd_speed = min(cmd_speed, 0.15)
                cmd_yaw_rate = turn * self.p.max_yaw_rate
                overridden = True
            elif front < self.slow_distance and cmd_speed > 0.45:
                cmd_speed = 0.45
                overridden = True

        if altitude is not None and np.isfinite(altitude) and 0 < altitude < self.min_altitude \
                and cmd_depth_rate > 0:
            cmd_depth_rate = -0.05
            overridden = True
        if depth_error is not None:
            if depth_error > self.depth_band and cmd_depth_rate > -0.05:
                cmd_depth_rate = -0.1
                overridden = True
            elif depth_error < -self.depth_band and cmd_depth_rate < 0.05:
                cmd_depth_rate = 0.1
                overridden = True
        return cmd_speed, cmd_yaw_rate, cmd_depth_rate, overridden
