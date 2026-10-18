"""
baseline_planner.py
---------------------
Re-implementation of the classical approach used in the existing literature
(entropy/gradient-richness-based potential-field planning, e.g. Penumarti &
Shin 2024; Palomeras et al. 2019 action-selection-by-entropy-reduction
principle) as the primary comparison baseline. This planner does NOT learn;
it uses a fixed, hand-designed rule combining goal-attraction with a
"detour toward information-rich terrain when uncertain" heuristic, and does
NOT reason about energy or detection-risk jointly (matching the literature's
stated scope).
"""

import numpy as np


class ClassicalPotentialFieldPlanner:
    def __init__(self, params, uncertainty_threshold=3000.0, detour_gain=0.6):
        self.p = params
        self.uncertainty_threshold = uncertainty_threshold
        self.detour_gain = detour_gain

    def act(self, obs, ocean_env, est_x, est_y, goal, uncertainty):
        goal_vec = goal - np.array([est_x, est_y])
        goal_heading = np.arctan2(goal_vec[1], goal_vec[0])

        # If uncertainty is high, bias heading toward the direction of
        # locally increasing bathymetric gradient (information-rich terrain),
        # exactly mirroring the entropy-based detour logic in the classical
        # TAN literature - sampled via finite differences on the grid.
        if uncertainty > self.uncertainty_threshold:
            eps = 200.0
            g0 = ocean_env.gradient_at(est_x, est_y)
            gx = ocean_env.gradient_at(est_x + eps, est_y) - g0
            gy = ocean_env.gradient_at(est_x, est_y + eps) - g0
            info_heading = np.arctan2(gy, gx)
            heading_cmd = (1 - self.detour_gain) * goal_heading + self.detour_gain * info_heading
        else:
            heading_cmd = goal_heading

        # obs[3], obs[4] are sin(heading), cos(heading) of the vehicle's ACTUAL
        # current heading (AHRS-provided, assumed accurate per Orca's public
        # nav spec) - this was previously mis-derived from goal_heading itself,
        # which made yaw_error always zero (the vehicle never turned).
        current_heading = np.arctan2(obs[3], obs[4])
        yaw_error = np.arctan2(np.sin(heading_cmd - current_heading), np.cos(heading_cmd - current_heading))
        cmd_yaw_rate_norm = np.clip(yaw_error / (self.p.max_yaw_rate * 10), -1, 1)

        speed_frac = 0.6  # fixed cruise-like speed fraction, no energy-awareness (matches baseline scope)
        cmd_speed_norm = 2 * speed_frac - 1  # maps to speed_frac * max_speed via env's action scaling
        cmd_depth_rate_norm = 0.0

        return np.array([cmd_speed_norm, cmd_yaw_rate_norm, cmd_depth_rate_norm])
