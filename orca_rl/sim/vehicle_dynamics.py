"""
vehicle_dynamics.py
--------------------
Simplified Fossen-style 4-DOF (surge, sway->heading-coupled, heave/depth, yaw)
marine vehicle dynamics model, parameterized with Boeing Orca XLUUV's real
publicly disclosed specifications. Where an exact figure is not public, a
clearly labeled engineering estimate is used, following standard AUV/UUV
hydrodynamic modeling practice (Fossen, "Handbook of Marine Craft
Hydrodynamics and Motion Control", 2011).

PUBLIC SPECS USED (Boeing / Navy / press, 2025-2026 sources):
  - Length (with payload section): 26 m
  - Displacement: ~85 metric tons
  - Beam/diameter: 2.6 m
  - Max speed: 8 knots (4.11 m/s); optimal cruise: 2.5-3 knots (~1.4 m/s)
  - Submerged propulsion power: ~18 kW
  - Depth rating: 3,400 m
  - Range: 6,500 nm
  - Navigation: Kalman-filtered INS + DVL + depth sensor (no GPS/active sonar)

ESTIMATED (not public, derived from comparable-class AUV/UUV hydrodynamics):
  - Drag coefficients (slender-body approximation)
  - Added mass coefficients
  - Battery usable capacity
"""

import numpy as np

G = 9.81
RHO_SEAWATER = 1025.0  # kg/m^3


class OrcaVehicleParams:
    """Physical parameters for the Boeing Orca XLUUV (public + labeled estimates)."""

    def __init__(self):
        # ---- PUBLIC ----
        self.length = 26.0                # m
        self.beam = 2.6                   # m (diameter)
        self.mass = 85_000.0              # kg (~85 metric tons)
        self.max_speed = 4.11             # m/s (8 knots)
        self.cruise_speed = 1.4           # m/s (~2.75 knots, optimal range)
        self.min_cruise_speed = 1.29      # m/s (2.5 knots)
        self.submerged_power_w = 18_000.0 # W (~18 kW)
        self.depth_rating = 3400.0        # m
        self.range_m = 6500 * 1852.0      # 6500 nm -> meters

        # ---- ESTIMATED [ESTIMATED] ----
        # Slender-body cylinder approximation for drag: Cd ~ 0.03-0.05 for a
        # streamlined torpedo-shaped AUV hull at this length/diameter ratio.
        self.Cd_axial = 0.04              # [ESTIMATED]
        self.Cd_lateral = 0.9             # [ESTIMATED] (broadside is bluff-body-like)
        frontal_radius = self.beam / 2.0
        self.A_frontal = np.pi * frontal_radius ** 2      # m^2, axial cross-section
        self.A_lateral = self.length * self.beam          # m^2, lateral profile area

        # Added mass (hydrodynamic inertia), estimated as a fraction of
        # displaced mass for a slender body per standard AUV modeling practice.
        self.added_mass_surge = 0.05 * self.mass          # [ESTIMATED]
        self.added_mass_sway = 0.85 * self.mass           # [ESTIMATED]
        self.added_mass_yaw_inertia = 0.30 * self.mass * (self.length ** 2) / 12  # [ESTIMATED]

        # Yaw damping / turning rate limit
        self.max_yaw_rate = np.deg2rad(3.0)               # [ESTIMATED] deg/s -> rad/s, slow large hull
        self.max_depth_rate = 0.3                          # [ESTIMATED] m/s vertical

        # Battery: usable energy sized so ~18 kW submerged draw is consistent
        # with multi-day submerged legs before a diesel-generator surface
        # recharge is needed (public: hybrid diesel-electric propulsion).
        self.battery_capacity_wh = 130_000.0               # [ESTIMATED] ~130 kWh usable
        self.hotel_load_w = 800.0                           # [ESTIMATED] sensors/computer baseline draw


class VehicleState:
    """Mutable simulation state (TRUE ground truth, not the estimate)."""

    def __init__(self, x=0.0, y=0.0, depth=50.0, heading=0.0, speed=0.0):
        self.x = x
        self.y = y
        self.depth = depth
        self.heading = heading  # radians
        self.speed = speed      # m/s, forward
        self.yaw_rate = 0.0
        self.battery_wh_remaining = None  # set by env using params

    def as_vector(self):
        return np.array([self.x, self.y, self.depth, self.heading, self.speed])


class OrcaDynamics:
    """
    Propagates true vehicle state given a commanded action and external
    disturbance (current vector), using drag/thrust-limited kinematics
    consistent with the Fossen surge/yaw/heave equations, simplified to
    a computationally light form suitable for RL training (thousands of
    episodes) while retaining the real physical scale of the vehicle.
    """

    def __init__(self, params: OrcaVehicleParams, dt=10.0):
        self.p = params
        self.dt = dt  # seconds per simulation step (10s -> minutes-scale missions tractable)

    def drag_force(self, speed):
        # F_d = 0.5 * rho * Cd * A * v^2  (axial drag opposing forward motion)
        return 0.5 * RHO_SEAWATER * self.p.Cd_axial * self.p.A_frontal * speed ** 2

    def max_thrust(self):
        # Available thrust from power limit at a nominal propulsive efficiency
        eff = 0.55  # [ESTIMATED] propeller efficiency
        # At cruise speed, P = F*v -> F = P*eff/v ; cap using cruise speed as reference
        return eff * self.p.submerged_power_w / max(self.p.cruise_speed, 0.5)

    def step(self, state: VehicleState, cmd_speed, cmd_yaw_rate, cmd_depth_rate,
             current_vec=(0.0, 0.0, 0.0)):
        """
        cmd_speed: desired forward speed (m/s), clipped to [0, max_speed]
        cmd_yaw_rate: desired yaw rate (rad/s), clipped to +-max_yaw_rate
        cmd_depth_rate: desired vertical rate (m/s), clipped
        current_vec: (u_curr, v_curr, w_curr) external current, m/s in world frame
        Returns: new VehicleState, energy_used_wh this step, power_w
        """
        p = self.p
        cmd_speed = float(np.clip(cmd_speed, 0.0, p.max_speed))
        cmd_yaw_rate = float(np.clip(cmd_yaw_rate, -p.max_yaw_rate, p.max_yaw_rate))
        cmd_depth_rate = float(np.clip(cmd_depth_rate, -p.max_depth_rate, p.max_depth_rate))

        # First-order speed response toward commanded speed (inertia-limited)
        tau_speed = 60.0  # [ESTIMATED] seconds time-constant for an 85-ton hull
        alpha = 1 - np.exp(-self.dt / tau_speed)
        new_speed = state.speed + alpha * (cmd_speed - state.speed)

        # Heading integrates yaw rate (first-order lag for a large hull)
        tau_yaw = 20.0  # [ESTIMATED]
        alpha_yaw = 1 - np.exp(-self.dt / tau_yaw)
        new_yaw_rate = state.yaw_rate + alpha_yaw * (cmd_yaw_rate - state.yaw_rate)
        new_heading = state.heading + new_yaw_rate * self.dt

        # Depth integrates depth rate directly (ballast-controlled, slower dynamics)
        new_depth = state.depth + cmd_depth_rate * self.dt
        new_depth = float(np.clip(new_depth, 0.0, p.depth_rating))

        # Position update: vehicle velocity in world frame + current disturbance
        vx = new_speed * np.cos(new_heading) + current_vec[0]
        vy = new_speed * np.sin(new_heading) + current_vec[1]
        new_x = state.x + vx * self.dt
        new_y = state.y + vy * self.dt

        # ---- Energy consumption model ----
        drag = self.drag_force(new_speed)
        power_propulsion = drag * new_speed / 0.55  # divide by estimated efficiency
        power_total = power_propulsion + p.hotel_load_w
        power_total = min(power_total, p.submerged_power_w * 1.2)  # allow small headroom
        energy_wh = power_total * self.dt / 3600.0

        new_state = VehicleState(new_x, new_y, new_depth, new_heading, new_speed)
        new_state.yaw_rate = new_yaw_rate
        return new_state, energy_wh, power_total


class StonefishOrcaParams(OrcaVehicleParams):
    """
    The Orca as it actually behaves in the Stonefish model
    (stonefish_sim/resources/data/orca_xluuv_survey.scn), identified with
    tools/sysid_stonefish.py (raw MainThruster/YawThruster steps):

        10 / 20 / 30 kN  ->  0.37 / 0.68 / 0.93 m/s  (40 kN limit -> ~1.15 m/s)
        surge time constant ~6.5 s open loop, ~5 s with the C++ speed loop
        yaw: r_ss ~= F_yaw / 150000; 8 kN YawThruster limit -> ~0.048 rad/s
        yaw-rate time constant < 1 s; depth held by HeaveThruster (+-55 kN)

    Everything else (mass, depth rating, battery, hotel load, drag model for
    the energy estimate) is inherited unchanged from OrcaVehicleParams. Use
    this for any policy that will drive the Stonefish Orca, so that the
    action -> setpoint scaling asks for things the vehicle can actually do.
    """

    def __init__(self):
        super().__init__()
        self.max_speed = 1.1                 # m/s  (MEASURED in Stonefish)
        self.cruise_speed = 0.8              # m/s
        self.min_cruise_speed = 0.3          # m/s, limp-home speed for the shield
        self.max_yaw_rate = 0.045            # rad/s (MEASURED, 8 kN YawThruster, ~0.5 m/s)
        self.max_depth_rate = 0.3            # m/s
        self.tau_speed = 5.0                 # s, closed-loop (C++ adapter) surge response
        self.tau_yaw = 1.5                   # s, closed-loop yaw-rate response
        self.tau_depth = 4.0                 # s, depth reference tracking
        # Stonefish hull: thrust needed to hold speed u, F = kq u|u| + kl u
        self.thrust_quad = 9000.0
        self.thrust_lin = 23500.0
        self.hull_half_len_fwd = 13.85       # nose tip ahead of hull origin (m)
        self.hull_half_len_aft = 13.30       # stern cap behind hull origin (m)
        self.hull_radius = 1.3

    def yaw_rate_limit(self, u):
        """Achievable yaw rate falls with speed (hull directional stability):
        MEASURED 0.048 rad/s at 0.49 m/s, 0.037 rad/s at 0.87 m/s (8 kN)."""
        return float(np.clip(0.0615 - 0.0280 * max(u, 0.0), 0.030, 0.050))

    def thrust_for_speed(self, u):
        return self.thrust_quad * u * abs(u) + self.thrust_lin * u
