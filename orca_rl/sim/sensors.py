"""
sensors.py
----------
Passive sensor suite matching Orca's real, publicly documented navigation
stack: Kalman-filtered INS + Doppler Velocity Log (DVL) + depth sensor.
No GPS, no active sonar (both would compromise stealth).

Noise magnitudes are set to representative commercial-grade navigation-sensor
specs for a vehicle of this class (exact Orca sensor models are not public).
Labeled [ESTIMATED] throughout.
"""

import numpy as np


class SensorParams:
    def __init__(self):
        # INS (inertial navigation) drift rate: typical tactical-grade INS
        # drifts on the order of ~0.1-1% of distance traveled without aiding.
        self.ins_drift_rate = 0.003          # [ESTIMATED] fraction of distance traveled, per axis

        # DVL velocity measurement noise (typical DVL: ~0.1-0.5% of velocity, +floor)
        self.dvl_noise_std = 0.02            # [ESTIMATED] m/s std
        self.dvl_dropout_prob = 0.01         # [ESTIMATED] chance of DVL bottom-lock loss per step (deep water)

        # Depth sensor (pressure-based): very accurate, standard spec ~0.01% FS
        self.depth_noise_std = 0.05          # [ESTIMATED] m std

        # Terrain-aided correction: bathymetric depth reading noise
        self.bathy_noise_std = 1.5           # [ESTIMATED] m std, coarser than direct depth sensor


class NavigationSensors:
    """
    Generates noisy sensor readings from TRUE state, simulating what the
    real onboard navigation computer would receive (never true position).
    """

    def __init__(self, params: SensorParams, dt=10.0, rng=None):
        self.p = params
        self.dt = dt
        self.rng = rng or np.random.default_rng()
        self.ins_bias_x = 0.0
        self.ins_bias_y = 0.0

    def reset(self):
        self.ins_bias_x = 0.0
        self.ins_bias_y = 0.0

    def measure(self, true_state, distance_step):
        """Returns a dict of noisy sensor readings for this timestep."""
        p = self.p

        # INS drift accumulates with distance traveled (unaided dead-reckoning)
        drift_std = p.ins_drift_rate * max(distance_step, 0.1)
        self.ins_bias_x += self.rng.normal(0, drift_std)
        self.ins_bias_y += self.rng.normal(0, drift_std)

        dvl_dropout = self.rng.random() < p.dvl_dropout_prob
        dvl_speed = true_state.speed + (0.0 if dvl_dropout
                                         else self.rng.normal(0, p.dvl_noise_std))

        depth_reading = true_state.depth + self.rng.normal(0, p.depth_noise_std)

        return {
            "ins_bias_x": self.ins_bias_x,
            "ins_bias_y": self.ins_bias_y,
            "dvl_speed": dvl_speed,
            "dvl_dropout": dvl_dropout,
            "depth_reading": depth_reading,
            "heading": true_state.heading,  # AHRS-type heading, assumed accurate (public: INS provides orientation)
        }

    def bathymetry_reading(self, true_depth_at_location):
        return true_depth_at_location + self.rng.normal(0, self.p.bathy_noise_std)
