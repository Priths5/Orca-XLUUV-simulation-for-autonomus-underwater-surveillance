"""
navigation.py
-------------
Position estimate for the RL observation, in two explicitly separate modes:

  REALISTIC_SENSOR   (default, use for every real RL experiment)
      DVL body velocity (bottom lock) rotated by IMU heading -> dead-reckoned
      displacement -> the existing ParticleFilterLocalizer.predict();
      seabed depth measured as pressure depth + DVL altitude ->
      ParticleFilterLocalizer.update_with_bathymetry();
      GPS fix (only possible at the surface) -> re-centre the filter.
      If the DVL loses bottom lock, the last DVL velocity is held for a few
      seconds, then the commanded speed is used (a propeller-speed log),
      with inflated process noise.

  DEBUG_GROUND_TRUTH (connectivity checks only)
      Stonefish Odometry pose (/orca/ground_truth/odometry) is used as the
      estimate. Every observation built in this mode is flagged; the bridge
      prints a warning once a second so it cannot be used by accident.
"""

import math

import numpy as np

from sim.localization import ParticleFilterLocalizer
from .frames import body_to_ned_velocity

REALISTIC_SENSOR = "REALISTIC_SENSOR"
DEBUG_GROUND_TRUTH = "DEBUG_GROUND_TRUTH"


class StonefishNavigator:
    def __init__(self, ocean_model, mode=REALISTIC_SENSOR, n_particles=120, init_std=2.0,
                 bathy_noise_std=1.5, rng=None):
        if mode not in (REALISTIC_SENSOR, DEBUG_GROUND_TRUTH):
            raise ValueError(f"unknown navigation mode {mode}")
        self.mode = mode
        self.rng = rng or np.random.default_rng()
        self.localizer = ParticleFilterLocalizer(ocean_model, n_particles=n_particles,
                                                 init_std=init_std, rng=self.rng)
        self.bathy_noise_std = bathy_noise_std
        self.acc_dx = 0.0
        self.acc_dy = 0.0
        self.dist_since_predict = 0.0
        self.dr_quality = 1.0           # 1 = DVL-aided, <1 = degraded
        self.gt_xy = None
        self.est = (0.0, 0.0)
        self.uncertainty = 2 * init_std ** 2
        self.last_dvl_uv = (0.0, 0.0)
        self.time_since_dvl = 1e9

    def reset(self, x0, y0):
        self.localizer.reset(x0, y0)
        self.acc_dx = self.acc_dy = self.dist_since_predict = 0.0
        self.est = (x0, y0)
        self.uncertainty = 2 * self.localizer.init_std ** 2
        self.time_since_dvl = 1e9

    # ---- high-rate integration (called on every IMU/DVL sample) ----
    def integrate(self, dt, yaw, dvl_uv=None, commanded_speed=0.0):
        if dt <= 0:
            return
        if dvl_uv is not None:
            u, v = dvl_uv
            self.last_dvl_uv = (u, v)
            self.time_since_dvl = 0.0
            self.dr_quality = 1.0
        else:
            self.time_since_dvl += dt
            if self.time_since_dvl < 5.0:
                u, v = self.last_dvl_uv
                self.dr_quality = 0.7
            else:
                u, v = commanded_speed, 0.0
                self.dr_quality = 0.3
        vn, ve = body_to_ned_velocity(u, v, yaw)
        self.acc_dx += vn * dt
        self.acc_dy += ve * dt
        self.dist_since_predict += math.hypot(vn, ve) * dt

    def set_ground_truth(self, x, y):
        self.gt_xy = (x, y)

    # ---- once per RL control step ----
    def update(self, measured_seabed_depth=None, gps_ne=None, gps_std=0.5):
        if self.mode == DEBUG_GROUND_TRUTH:
            if self.gt_xy is not None:
                self.est = self.gt_xy
            self.uncertainty = 1.0
            self.acc_dx = self.acc_dy = self.dist_since_predict = 0.0
            return self.est[0], self.est[1], self.uncertainty

        # same process-noise law as OrcaNavEnv.step(), scaled by DR quality
        noise = (0.5 + 0.5 * self.dist_since_predict / 50.0) / max(self.dr_quality, 0.1)
        self.localizer.predict(self.acc_dx, self.acc_dy, noise * 0.2)
        self.acc_dx = self.acc_dy = self.dist_since_predict = 0.0
        if measured_seabed_depth is not None and np.isfinite(measured_seabed_depth):
            self.localizer.update_with_bathymetry(measured_seabed_depth, self.bathy_noise_std)
        if gps_ne is not None:
            # A surface fix re-centres the filter around the fix.
            self.localizer.particles = np.stack([
                self.rng.normal(gps_ne[0], gps_std, self.localizer.n),
                self.rng.normal(gps_ne[1], gps_std, self.localizer.n)], axis=1)
            self.localizer.weights = np.ones(self.localizer.n) / self.localizer.n
        x, y, unc, _ = self.localizer.estimate()
        self.est = (float(x), float(y))
        self.uncertainty = float(unc)
        return self.est[0], self.est[1], self.uncertainty
