"""
localization.py
-----------------
Particle-filter based localization, consistent with standard practice in
the terrain-aided-navigation (TAN) literature. The vehicle never has access
to true position - only this filter's estimate + uncertainty, fed by noisy
dead-reckoning (INS/DVL) propagation and passive bathymetric-depth
correction updates.
"""

import numpy as np


class ParticleFilterLocalizer:
    def __init__(self, ocean_env, n_particles=300, init_std=50.0, rng=None):
        self.env = ocean_env
        self.n = n_particles
        self.rng = rng or np.random.default_rng()
        self.particles = None  # (n, 2) x,y
        self.weights = None
        self.init_std = init_std

    def reset(self, x0, y0):
        self.particles = np.stack([
            self.rng.normal(x0, self.init_std, self.n),
            self.rng.normal(y0, self.init_std, self.n),
        ], axis=1)
        self.weights = np.ones(self.n) / self.n

    def predict(self, dx, dy, process_noise_std):
        noise = self.rng.normal(0, process_noise_std, size=(self.n, 2))
        self.particles[:, 0] += dx + noise[:, 0]
        self.particles[:, 1] += dy + noise[:, 1]

    def update_with_bathymetry(self, measured_depth, bathy_noise_std):
        """Weight particles by how well their location's true bathymetry
        matches the noisy depth reading actually observed."""
        true_depths = self.env.depth_at_array(self.particles[:, 0], self.particles[:, 1])
        residual = measured_depth - true_depths
        likelihood = np.exp(-0.5 * (residual / bathy_noise_std) ** 2)
        self.weights *= likelihood + 1e-12
        self.weights /= self.weights.sum()
        # Resample if effective sample size too low
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.n / 2:
            self._resample()

    def _resample(self):
        idx = self.rng.choice(self.n, size=self.n, p=self.weights)
        self.particles = self.particles[idx]
        self.weights = np.ones(self.n) / self.n

    def estimate(self):
        mean = np.average(self.particles, axis=0, weights=self.weights)
        cov = np.cov(self.particles.T, aweights=self.weights)
        uncertainty = float(np.trace(cov))  # scalar uncertainty measure used in reward
        return mean[0], mean[1], uncertainty, cov
