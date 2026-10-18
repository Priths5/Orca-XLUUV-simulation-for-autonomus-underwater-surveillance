"""
ocean_environment.py
---------------------
Generates a procedurally-structured (but reproducible) ocean mission area:
  - Bathymetric field (seafloor depth) with realistic spatial correlation,
    used for passive terrain-aided navigation. Real GEBCO bathymetry could
    be substituted here in place of the procedural field for a higher-
    fidelity follow-on study.
  - A time-varying current field (structured, not pure noise - representing
    a persistent regional current with local eddies), standing in for real
    HYCOM/Copernicus current data.
  - A detection-risk map: representing shallow water, shipping lanes, and
    likely sensor coverage as a scalar risk field in [0, 1].
"""

import numpy as np


class OceanEnvironment:
    def __init__(self, size_m=20000.0, grid_res=200, seed=0):
        self.size_m = size_m
        self.grid_res = grid_res
        self.rng = np.random.default_rng(seed)
        self._build_bathymetry()
        self._build_risk_map()
        self._build_current_field(seed)

    # ---------------- Bathymetry ----------------
    def _build_bathymetry(self):
        n = self.grid_res
        # Sum of a few smooth random Fourier components -> spatially
        # correlated "terrain roughness" field, analogous to real seafloor
        # magnetic/bathymetric gradient structure used in TAN literature.
        xs = np.linspace(0, 2 * np.pi * 3, n)
        ys = np.linspace(0, 2 * np.pi * 3, n)
        X, Y = np.meshgrid(xs, ys)
        field = np.zeros((n, n))
        for k in range(1, 6):
            amp = 1.0 / k
            phase_x = self.rng.uniform(0, 2 * np.pi)
            phase_y = self.rng.uniform(0, 2 * np.pi)
            field += amp * np.sin(k * X + phase_x) * np.cos(k * Y + phase_y)
        field = (field - field.min()) / (field.max() - field.min())
        # Scale to plausible ocean depths: 200m (shallow shelf) - 4000m (abyssal)
        self.bathy_grid = 200 + field * 3800.0
        # Local gradient magnitude -> "information richness" for TAN
        gy, gx = np.gradient(self.bathy_grid)
        self.gradient_grid = np.sqrt(gx ** 2 + gy ** 2)
        self.gradient_grid /= self.gradient_grid.max()

    def _build_risk_map(self):
        n = self.grid_res
        # Risk is higher in shallow water (correlate inversely with depth)
        depth_norm = (self.bathy_grid - self.bathy_grid.min()) / (
            self.bathy_grid.max() - self.bathy_grid.min())
        shallow_risk = 1.0 - depth_norm  # shallower -> higher risk
        # Add a couple of "shipping lane" risk bands
        lane_risk = np.zeros((n, n))
        for _ in range(2):
            row = self.rng.integers(int(n * 0.2), int(n * 0.8))
            width = self.rng.integers(4, 9)
            lane_risk[max(0, row - width):row + width, :] += 0.6
        self.risk_grid = np.clip(0.5 * shallow_risk + lane_risk, 0, 1)

    def _build_current_field(self, seed):
        # A slowly-varying regional current, direction/strength change smoothly
        # with a periodic time component (tidal-like) plus small random eddies.
        self.current_seed = seed
        self.current_base_dir = self.rng.uniform(0, 2 * np.pi)
        self.current_base_speed = self.rng.uniform(0.15, 0.45)  # m/s, realistic ocean current

    def current_at(self, x, y, t):
        """Returns (u, v) current vector (m/s) at world position (x,y), time t (s)."""
        tidal = 0.5 * np.sin(2 * np.pi * t / (12.4 * 3600))  # semi-diurnal-like modulation
        eddy_x = 0.05 * np.sin(x / 1500.0 + t / 5000.0)
        eddy_y = 0.05 * np.cos(y / 1500.0 - t / 4000.0)
        speed = max(0.02, self.current_base_speed * (1 + 0.3 * tidal))
        u = speed * np.cos(self.current_base_dir) + eddy_x
        v = speed * np.sin(self.current_base_dir) + eddy_y
        return u, v

    # ---------------- Grid sampling helpers ----------------
    def _to_grid_idx(self, x, y):
        gx = int(np.clip((x / self.size_m + 0.5) * self.grid_res, 0, self.grid_res - 1))
        gy = int(np.clip((y / self.size_m + 0.5) * self.grid_res, 0, self.grid_res - 1))
        return gy, gx

    def depth_at(self, x, y):
        gy, gx = self._to_grid_idx(x, y)
        return float(self.bathy_grid[gy, gx])

    def depth_at_array(self, xs, ys):
        """Vectorized depth lookup for an array of positions (fast path for
        particle-filter updates, avoiding per-particle Python overhead)."""
        n = self.grid_res
        gx = np.clip(((xs / self.size_m + 0.5) * n).astype(np.int64), 0, n - 1)
        gy = np.clip(((ys / self.size_m + 0.5) * n).astype(np.int64), 0, n - 1)
        return self.bathy_grid[gy, gx]

    def gradient_at(self, x, y):
        gy, gx = self._to_grid_idx(x, y)
        return float(self.gradient_grid[gy, gx])

    def risk_at(self, x, y):
        gy, gx = self._to_grid_idx(x, y)
        return float(self.risk_grid[gy, gx])
