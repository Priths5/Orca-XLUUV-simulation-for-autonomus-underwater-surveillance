"""
world_model.py
--------------
Stand-in for sim/ocean_environment.OceanEnvironment when the vehicle is in
Stonefish. It exposes the same interface the existing, unchanged modules
call -

    ParticleFilterLocalizer -> depth_at_array(xs, ys)
    EnvironmentHealth       -> current_at(x, y, t), gradient_at(x, y), risk_at(x, y)

- but the values come from (a) what is legitimately known before the dive
(the chart: seabed depth of the scenario, the scenario's current) and
(b) what the Orca's own sonars measured most recently (gradient richness
from the Multibeam, risk from the Profiler + DVL altitude). Obstacle
positions are never put in here.

Note on terrain-aided navigation: every Stonefish scenario in this project
has a FLAT seabed plane, so bathymetric matching carries no position
information and the particle filter correctly falls back to dead
reckoning (uncertainty grows). Swap `seabed_depth` for a real chart grid to
make TAN informative again.
"""

import numpy as np


class StonefishOceanModel:
    def __init__(self, seabed_depth, current_ne=(0.0, 0.0)):
        self.seabed_depth = float(seabed_depth)
        self.current_ne = (float(current_ne[0]), float(current_ne[1]))
        self._gradient = 0.0
        self._risk = 0.0

    # ---- chart / scenario knowledge ----
    def depth_at(self, x, y):
        return self.seabed_depth

    def depth_at_array(self, xs, ys):
        return np.full(np.shape(xs), self.seabed_depth, dtype=float)

    def current_at(self, x, y, t):
        return self.current_ne

    # ---- latest onboard measurements (valid at the vehicle's position) ----
    def set_measurements(self, gradient_richness=None, risk=None):
        if gradient_richness is not None:
            self._gradient = float(np.clip(gradient_richness, 0.0, 1.0))
        if risk is not None:
            self._risk = float(np.clip(risk, 0.0, 1.0))

    def gradient_at(self, x, y):
        return self._gradient

    def risk_at(self, x, y):
        return self._risk
