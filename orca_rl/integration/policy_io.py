"""
policy_io.py
------------
Loads a trained policy .npz into the existing rl/policy_network.MLPPolicy
(architecture untouched) and tells the caller which observation layout and
vehicle limits it was trained with.

  logs/trained_policy.npz           theta, sizes                -> "legacy12", OrcaVehicleParams
  logs/trained_policy_obstacle.npz  theta, sizes, obs_layout,
                                    params_class                -> "obstacle_v1", StonefishOrcaParams
"""

import json

import numpy as np

from rl.policy_network import MLPPolicy
from sim.vehicle_dynamics import OrcaVehicleParams, StonefishOrcaParams


def load_policy(path):
    data = np.load(path, allow_pickle=False)
    theta = data["theta"]
    sizes = [int(s) for s in data["sizes"]]
    layout = str(data["obs_layout"]) if "obs_layout" in data.files else "legacy12"
    params_name = str(data["params_class"]) if "params_class" in data.files else "OrcaVehicleParams"
    meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
    policy = MLPPolicy(sizes[0], sizes[-1], hidden=tuple(sizes[1:-1]))
    policy.set_flat_params(theta)
    params = StonefishOrcaParams() if params_name == "StonefishOrcaParams" else OrcaVehicleParams()
    return policy, layout, params, meta
