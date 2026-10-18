"""
policy_network.py
-------------------
Minimal feed-forward policy network implemented in pure NumPy (no torch
available in this sandbox). Small enough (~2-3 hidden layers, <=64 units,
matching the technical proposal's stated architecture) that a from-scratch
implementation is entirely tractable and fast to evaluate.
"""

import numpy as np


class MLPPolicy:
    def __init__(self, obs_dim, act_dim, hidden=(64, 64), seed=0):
        rng = np.random.default_rng(seed)
        sizes = [obs_dim] + list(hidden) + [act_dim]
        self.sizes = sizes
        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            w = rng.normal(0, np.sqrt(2.0 / fan_in), size=(sizes[i], sizes[i + 1]))
            b = np.zeros(sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)

    def get_flat_params(self):
        parts = []
        for w, b in zip(self.weights, self.biases):
            parts.append(w.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)

    def set_flat_params(self, flat):
        idx = 0
        for i in range(len(self.weights)):
            w_shape = self.weights[i].shape
            b_shape = self.biases[i].shape
            w_size = int(np.prod(w_shape))
            b_size = int(np.prod(b_shape))
            self.weights[i] = flat[idx: idx + w_size].reshape(w_shape)
            idx += w_size
            self.biases[i] = flat[idx: idx + b_size].reshape(b_shape)
            idx += b_size

    def n_params(self):
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))

    def forward(self, obs):
        x = obs
        n_layers = len(self.weights)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < n_layers - 1:
                x = np.tanh(x)
        return np.tanh(x)  # output actions squashed to [-1, 1]
