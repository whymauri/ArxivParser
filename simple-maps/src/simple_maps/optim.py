"""Minimal optimizers for full-batch embedding objectives."""

from __future__ import annotations

import numpy as np


class Adam:
    """Standard Adam on a single dense parameter array."""

    def __init__(
        self,
        shape: tuple[int, ...],
        lr: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-7,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, grad: np.ndarray) -> np.ndarray:
        """Return the parameter update (to be added) for this gradient."""
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad * grad
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
