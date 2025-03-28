"Create pydantic object for keeping track of statistics"

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class OptimizationStats:
    """Class for storing metrics to measure convergence of optimization schemes"""

    matrix: np.ndarray
    measurements: np.ndarray
    x_vals: list[np.ndarray] = field(default_factory=list)
    x_p_vals: list[np.ndarray] = field(default_factory=list)
    execution_time: float = 0.0

    def add_iteration(self, x: np.ndarray, x_p: np.ndarray):
        """Add a new iteration's x and x_p values and compute residual."""
        self.x_vals.append(x)
        self.x_p_vals.append(x_p)

    def get_metrics(self) -> dict[str, Any]:
        """Get numpy arrays of each metric"""
        num_iters = len(self.x_vals)
        residuals = np.zeros((num_iters))
        objectives = np.zeros((num_iters))
        violations = np.zeros((num_iters))
        for i in range(num_iters):
            residuals[i] = np.linalg.norm(self.x_vals[i] - self.x_p_vals[i])
            objectives[i] = np.linalg.norm(self.x_vals[i], ord=1)
            violations[i] = np.linalg.norm(self.matrix @ self.x_vals[i] - self.measurements)
        vals = {
            "res": residuals,
            "obj": objectives,
            "viol": violations,
            "time": self.execution_time,
        }
        return vals
