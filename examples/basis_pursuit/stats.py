"Create pydantic object for keeping track of statistics"

# from pydantic.dataclasses import dataclass
import numpy as np
import dataclasses

from dataclasses import dataclass

@dataclass
class OptimizationStats:
    matrix: np.ndarray
    measurements: np.ndarray
    x_vals: list[np.ndarray] =  dataclasses.field(default_factory=list) # List of x values per iteration
    x_p_vals: list[np.ndarray] =  dataclasses.field(default_factory=list) # List of x_p values per iteration
    residuals: list[float] = dataclasses.field(default_factory=list)

    def compute_residuals(self):
        """Compute and store residuals ||x - x_p|| for all iterations."""
        self.residuals = [np.linalg.norm(x - x_p) for x, x_p in zip(self.x_vals, self.x_p_vals)]
    
    def add_iteration(self, x: np.ndarray, x_p: np.ndarray):
        """Add a new iteration's x and x_p values and compute residual."""
        self.x_vals.append(x)
        self.x_p_vals.append(x_p)
        self.residuals.append(np.linalg.norm(x - x_p))

    def latest_residual(self) -> float:
        """Return the most recent residual value."""
        return self.residuals[-1] if self.residuals else None
    
    def metrics(self) -> dict[np.ndarray]:
        num_iters = len(self.x_vals)
        residuals = np.zeros((num_iters))
        for i in range(num_iters):
            residuals[i] = np.linalg.norm(self.x_vals[i] - self.x_p_vals[i])
        
        vals = {"res": residuals, "obj": residuals, "viol": residuals}
        return vals