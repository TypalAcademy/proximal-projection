"""Implement each method for benchmarking"""

import time

import numpy as np

from proximal_projection.examples.basis_pursuit.stats import OptimizationStats


def shrink(xi: np.ndarray, alpha: float) -> np.ndarray:
    """The proximal for $|$x$|_1$ is an element-wise shrink operation."""
    output: np.ndarray = np.sign(xi) * np.maximum(np.abs(xi) - alpha, 0)
    return output


def proximal_projection(A, b, alpha=1.0e-1, num_iters=2000) -> tuple[np.ndarray, OptimizationStats]:
    """Proximal Projection"""
    stats = OptimizationStats(matrix=A, measurements=b)
    start_time = time.time()
    z = np.zeros((A.shape[1], 1))
    x = np.zeros((A.shape[1], 1))
    AAt = np.linalg.inv(A @ A.T)
    for _ in range(num_iters):
        x_p = x.copy()
        x = z - A.T @ (AAt @ (A @ z - b))
        z = z + shrink(2.0 * x - z, alpha) - x
        stats.add_iteration(x, x_p)
    stats.execution_time = time.time() - start_time
    return x, stats


def linearized_bregman(A, b, mu=2.0, num_iters=2000) -> tuple[np.ndarray, OptimizationStats]:
    """Linearized Bregman"""
    stats = OptimizationStats(matrix=A, measurements=b)
    start_time = time.time()
    x = np.zeros((A.shape[1], 1))
    v = np.zeros((A.shape[1], 1))
    matrix_norm = np.linalg.norm(A @ A.T)
    alpha = 2.0 / matrix_norm
    mu *= matrix_norm
    for _ in range(num_iters):
        x_p = x.copy()
        v = v - A.T @ (A @ x - b)
        x = shrink(alpha * v, alpha * mu)
        stats.add_iteration(x, x_p)
    stats.execution_time = time.time() - start_time
    return x, stats


def linearized_method_multipliers(
    A, b, lambd=100.0, num_iters=2000
) -> tuple[np.ndarray, OptimizationStats]:
    """Linearized Method of Multipliers"""
    stats = OptimizationStats(matrix=A, measurements=b)
    start_time = time.time()
    rows, cols = A.shape
    x = np.zeros((cols, 1))
    v = np.zeros((rows, 1))
    lambd *= np.linalg.norm(A.T @ A)
    alpha = 1.0 / (lambd * np.linalg.norm(A.T @ A))
    for _ in range(num_iters):
        x_p = x.copy()
        x = shrink(x - alpha * A.T @ (v + lambd * (A @ x - b)), alpha)
        v = v + lambd * (A @ x - b)
        stats.add_iteration(x, x_p)
    stats.execution_time = time.time() - start_time
    return x, stats


def primal_dual_hybrid_gradient(
    A, b, lambd=100.0, num_iters=2000
) -> tuple[np.ndarray, OptimizationStats]:
    """Primal Dual Hybrid Gradient"""
    stats = OptimizationStats(matrix=A, measurements=b)
    start_time = time.time()
    rows, cols = A.shape
    x = np.zeros((cols, 1))
    v = np.zeros((rows, 1))
    alpha = 1.0 / (lambd * np.linalg.norm(A.T @ A))
    for _ in range(num_iters):
        x_p = x.copy()
        x = shrink(x - alpha * A.T @ v, alpha)
        v = v + lambd * (A @ (2 * x - x_p) - b)
        stats.add_iteration(x, x_p)
    stats.execution_time = time.time() - start_time
    return x, stats
