---
title: Home
description: Documentation for code used in the paper for the Proximal Projection (PP) algorithm.

---

# Basis Pursuit

This page shows the numerical example for solving the problem

$$
    \mathsf{ \underset{x}{min} \ |x|_1 \quad s.t. \quad Ax=b. }
$$

::: src.basis_pursuit.methods.shrink
    options:
      show_root_heading: false

## Methods

The update iteration for each benchmarked algorithm is in Appendix B.1 of the [paper](https://arxiv.org/abs/2407.16998). Code implementations are shown below.

::: examples.basis_pursuit.methods.proximal_projection
    options:
      show_root_heading: false

::: examples.basis_pursuit.methods.linearized_bregman
    options:
      show_root_heading: false

::: examples.basis_pursuit.methods.linearized_method_multipliers
    options:
      show_root_heading: false

::: examples.basis_pursuit.methods.primal_dual_hybrid_gradient
    options:
      show_root_heading: false

<br>

## Experiment

The matrix $\mathsf{A \in \mathbb{R}^{m\times n}}$ i.i.d. Gaussian entries, with $\mathsf{m=500}$ and $\mathsf{n=2000}$. Elements of a sparse vector $\mathsf{x^\star}$ are independently nonzero with probability $\mathsf{p = 0.05}$ and the nonzero values are i.i.d. Gaussian. This is used to compute $\mathsf{b = Ax^\star}$. Ten trials are executed with distinct random seeds. The mean time for 10 trials of proximal projection, linearized Bregman, primal dual hybrid gradient, and linearized method of multipliers to compute 2,000 iterations were, respectively, 8.06s, 7.40s, 20.60s, and 22.85s.

::: examples.basis_pursuit.basis_pursuit_experiment.run_basis_pursuit_experiment
    options:
      show_root_heading: false

<br>
