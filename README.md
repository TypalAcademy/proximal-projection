[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![Python Linting Badge](https://github.com/typalacademy/proximal-projection-algorithm/actions/workflows/python-linting.yml/badge.svg)](https://github.com/TypalAcademy/proximal-projection-algorithm/blob/main/.github/workflows/python-linting.yml)
[![Docs Website Badge](https://github.com/typalacademy/proximal-projection-algorithm/actions/workflows/docs-website.yml/badge.svg)](https://proximal-projection.research.typal.academy)

# Proximal Projection Algorithm
Code for the paper [_Proximal Projection Method for Stable Linearly Constrained Optimization_](https://arxiv.org/abs/2407.16998) by [Howard Heaton](https://www.linkedin.com/in/howard-heaton/).

## Abstract

Many applications using large datasets require efficient methods for minimizing a proximable convex function subject to satisfying a set of linear constraints within a specified tolerance. For this task, we present a proximal projection (PP) algorithm, which is an instance of Douglas-Rachford splitting that directly uses projections onto the set of constraints. Formal guarantees are presented to prove convergence of PP estimates to optimizers. Unlike many methods that obtain feasibility asymptotically, each PP iterate is feasible. Numerically, we show PP either matches or outperforms alternatives (e.g. linearized Bregman, primal dual hybrid gradient, proximal augmented Lagrangian, proximal gradient) on problems in basis pursuit, stable matrix completion, stable principal component pursuit, and the computation of earth mover’s distances.

<br>

## Usage Overview

### Setup
On a Mac with homebrew installed, one can run the following to install poetry.[^poetryInstall]
```
brew install poetry
```
Next clone the repository and go to the directory containing this repository's code.
```
git clone https://github.com/typalacademy/proximal-projection-algorithm.git
cd proximal-projection-algorithm
```
Lastly, create and activate a virtual environment for the project.
```
poetry install
```

[^poetryInstall]: For more installation options, see the [Poetry documentation](https://python-poetry.org/docs/).

### Experiments

To run each experiment, execute the following.

```
poetry run experiment_basis_pursuit
poetry run experiment_principal_component_pursuit
poetry run experiment_earth_mover_distance
poetry run experiment_matrix_completion
```

See the [documentation website](https://pp.research.typal.academy) for more code details.

<br>


## Citation
    
    @article{heaton2024proximal,
             title={{Proximal Projection Method for Stable Linearly Constrained Optimization}},
             author={Heaton, Howard},
             journal={{arXiv preprint}},
             year={2024}
    }
