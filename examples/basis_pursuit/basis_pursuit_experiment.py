"""Script to run basis pursuit experiments"""

import numpy as np 

from examples.basis_pursuit.methods import (
    proximal_projection,
    linearized_bregman,
    linearized_method_multipliers,
    primal_dual_hybrid_gradient,
)

import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set level to capture all messages

# Create a file handler
file_handler = logging.FileHandler("experiment_basis_pursuit.log")  # Log messages will be saved here
file_handler.setLevel(logging.DEBUG)  # Capture debug and above messages

# Create a formatter for file output
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)    



def run_basis_pursuit_experiment(seeds=2, iters=2000, m=500, n=2000) -> None:
    """Dummy function"""
    methods = {"pp": proximal_projection, 
               "lmm": linearized_method_multipliers,
               "lb": linearized_bregman,
               "pdhg": primal_dual_hybrid_gradient}
    metrics = ["obj", "res", "viol"]

    exp_stats = {name: {} for name in methods.keys()}
    for method in exp_stats.keys():
        for metric in metrics:
            exp_stats[method][metric] = np.zeros((seeds, iters))

    for seed in range(seeds):
        print("seed = ", str(seed + 1), " of ", seeds)
        np.random.seed(seed)
        A = np.random.normal(0, 1.0 / m, size=(m, n))
        x = np.random.normal(0, 1.0, size=(n, 1)) * np.random.binomial(n=1, p=0.05, size=(n, 1))
        b = A @ x

        for method_name, method in methods.items():
            stats = method(A, b, num_iters=iters)[1]

            logger.info(f'executed method {method_name}')
            
            # print('stats = \n', stats.metrics())
            for metric in metrics:
                exp_stats[method_name][metric][seed, :] = stats.metrics()[metric]

if __name__ == "__main__":
    run_basis_pursuit_experiment()
