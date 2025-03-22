"""Script to run basis pursuit experiments"""

import numpy as np

from src.basis_pursuit.methods import (
    proximal_projection,
    linearized_bregman,
    linearized_method_multipliers,
    prial_dual_hybrid_gradient,
)


def run_basis_pursuit_experiment() -> None:
    """Dummy function"""
    viol = np.zeros((4, seeds, iters))
    obj = np.zeros((4, seeds, iters))
    res = np.zeros((4, seeds, iters))
    times = np.zeros((4, seeds))

    for seed in range(seeds):
        print("seed = ", str(seed + 1), " of ", seeds)
        np.random.seed(seed)
        A = np.random.normal(0, 1.0 / m, size=(m, n))
        x = np.random.normal(0, 1.0, size=(n, 1)) * np.random.binomial(n=1, p=0.05, size=(n, 1))
        b = A @ x

        _, viol[0, seed, :], obj[0, seed, :], res[0, seed, :], times[0, seed] = PP(
            A, b, num_iters=iters
        )
        _, viol[1, seed, :], obj[1, seed, :], res[1, seed, :], times[1, seed] = LMM(
            A, b, num_iters=iters
        )
        _, viol[2, seed, :], obj[2, seed, :], res[2, seed, :], times[2, seed] = LB(
            A, b, num_iters=iters
        )
        _, viol[3, seed, :], obj[3, seed, :], res[3, seed, :], times[3, seed] = PDHG(
            A, b, num_iters=iters
        )

    viol_pp = np.median(viol[0, :, :], axis=0)
    viol_lmm = np.median(viol[1, :, :], axis=0)
    viol_lb = np.median(viol[2, :, :], axis=0)
    viol_pdhg = np.median(viol[3, :, :], axis=0)

    obj_pp = np.median(obj[0, :, :], axis=0)
    obj_lmm = np.median(obj[1, :, :], axis=0)
    obj_lb = np.median(obj[2, :, :], axis=0)
    obj_pdhg = np.median(obj[3, :, :], axis=0)

    res_pp = np.median(res[0, :, :], axis=0)
    res_lmm = np.median(res[1, :, :], axis=0)
    res_lb = np.median(res[2, :, :], axis=0)
    res_pdhg = np.median(res[3, :, :], axis=0)

    times_pp = np.mean(times[0, :])
    times_lmm = np.mean(times[1, :])
    times_lb = np.mean(times[2, :])
    times_pdhg = np.mean(times[3, :])

    print("PP   median time = %0.2f s" % times_pp)
    print("LMM  median time = %0.2f s" % times_lmm)
    print("LB   median time = %0.2f s" % times_lb)
    print("PDHG median time = %0.2f s" % times_pdhg)


if __name__ == "__main__":
    run_basis_pursuit_experiment()
