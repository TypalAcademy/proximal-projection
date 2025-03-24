"""Script to run basis pursuit experiments"""

import numpy as np 
import matplotlib.pyplot as plt

from examples.basis_pursuit.methods import (
    proximal_projection,
    linearized_bregman,
    linearized_method_multipliers,
    primal_dual_hybrid_gradient,
)


def run_basis_pursuit_experiment(seeds=1, iters=2000, m=500, n=2000) -> None:
    """Execute numerical experiment for basis pursuit"""
    methods = {"pp": proximal_projection, 
               "lmm": linearized_method_multipliers,
               "lb": linearized_bregman,
               "pdhg": primal_dual_hybrid_gradient}
    metrics = ["obj", "res", "viol", "time"]

    exp_stats = {name: {} for name in methods.keys()}
    for method in exp_stats.keys():
        for metric in metrics:
            exp_stats[method][metric] = np.zeros((seeds, 1)) if metric == "time" else np.zeros((seeds, iters))

    print('Beginning numerical experiment')
    for seed in range(seeds):
        print(f"seed = {(seed+1):4} of {seeds}")
        np.random.seed(seed)
        A = np.random.normal(0, 1.0 / m, size=(m, n))
        x = np.random.normal(0, 1.0, size=(n, 1)) * np.random.binomial(n=1, p=0.05, size=(n, 1))
        b = A @ x

        for method_name, method in methods.items():
            stats = method(A, b, num_iters=iters)[1]
            for metric in metrics:
                exp_stats[method_name][metric][seed, :] = stats.metrics()[metric]

    print('Computing summary statistics')
    summary = {name: {} for name in methods.keys()}
    for method_name, method in methods.items():
        for metric in metrics:
            summary[method_name][metric] = np.median(exp_stats[method_name][metric], axis=0)

    # print('Making plots')
    # fig, ax = plt.subplots()
    # plt.title("Violation $|Ax-b|$")
    # plt.plot(viol_pp, color="b")
    # plt.plot(viol_lb, color="g")
    # plt.plot(viol_lmm, color="k")
    # plt.plot(viol_pdhg, color="r")
    # plt.yscale("log")
    # plt.show()

    # fig, ax = plt.subplots()
    # plt.title("Objective |x^k|_1")
    # plt.plot(obj_pp, color="b")
    # plt.plot(obj_lb, color="g")
    # plt.plot(obj_lmm, color="k")
    # plt.plot(obj_pdhg, color="r")
    # plt.show()

    # fig, ax = plt.subplots()
    # plt.title("Residual $|x^{k+1}-x^k|$")
    # plt.plot(res_pp, color="b")
    # plt.plot(res_lb, color="g")
    # plt.plot(res_lmm, color="k")
    # plt.plot(res_pdhg, color="r")
    # plt.yscale("log")
    # plt.show()    

    print('Writing summary statistics to file')
    for metric in ["obj", "res", "viol"]:
        filename = f"bp-{metric}-plots.csv"
        with open(filename, "w") as csv_file:
            for k in range(iters): 
                vals = [f'{float(summary[name][metric][k]):0.5e}' for name in methods]
                msg = ",".join(vals) + "\n"
                csv_file.write(msg)        

    with open("bp-times.tex", "w") as csv_file:
        csv_file.write(f"\\def\\bpTimePP{{{float(summary["pp"]["time"]):0.2f}}}\n")
        csv_file.write(f"\\def\\bpTimeLB{{{float(summary["lb"]["time"]):0.2f}}}\n")
        csv_file.write(f"\\def\\bpTimePDHG{{{float(summary["pdhg"]["time"]):0.2f}}}\n")
        csv_file.write(f"\\def\\bpTimeLMM{{{float(summary["lmm"]["time"]):0.2f}}}")           

if __name__ == "__main__":
    run_basis_pursuit_experiment()
