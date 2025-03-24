"""Script to run basis pursuit experiments"""

import os
import subprocess
from typing import Any, Callable

import numpy as np

from proximal_projection.examples.basis_pursuit.methods import (
    linearized_bregman,
    linearized_method_multipliers,
    primal_dual_hybrid_gradient,
    proximal_projection,
)


def write_stats_to_file(exp_stats, method_names, metrics, iters) -> None:
    """Save summary statistics of each method to file"""
    print("Computing summary statistics")
    summary: dict[str, Any] = {name: {} for name in method_names}
    for method_name in method_names:
        for metric in metrics:
            summary[method_name][metric] = np.median(exp_stats[method_name][metric], axis=0)

    print("Writing summary statistics to file")
    for metric in ["obj", "res", "viol"]:
        filename = f"./examples/basis_pursuit/bp-{metric}-plots.csv"
        with open(filename, "w", encoding="utf-8") as csv_file:
            for k in range(iters):
                vals = [f"{float(summary[name][metric][k]):0.5e}" for name in method_names]
                msg = ",".join(vals)
                if k < iters - 1:
                    msg += "\n"
                csv_file.write(msg)

    with open("./examples/basis_pursuit/bp-times.tex", "w", encoding="utf-8") as csv_file:
        csv_file.write("\\def\\bpTimePP{{ {:0.2f} }}\n".format(float(summary["pp"]["time"])))
        csv_file.write("\\def\\bpTimeLB{{ {:0.2f} }}\n".format(float(summary["lb"]["time"])))
        csv_file.write("\\def\\bpTimePDHG{{ {:0.2f} }}\n".format(float(summary["pdhg"]["time"])))
        csv_file.write("\\def\\bpTimeLMM{{ {:0.2f} }}".format(float(summary["lmm"]["time"])))


def compile_latex() -> None:
    """Generate .pdf plots using LaTex"""
    print("Compiling LaTex plots")
    latex_dir = "./examples/basis_pursuit/"
    latex_file = "bp-plots.tex"
    tex_dir = os.path.dirname(os.path.abspath(latex_dir + latex_file))

    # Run pdflatex in the same directory as the .tex file, hiding pdflatex terminal outputs
    with open(os.devnull, "w", encoding="utf-8") as FNULL:
        subprocess.run(
            ["pdflatex", latex_file], stdout=FNULL, stderr=subprocess.PIPE, cwd=tex_dir, check=True
        )
    print(f"PDF with basis pursuit plots saved in: {tex_dir}")


def sample_initialization(seed, m=500, n=2000) -> tuple[np.array, np.array]:
    """Randomly pick matrix A and vector b using random seed"""
    np.random.seed(seed)
    A = np.random.normal(0, 1.0 / m, size=(m, n))
    x = np.random.normal(0, 1.0, size=(n, 1)) * np.random.binomial(n=1, p=0.05, size=(n, 1))
    b = A @ x
    return A, b


def run_basis_pursuit_experiment(seeds=5, iters=2000) -> None:
    """Execute numerical experiment for basis pursuit"""
    methods: dict[str, Callable] = {
        "pp": proximal_projection,
        "lmm": linearized_method_multipliers,
        "lb": linearized_bregman,
        "pdhg": primal_dual_hybrid_gradient,
    }
    metrics = ["obj", "res", "viol", "time"]

    exp_stats: dict[str, dict] = {name: {} for name in methods}
    for method in exp_stats.keys():
        for metric in metrics:
            exp_stats[method][metric] = (
                np.zeros((seeds, 1)) if metric == "time" else np.zeros((seeds, iters))
            )

    print("Beginning numerical experiment")
    for seed in range(seeds):
        print(f"seed = {(seed+1):4} of {seeds}")
        A, b = sample_initialization(seed)
        for method_name, method_func in methods.items():
            stats = method_func(A, b, num_iters=iters)[1]
            for metric in metrics:
                exp_stats[method_name][metric][seed, :] = stats.get_metrics()[metric]

    write_stats_to_file(exp_stats, methods.keys(), metrics, iters)
    compile_latex()


if __name__ == "__main__":
    run_basis_pursuit_experiment()
