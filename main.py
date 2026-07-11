"""Run experiments 1-5 from arc_llc_context.md, print result tables, and save
plots to plots/ and a machine-readable summary to results/summary.json.
"""
import json
import time

import numpy as np

import experiments as ex
import plots


def main():
    t0 = time.time()
    all_tables = []
    summary = {}

    print("Running Experiment 1 (ground truth d=2, true ratio=0.5)...")
    e1 = ex.experiment_1()
    print(ex.format_table(e1))
    plots.plot_volume_scaling(e1, "plots/exp1_volume_scaling.png")
    plots.plot_sgld_free_energy(e1, "plots/exp1_sgld.png")
    all_tables.append(ex.format_table(e1))
    summary["experiment_1"] = {"rows": e1["rows"], "lambda_true": e1["lambda_true"],
                                "ratio_true": e1["ratio_true"]}

    print("\nRunning Experiment 2 (ground truth d=4, true ratio=0.5)...")
    e2 = ex.experiment_2()
    print(ex.format_table(e2))
    plots.plot_volume_scaling(e2, "plots/exp2_volume_scaling.png")
    plots.plot_sgld_free_energy(e2, "plots/exp2_sgld.png")
    all_tables.append(ex.format_table(e2))
    summary["experiment_2"] = {"rows": e2["rows"], "lambda_true": e2["lambda_true"],
                                "ratio_true": e2["ratio_true"]}

    print("\nRunning Experiment 3 (regular model, true ratio=1.0)...")
    e3 = ex.experiment_3(d=4)
    print(ex.format_table(e3))
    plots.plot_volume_scaling(e3, "plots/exp3_volume_scaling.png")
    plots.plot_sgld_free_energy(e3, "plots/exp3_sgld.png")
    all_tables.append(ex.format_table(e3))
    summary["experiment_3"] = {"rows": e3["rows"], "lambda_true": e3["lambda_true"],
                                "ratio_true": e3["ratio_true"]}

    print("\nRunning Experiment 4 (training trajectory, ratio should drift 1 -> 0.5)...")
    e4 = ex.experiment_4()
    plots.plot_training_trajectory(e4, "plots/exp4_trajectory.png")
    traj = e4["trajectory"]
    print(f"  start ratio={traj[0]['ratio']:.3f} (dist={traj[0]['dist_to_origin']:.3f}), "
          f"end ratio={traj[-1]['ratio']:.3f} (dist={traj[-1]['dist_to_origin']:.3e})")
    summary["experiment_4"] = {
        "lambda_true": e4["lambda_true"], "ratio_true": e4["ratio_true"],
        "trajectory": [{"step": t["step"], "dist_to_origin": t["dist_to_origin"],
                         "ratio": t["ratio"]} for t in traj],
    }

    print("\nRunning Experiment 5 (arc direction distribution)...")
    e5 = ex.experiment_5()
    plots.plot_arc_direction(e5, "plots/exp5_arc_direction.png")
    print(f"  fitted scaling exponent for P(K(delta)<eps) ~ eps^exponent: {e5['exponent']:.4f}")
    summary["experiment_5"] = {"d": e5["d"], "exponent": e5["exponent"]}

    print("\nRunning Experiment 6 (asymmetric n!=m, Hessian multi-restart fix)...")
    e6 = ex.experiment_6()
    print(ex.format_table(e6))
    print(f"  codims observed across restarts: {e6['multi']['codims'].tolist()}")
    all_tables.append(ex.format_table(e6))
    summary["experiment_6"] = {"rows": e6["rows"], "lambda_true": e6["lambda_true"],
                                "ratio_true": e6["ratio_true"],
                                "codims": e6["multi"]["codims"].tolist()}

    print("\nRunning Experiment 7 (DDS validation on r=1 toy models)...")
    e7 = ex.experiment_7()
    plots.plot_dds_validation(e7, "plots/exp7_dds_validation.png")
    a = e7["analytic"]
    print(f"  analytic-limit: rho_structural={a['rho_structural']:.4f}, "
          f"slope_lam_h1={a['slope_lam_h1']:.3f} (predicted 2), "
          f"slope_sigma_h1={a['slope_sigma_h1']:.3f} (predicted 2)")
    print(f"  real trajectory: rho(lam_h1,sigma_h1)={e7['rho_structural_trajectory']:.4f}, "
          f"rho(h1,h2)={e7['rho_h1_h2_trajectory']:.4f} (both layers collapse together, r0=0)")
    summary["experiment_7"] = {
        "lambda_true": e7["lambda_true"],
        "rho_structural_analytic": a["rho_structural"],
        "slope_lam_h1": a["slope_lam_h1"], "slope_sigma_h1": a["slope_sigma_h1"],
        "rho_structural_trajectory": e7["rho_structural_trajectory"],
        "rho_h1_h2_trajectory": e7["rho_h1_h2_trajectory"],
    }

    print("\nRunning Experiment 8 (DDS cross-cell rank-tracking, Aoyagi 2005 anchor)...")
    e8 = ex.experiment_8()
    plots.plot_dds_cross_cell(e8, "plots/exp8_dds_cross_cell.png")
    print("  Cross-cell Spearman rho vs true lambda:")
    for name, rho in e8["cross_cell_rho"].items():
        print(f"    {name}: {rho:.3f}")
    summary["experiment_8"] = {
        "cross_cell_rho": e8["cross_cell_rho"],
        "n_cells": len(e8["cells"]),
    }

    print("\nRunning Experiment 9 (deep-linear noisy bridge, rank-multiplicative counting identity)...")
    e9 = ex.experiment_9()
    plots.plot_deep_linear_counting(e9, "plots/exp9_deep_linear_counting.png")
    print("  Slope ratio vs r=1 (predicted log_det_plus=r, lambda_plus_min=1):")
    for r, s in e9["ratio_summary"].items():
        print(f"    r={r}: log_det_plus_ratio={s['log_det_plus_ratio_mean']:.4f} "
              f"+/- {s['log_det_plus_ratio_std']:.4f}, "
              f"lambda_plus_min_ratio={s['lambda_plus_min_ratio_mean']:.4f} "
              f"+/- {s['lambda_plus_min_ratio_std']:.4f}")
    summary["experiment_9"] = {"ratio_summary": e9["ratio_summary"], "Ls": e9["Ls"], "rs": e9["rs"]}

    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))

    with open("results/tables.md", "w") as f:
        f.write("\n\n".join(all_tables))

    print(f"\nDone in {time.time()-t0:.1f}s. Plots in plots/, tables in results/tables.md, "
          f"raw summary in results/summary.json")


if __name__ == "__main__":
    main()
