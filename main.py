"""Run experiments 1-10, print result tables, and save
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

    print("\nRunning Experiment 8 (DDS cross-cell rank-tracking, SGLD posterior ensemble)...")
    e8 = ex.experiment_8()
    plots.plot_dds_cross_cell(e8, "plots/exp8_dds_cross_cell.png")
    print(f"  sampler LLC vs closed form: rho={e8['rho_llc']:.3f}, "
          f"rel. error {100*e8['llc_rel_err'].min():.1f}%..{100*e8['llc_rel_err'].max():.1f}%")
    print("  Cross-cell Spearman rho vs true lambda (95% bootstrap CI over chains):")
    for name, rho in e8["cross_cell_rho"].items():
        lo, hi = e8["cross_cell_rho_ci"][name]
        c, t = e8["within_H_concordance"][name]
        print(f"    {name}: {rho:.3f} [{lo:.3f}, {hi:.3f}]   within-H concordant pairs {c}/{t}")
    c, t = e8["within_H_concordance"]["llc"]
    print(f"    (sampler LLC within-H concordant pairs {c}/{t})")
    print("  Ablation (single constructed point), rho by perturbation scale:")
    for scale, d in e8["ablation_rho"].items():
        print(f"    scale={scale}: " + ", ".join(f"{k}={v:.2f}" for k, v in d.items()))
    summary["experiment_8"] = {
        "cross_cell_rho": e8["cross_cell_rho"],
        "cross_cell_rho_ci": e8["cross_cell_rho_ci"],
        "rho_llc": e8["rho_llc"], "within_H_concordance": e8["within_H_concordance"], "llc_rel_err": e8["llc_rel_err"].tolist(),
        "ablation_rho": {str(k): v for k, v in e8["ablation_rho"].items()},
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

    print("\nRunning Experiment 10 (Toy Model of Superposition, reproducing Chen et al. 2023)...")
    e10 = ex.experiment_10()
    plots.plot_tms(e10, "plots/exp10_tms.png")
    for name, c in e10["critical_points"].items():
        print(f"  {name:>3}-gon: loss={c['loss']:.5f} (paper {c['paper_loss']:.5f}), "
              f"|grad|={c['grad_norm']:.1e}, prior factor={c['prior']:.5f} (paper {c['paper_prior']:.5f})")
    for name, row in e10["llc"].items():
        ests = ", ".join(f"n={n}: {v['mean']:.2f}+/-{v['std']:.2f}" for n, v in row["est"].items())
        print(f"  LLC {name}-gon: theory {row['theory']}, {ests}")
    nc = e10["n_crit"]["5->6"]
    print(f"  5->6 critical sample size: {nc['ours']:.0f} (paper {nc['paper']})")
    vals, counts = np.unique(e10["sgd"]["near"], return_counts=True)
    print("  SGD checkpoints by plateau: " + ", ".join(f"{v or 'transit'}={c}" for v, c in zip(vals, counts)))
    summary["experiment_10"] = {
        "critical_points": e10["critical_points"],
        "llc": {k: {"theory": v["theory"], "paper_hat_n5000": v["paper_hat_n5000"],
                    "est": {str(n): e for n, e in v["est"].items()}} for k, v in e10["llc"].items()},
        "n_crit": e10["n_crit"],
        "sgd_plateau_counts": {(v or "transit"): int(c) for v, c in zip(vals, counts)},
        "sgd_final_loss": e10["sgd"]["loss"][-1].tolist(),
    }

    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))

    with open("results/tables.md", "w") as f:
        f.write("\n\n".join(all_tables))

    print(f"\nDone in {time.time()-t0:.1f}s. Plots in plots/, tables in results/tables.md, "
          f"raw summary in results/summary.json")


if __name__ == "__main__":
    main()
