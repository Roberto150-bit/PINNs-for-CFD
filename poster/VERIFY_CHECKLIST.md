# Poster verification checklist

Every number on `poster/poster_draft.pdf`, generated from `poster/poster_draft_sources.json`
(written by `poster/build_poster.py`). Errors are test-A relative L2 at the final training
step unless the description says otherwise; percentages are that value × 100. Ratios on the
poster (e.g. "5×", "5.5×", "8×") are one row divided by another; those rows show the raw value.

How to check one number: open the results file, find the listed run ids, and recompute the
mean (or read the value). Each run folder holds `metrics.json` (the scored values),
`eval_history.csv` (error every 500 steps) and `config.json` (settings).

## 1. Check by eye

- [ ] Title: "Evaluating Physics-Informed Learning for Fluid Dynamics"
- [ ] Name and year: "Roberto Pollicino (CS '26)"
- [ ] Adviser line: "Adviser: Brian C. O'Neill, PhD (Computer Science)"
- [ ] Contacts: Roberto.Pollicino@quinnipiac.edu · Brian.ONeill@quinnipiac.edu
- [ ] References [1]-[3]: authors, year, journal, pages (Marchi 2009, Hou 1995, Raissi 2019)
- [ ] Nothing cut off at the poster edges; every figure readable at arm's length

## 2. Numbers on the poster

| # | Printed as | Value | What it is | Results file | Run folders |
|---|---|---|---|---|---|
| 1 | 0.017, 1.7% | 0.0168777 | data_driven_nn GPU runs: mean relative L2 of u on test_a (`dnn_gpu_test_a_relative_l2_u_mean`) | experiments/results/experiment_results.csv (Re 100) | `experiments/experiment_001_re100/data_driven_nn/dnn_re100_gpu_s{1-5}/` |
| 2 | 0.082, 8.2% | 0.0823162 | pinn GPU runs: mean relative L2 of u on test_a (`pinn_gpu_test_a_relative_l2_u_mean`) | experiments/results/experiment_results.csv (Re 100) | `experiments/experiment_001_re100/pinn/pinn_re100_gpu_s{1-5}/` |
| 3 | 6.1% | 0.0607195 | hybrid GPU runs: mean relative L2 of u on test_a (`hybrid_gpu_test_a_relative_l2_u_mean`) | experiments/results/experiment_results.csv (Re 100) | `experiments/experiment_001_re100/hybrid/hybrid_re100_gpu_s{1-5}/` |
| 4 | 4.1% | 0.04112 | hybrid_re100_gpu_s5: best test-A u error during training (`spike_re100_hybrid_worst_best`) | eval_history.csv of each listed run (column rel_l2_u: final row vs smallest value) | `experiments/experiment_001_re100/hybrid/hybrid_re100_gpu_s{5}/` |
| 5 | 12% | 0.117545 | hybrid_re100_gpu_s5: test-A u error at the final step (`spike_re100_hybrid_worst_final`) | eval_history.csv of each listed run (column rel_l2_u: final row vs smallest value) | `experiments/experiment_001_re100/hybrid/hybrid_re100_gpu_s{5}/` |
| 6 | 0.546, 0.55 | 0.545592 | data_driven_nn GPU: mean continuity_residual (`dnn_gpu_continuity_residual_mean`) | experiments/results/experiment_results.csv (Re 100) | `experiments/experiment_001_re100/data_driven_nn/dnn_re100_gpu_s{1-5}/` |
| 7 | 0.071 | 0.0705346 | pinn GPU: mean continuity_residual (`pinn_gpu_continuity_residual_mean`) | experiments/results/experiment_results.csv (Re 100) | `experiments/experiment_001_re100/pinn/pinn_re100_gpu_s{1-5}/` |
| 8 | 0.16 | 0.159405 | hybrid GPU: mean continuity_residual (`hybrid_gpu_continuity_residual_mean`) | experiments/results/experiment_results.csv (Re 100) | `experiments/experiment_001_re100/hybrid/hybrid_re100_gpu_s{1-5}/` |
| 9 | 0.48 | 0.477131 | Re 100, seeds of the control runs: dataonly mean continuity_residual (`ctrl_re100_dataonly_cont_mean`) | experiments/results/control_re100.csv and experiment_results.csv (same seeds) | `experiments/experiment_001_re100/data_driven_nn/dnn_re100_gpu_s{1-5}/`; `experiments/experiment_001_re100/hybrid/hybrid_re100_gpu_s{1-5}/`; `experiments/experiment_001_re100/control/hybrid/hybrid_re100_gpu_s{1-5}_dataonly/` |
| 10 | 3.0% | 0.0295657 | Re 100, seeds of the control runs: dataonly mean test_a_relative_l2_u (`ctrl_re100_dataonly_u_mean`) | experiments/results/control_re100.csv and experiment_results.csv (same seeds) | `experiments/experiment_001_re100/data_driven_nn/dnn_re100_gpu_s{1-5}/`; `experiments/experiment_001_re100/hybrid/hybrid_re100_gpu_s{1-5}/`; `experiments/experiment_001_re100/control/hybrid/hybrid_re100_gpu_s{1-5}_dataonly/` |
| 11 | 6.1% | 0.0607195 | Re 100, seeds of the control runs: hybrid mean test_a_relative_l2_u (`ctrl_re100_hybrid_u_mean`) | experiments/results/control_re100.csv and experiment_results.csv (same seeds) | `experiments/experiment_001_re100/data_driven_nn/dnn_re100_gpu_s{1-5}/`; `experiments/experiment_001_re100/hybrid/hybrid_re100_gpu_s{1-5}/`; `experiments/experiment_001_re100/control/hybrid/hybrid_re100_gpu_s{1-5}_dataonly/` |
| 12 | 9.2% | 0.0915842 | Re 1000, seeds of the control runs: dataonly mean test_a_relative_l2_u (`ctrl_re1000_dataonly_u_mean`) | experiments/results/control_re1000.csv and experiment_results_re1000.csv (same seeds) | `experiments/experiment_004_re1000/data_driven_nn/dnn_re1000_gpu_s{1-5}/`; `experiments/experiment_004_re1000/hybrid/hybrid_re1000_gpu_s{1-5}/`; `experiments/experiment_004_re1000/control/hybrid/hybrid_re1000_gpu_s{1-5}_dataonly/` |
| 13 | 8.2% | 0.082003 | Re 1000, seeds of the control runs: hybrid mean test_a_relative_l2_u (`ctrl_re1000_hybrid_u_mean`) | experiments/results/control_re1000.csv and experiment_results_re1000.csv (same seeds) | `experiments/experiment_004_re1000/data_driven_nn/dnn_re1000_gpu_s{1-5}/`; `experiments/experiment_004_re1000/hybrid/hybrid_re1000_gpu_s{1-5}/`; `experiments/experiment_004_re1000/control/hybrid/hybrid_re1000_gpu_s{1-5}_dataonly/` |
| 14 | 4 | 4 | Re 1000: seeds where the hybrid's test-A u error is below its own data-only control's (`ctrl_re1000_hybrid_better_u_n`) | experiments/results/control_re1000.csv and experiment_results_re1000.csv (same seeds) | `experiments/experiment_004_re1000/data_driven_nn/dnn_re1000_gpu_s{1-5}/`; `experiments/experiment_004_re1000/hybrid/hybrid_re1000_gpu_s{1-5}/`; `experiments/experiment_004_re1000/control/hybrid/hybrid_re1000_gpu_s{1-5}_dataonly/` |
| 15 | 5 | 5 | hybrid data-only control at Re 1000: runs with same-seed hybrid and data NN (`ctrl_re1000_n_runs`) | experiments/results/control_re1000.csv and experiment_results_re1000.csv (same seeds) | `experiments/experiment_004_re1000/data_driven_nn/dnn_re1000_gpu_s{1-5}/`; `experiments/experiment_004_re1000/hybrid/hybrid_re1000_gpu_s{1-5}/`; `experiments/experiment_004_re1000/control/hybrid/hybrid_re1000_gpu_s{1-5}_dataonly/` |
| 16 | 1.8% | 0.0180833 | Re 10 data_driven_nn GPU runs: mean test_a_relative_l2_u (`re10_dnn_gpu_test_a_relative_l2_u_mean`) | experiments/results/experiment_results_re10.csv | `experiments/experiment_003_re10/data_driven_nn/dnn_re10_gpu_s{1-5}/` |
| 17 | 2.3% | 0.0230678 | Re 400 data_driven_nn GPU runs: mean test_a_relative_l2_u (`re400_dnn_gpu_test_a_relative_l2_u_mean`) | experiments/results/experiment_results_re400.csv | `experiments/experiment_002_re400/data_driven_nn/dnn_re400_gpu_s{1-5}/` |
| 18 | 1.6% | 0.0164315 | Re 1000 data_driven_nn GPU runs: mean test_a_relative_l2_u (`re1000_dnn_gpu_test_a_relative_l2_u_mean`) | experiments/results/experiment_results_re1000.csv | `experiments/experiment_004_re1000/data_driven_nn/dnn_re1000_gpu_s{1-5}/` |
| 19 | 6.1% | 0.0614105 | Re 10 hybrid GPU runs: mean test_a_relative_l2_u (`re10_hybrid_gpu_test_a_relative_l2_u_mean`) | experiments/results/experiment_results_re10.csv | `experiments/experiment_003_re10/hybrid/hybrid_re10_gpu_s{1-5}/` |
| 20 | 6.6% | 0.0660751 | Re 400 hybrid GPU runs: mean test_a_relative_l2_u (`re400_hybrid_gpu_test_a_relative_l2_u_mean`) | experiments/results/experiment_results_re400.csv | `experiments/experiment_002_re400/hybrid/hybrid_re400_gpu_s{1-5}/` |
| 21 | 8.2% | 0.082003 | Re 1000 hybrid GPU runs: mean test_a_relative_l2_u (`re1000_hybrid_gpu_test_a_relative_l2_u_mean`) | experiments/results/experiment_results_re1000.csv | `experiments/experiment_004_re1000/hybrid/hybrid_re1000_gpu_s{1-5}/` |
| 22 | 16% | 0.160834 | Re 10 pinn GPU runs: mean test_a_relative_l2_u (`re10_pinn_gpu_test_a_relative_l2_u_mean`) | experiments/results/experiment_results_re10.csv | `experiments/experiment_003_re10/pinn/pinn_re10_gpu_s{1-5}/` |
| 23 | 36% | 0.35649 | Re 400 pinn GPU runs: mean test_a_relative_l2_u (`re400_pinn_gpu_test_a_relative_l2_u_mean`) | experiments/results/experiment_results_re400.csv | `experiments/experiment_002_re400/pinn/pinn_re400_gpu_s{1-5}/` |
| 24 | 42% | 0.416077 | Re 1000 pinn GPU runs: mean test_a_relative_l2_u (`re1000_pinn_gpu_test_a_relative_l2_u_mean`) | experiments/results/experiment_results_re1000.csv | `experiments/experiment_004_re1000/pinn/pinn_re1000_gpu_s{1-5}/` |
| 25 | 0.03 | 0.028804 | Re 400 pinn GPU runs: mean continuity_residual (`re400_pinn_gpu_continuity_residual_mean`) | experiments/results/experiment_results_re400.csv | `experiments/experiment_002_re400/pinn/pinn_re400_gpu_s{1-5}/` |
| 26 | 0.55 | 0.547773 | Re 400 data_driven_nn GPU runs: mean continuity_residual (`re400_dnn_gpu_continuity_residual_mean`) | experiments/results/experiment_results_re400.csv | `experiments/experiment_002_re400/data_driven_nn/dnn_re400_gpu_s{1-5}/` |
| 27 | 35% | 0.348781 | S5 Re 400 pinn: test-A rel. L2 u at step 20,000 (`s5_re400_pinn_u_at_20k_mean`) | eval_history.csv of each listed run (rel_l2_u at step 20000 / 40000 / 100000) | `experiments/experiment_002_re400/extended/pinn/pinn_re400_gpu_s{1-3}_steps100k/` |
| 28 | 14% | 0.140634 | S5 Re 400 pinn: test-A rel. L2 u at step 100,000 (`s5_re400_pinn_u_at_100k_mean`) | eval_history.csv of each listed run (rel_l2_u at step 20000 / 40000 / 100000) | `experiments/experiment_002_re400/extended/pinn/pinn_re400_gpu_s{1-3}_steps100k/` |
| 29 | 3 | 3 | S5 Re 400 pinn: completed 100k-step runs (`s5_re400_pinn_n_runs`) | eval_history.csv of each listed run (rel_l2_u at step 20000 / 40000 / 100000) | `experiments/experiment_002_re400/extended/pinn/pinn_re400_gpu_s{1-3}_steps100k/` |
| 30 | 36% | 0.35649 | S6: standard Re 400 PINN runs with the same seeds, mean test_a_relative_l2_u (`s6_re400_base_u_mean`) | experiments/results/continuation_re400.csv vs experiment_results_re400.csv (same seed) | `experiments/experiment_002_re400/pinn/pinn_re400_gpu_s{1-5}/`; `experiments/experiment_002_re400/continuation/pinn/pinn_re400_gpu_s{1-5}_from_re100/` |
| 31 | 15% | 0.151907 | S6: Re 400 PINN started from the trained Re 100 PINN, mean test_a_relative_l2_u (`s6_re400_warm_u_mean`) | experiments/results/continuation_re400.csv vs experiment_results_re400.csv (same seed) | `experiments/experiment_002_re400/pinn/pinn_re400_gpu_s{1-5}/`; `experiments/experiment_002_re400/continuation/pinn/pinn_re400_gpu_s{1-5}_from_re100/` |
| 32 | 5 | 5 | S6: warm-started Re 400 PINN runs with a same-seed standard run (`s6_re400_n_runs`) | experiments/results/continuation_re400.csv vs experiment_results_re400.csv (same seed) | `experiments/experiment_002_re400/pinn/pinn_re400_gpu_s{1-5}/`; `experiments/experiment_002_re400/continuation/pinn/pinn_re400_gpu_s{1-5}_from_re100/` |
| 33 | 37% | 0.374594 | S5 Re 1000 pinn: test-A rel. L2 u at step 100,000 (`s5_re1000_pinn_u_at_100k_mean`) | eval_history.csv of each listed run (rel_l2_u at step 20000 / 40000 / 100000) | `experiments/experiment_004_re1000/extended/pinn/pinn_re1000_gpu_s{1-3}_steps100k/` |
| 34 | 3 | 3 | S5 Re 1000 pinn: completed 100k-step runs (`s5_re1000_pinn_n_runs`) | eval_history.csv of each listed run (rel_l2_u at step 20000 / 40000 / 100000) | `experiments/experiment_004_re1000/extended/pinn/pinn_re1000_gpu_s{1-3}_steps100k/` |
| 35 | 3.9 | 3.92744 | FD 64x64 transient (5,000 steps) solve time, median of repeats (`fd_n64_solve_seconds_median`) | experiments/results/poster_numbers.json (FD timing / validation; see description) | `experiments/experiment_001_re100/finite_difference/fd_re100_n64_transient_timing_r1/`; `experiments/experiment_001_re100/finite_difference/fd_re100_n64_transient_timing_r2/`; `experiments/experiment_001_re100/finite_difference/fd_re100_n64_transient_timing_r3/` |
| 36 | 781.969, 782 s | 781.969 | pinn gpu: total cost to a usable field (`pinn_gpu_total_cost_to_field_seconds`) | experiments/results/experiment_results.csv (Re 100) | `experiments/experiment_001_re100/pinn/pinn_re100_gpu_s{1-5}/` |
| 37 | 140.930, 141 s | 140.93 | data_driven_nn gpu: total cost to a usable field (`dnn_gpu_total_cost_to_field_seconds`) | experiments/results/experiment_results.csv (Re 100) | `experiments/experiment_001_re100/data_driven_nn/dnn_re100_gpu_s{1-5}/` |
| 38 | 886 s | 886.351 | hybrid gpu: total cost to a usable field (`hybrid_gpu_total_cost_to_field_seconds`) | experiments/results/experiment_results.csv (Re 100) | `experiments/experiment_001_re100/hybrid/hybrid_re100_gpu_s{1-5}/` |

## 3. Figures

Every poster figure has a `.values.json` next to its PNG in `experiments/analysis/plots/`
(`v4_*`, `pipelines`). `python experiments/analysis/make_figures.py` re-checks each plotted
value against the results files (gate G15) and prints PASS / FAIL per figure.
