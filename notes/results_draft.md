# Results (draft)

Draft of the Results chapter. The method is described in `notes/methodology_draft.md`; a short version of the findings is in `notes/findings_summary.md`.

**Conventions.** Errors are relative $L_2$ errors against the $64 \times 64$ finite-difference (FD) reference at the final training step. Test A is the held-out 15% of the space-time points, test B the ten held-out snapshots. Tables show the mean over seeds with the sample standard deviation in brackets and list the seeds used; every number comes from the file named under the table. Tables are generated from those files, not typed.

**Still changing.** The data-only control at $Re = 10$ and 1000 and further 100,000-step runs (hybrid at $Re = 400$, seeds 2–3; PINN at $Re = 1000$, seeds 2–3; hybrid at $Re = 1000$, seed 1) are being trained. Every number marked **[3 seeds; 5 pending]** or **[pending]** will change when they are scored; the others are final.

## 4.1 Reference solution

The FD solver reproduces the steady benchmark of Marchi et al. (2009) at $Re = 100$ to within 0.0056 in $u$ and 0.0065 in $v$ on the $64 \times 64$ grid, and its primary vortex centre lies 0.0039 from the benchmark position [`experiments/results/poster_numbers.json`]. The validation at all four Reynolds numbers, the grid convergence and the time-step check are in Section 1 of the methodology draft.

**Figures.** `experiments/analysis/plots/fd_centerlines__*.png`, `fd_vortex_center__*.png`, `fd_grid_convergence__*.png`.

## 4.2 RQ1: Accuracy

**Finding.** The data NN reproduces the FD field to about 2% in $u$ and $v$ at every Reynolds number. The PINN is about five times less accurate at $Re = 100$ and converges to a wrong flow at $Re = 400$ and 1000. The hybrid lies between them at every Reynolds number.

<!-- table:accuracy -->
| $Re$ | Network | Seeds | Test A $u$ | Test A $v$ | Test A $p$ | Test B $u$ | Test B $v$ | Test B $p$ |
|---|---|---|---|---|---|---|---|---|
| 10 | Data NN | 1,2,3,4,5 | 0.018 (0.0021) | 0.019 (0.0017) | 0.119 (0.0015) | 0.015 (0.0027) | 0.015 (0.0017) | 0.122 (0.0013) |
| 10 | PINN | 1,2,3,4,5 | 0.161 (0.0073) | 0.196 (0.0035) | 0.524 (0.0046) | 0.156 (0.007) | 0.196 (0.0039) | 0.509 (0.005) |
| 10 | Hybrid | 1,2,3,4,5 | 0.061 (0.0081) | 0.059 (0.013) | 0.217 (0.0035) | 0.052 (0.008) | 0.055 (0.013) | 0.211 (0.0021) |
| 100 | Data NN | 1,2,3,4,5 | 0.017 (0.001) | 0.018 (0.0034) | 0.113 (0.0013) | 0.014 (0.0012) | 0.017 (0.0039) | 0.118 (0.0018) |
| 100 | PINN | 1,2,3,4,5 | 0.082 (0.011) | 0.166 (0.021) | 0.265 (0.0049) | 0.077 (0.011) | 0.161 (0.020) | 0.239 (0.0057) |
| 100 | Hybrid | 1,2,3,4,5 | 0.061 (0.032) | 0.059 (0.021) | 0.181 (0.029) | 0.056 (0.031) | 0.055 (0.022) | 0.190 (0.029) |
| 400 | Data NN | 1,2,3,4,5 | 0.023 (0.0015) | 0.025 (0.0024) | 0.161 (0.0084) | 0.020 (0.0011) | 0.023 (0.0026) | 0.166 (0.0099) |
| 400 | PINN | 1,2,3,4,5 | 0.356 (0.019) | 0.905 (0.076) | 0.889 (0.090) | 0.352 (0.018) | 0.901 (0.076) | 0.888 (0.084) |
| 400 | Hybrid | 1,2,3,4,5 | 0.066 (0.012) | 0.071 (0.013) | 0.228 (0.014) | 0.055 (0.0097) | 0.067 (0.013) | 0.233 (0.0072) |
| 1000 | Data NN | 1,2,3,4,5 | 0.016 (0.0024) | 0.019 (0.0047) | 0.020 (0.0084) | 0.014 (0.002) | 0.017 (0.0048) | 0.016 (0.0067) |
| 1000 | PINN | 1,2,3,4,5 | 0.416 (0.0096) | 0.984 (0.0065) | 1.02 (0.023) | 0.411 (0.0089) | 0.981 (0.0063) | 0.989 (0.0033) |
| 1000 | Hybrid | 1,2,3,4,5 | 0.082 (0.012) | 0.102 (0.0045) | 0.169 (0.036) | 0.062 (0.016) | 0.096 (0.0044) | 0.116 (0.060) |

Mean (sample s.d.) over the seeds listed; relative $L_2$ error at the final step. Source: `experiments/results/experiment_results*.csv` (main GPU runs).
<!-- /table:accuracy -->


**Figures.**
- `analysis_reynolds_trend.png`: test-A error against $Re$, every seed.
- `analysis_error_vs_t.png`: error over time $t$ at each $Re$.
- `analysis_error_map_re100.png` (and `_re10`, `_re400`, `_re1000`): where in the cavity the error lies.
- Poster version: `v4_accuracy__*.png`, `v4_trend__*.png`.

**Caveats.**
- The data NN is scored on the same FD solution it was trained on: test A interpolates between training points, test B between training snapshots. The result shows how well the network represents a solved flow, not how it would do without the solver.
- Test B errors are similar to or smaller than test A errors for every network, so held-out times are not harder than held-out points here.
- At $Re = 1000$ the reference is the $127 \times 127$ solution at every second grid point, because the $64 \times 64$ FD solution is itself 0.251 off in $u$ [`experiment_results_re1000.csv`]. The data NN at $Re = 1000$ is also trained on the $127 \times 127$ solution.
- Pressure errors are larger than velocity errors for every network. The FD pressure has sharp peaks at the lid corners, and pressure is compared after removing each snapshot's mean.

## 4.3 RQ2: Physical consistency

**Finding.** The PINN satisfies the governing equations far better than the data NN at every Reynolds number, including where its flow is wrong. A small residual does not mean an accurate flow. The hybrid's residuals lie between the two. The data NN fits the boundary conditions best; for the initial condition the data NN and the PINN are similar.

<!-- table:physics -->
| $Re$ | Network | Seeds | Continuity | x-momentum | y-momentum | Continuity, grid stencil (core) | Boundary RMS | Initial-condition RMS |
|---|---|---|---|---|---|---|---|---|
| 10 | Data NN | 1,2,3,4,5 | 0.569 (0.037) | 51.6 (13.6) | 19.4 (1.58) | 0.061 (0.0039) | 0.0054 (5.9e-04) | 0.0057 (3.6e-04) |
| 10 | PINN | 1,2,3,4,5 | 0.018 (0.002) | 0.054 (0.017) | 0.029 (0.0043) | 0.017 (7.6e-04) | 0.039 (8.5e-04) | 0.0052 (7.1e-04) |
| 10 | Hybrid | 1,2,3,4,5 | 0.078 (0.012) | 0.291 (0.099) | 0.341 (0.087) | 0.028 (0.0026) | 0.024 (0.0014) | 0.011 (0.0032) |
| 100 | Data NN | 1,2,3,4,5 | 0.546 (0.063) | 5.57 (0.165) | 2.06 (0.199) | 0.055 (0.0027) | 0.0053 (6.6e-04) | 0.0035 (3.9e-04) |
| 100 | PINN | 1,2,3,4,5 | 0.071 (0.030) | 0.096 (0.045) | 0.054 (0.016) | 0.023 (0.0021) | 0.017 (0.0026) | 0.0049 (0.0017) |
| 100 | Hybrid | 1,2,3,4,5 | 0.159 (0.059) | 0.875 (0.680) | 0.403 (0.236) | 0.028 (0.0039) | 0.022 (0.013) | 0.0066 (0.0021) |
| 400 | Data NN | 1,2,3,4,5 | 0.548 (0.022) | 2.08 (0.483) | 0.724 (0.058) | 0.063 (0.012) | 0.0055 (6.4e-04) | 0.0033 (4.8e-04) |
| 400 | PINN | 1,2,3,4,5 | 0.029 (0.020) | 0.036 (0.0039) | 0.026 (0.017) | 0.017 (0.0021) | 0.031 (0.0044) | 0.0034 (7.1e-04) |
| 400 | Hybrid | 1,2,3,4,5 | 0.075 (0.070) | 0.213 (0.207) | 0.090 (0.055) | 0.031 (0.006) | 0.022 (0.0033) | 0.0062 (0.0028) |
| 1000 | Data NN | 1,2,3,4,5 | 0.338 (0.024) | 1.32 (0.279) | 0.436 (0.019) | 0.043 (0.0025) | 0.004 (9.1e-04) | 0.0023 (2.5e-04) |
| 1000 | PINN | 1,2,3,4,5 | 0.012 (0.0016) | 0.030 (0.012) | 0.013 (0.0032) | 0.014 (5.8e-04) | 0.028 (8.5e-04) | 0.0049 (0.0016) |
| 1000 | Hybrid | 1,2,3,4,5 | 0.052 (0.0042) | 0.119 (0.013) | 0.096 (0.019) | 0.035 (0.0028) | 0.025 (0.0036) | 0.0076 (8.7e-04) |

Residuals: RMS by automatic differentiation at 50,000 fresh interior points; grid stencil: second-order central differences on the $64 \times 64$ grid, 3 points from the walls. Mean (s.d.) over the seeds listed. Source: `experiments/results/experiment_results*.csv`.
<!-- /table:physics -->

The residuals by automatic differentiation are averages over random points, and a few points near the two lid corners can dominate them (Section 4.7 shows where this matters). The next table repeats the continuity residual without the points closest to either lid corner. For the 20,000-step runs the order PINN < hybrid < data NN is the same at every radius; for the long-trained runs the picture changes (Section 4.7).

<!-- table:corner -->
| $Re$ | Network | Runs | Seeds | All points (primary) | $r \ge 0.02$ | $r \ge 0.05$ | $r \ge 0.1$ |
|---|---|---|---|---|---|---|---|
| 10 | Data NN | 20,000 steps | 1,2,3,4,5 | 0.569 (0.037) | 0.526 (0.041) | 0.318 (0.039) | 0.169 (0.065) |
| 10 | PINN | 20,000 steps | 1,2,3,4,5 | 0.018 (0.002) | 0.013 (9.0e-04) | 0.012 (6.5e-04) | 0.011 (5.0e-04) |
| 10 | Hybrid | 20,000 steps | 1,2,3,4,5 | 0.078 (0.012) | 0.034 (0.0036) | 0.028 (0.0032) | 0.020 (0.0021) |
| 10 | Control | control | 1,2,3 | 0.688 (0.058) | 0.636 (0.070) | 0.310 (0.090) | 0.162 (0.055) |
| 10 | PINN | warm start (S6) | 1,2,3 | 0.016 (0.0018) | 0.012 (5.8e-04) | 0.011 (5.6e-04) | 0.0092 (4.8e-04) |
| 100 | Data NN | 20,000 steps | 1,2,3,4,5 | 0.546 (0.063) | 0.494 (0.053) | 0.300 (0.080) | 0.140 (0.024) |
| 100 | PINN | 20,000 steps | 1,2,3,4,5 | 0.071 (0.030) | 0.019 (0.0046) | 0.016 (0.0048) | 0.014 (0.0042) |
| 100 | Hybrid | 20,000 steps | 1,2,3,4,5 | 0.159 (0.059) | 0.034 (0.011) | 0.027 (0.012) | 0.023 (0.011) |
| 100 | Control | control | 1,2,3 | 0.448 (0.029) | 0.392 (0.040) | 0.214 (0.026) | 0.138 (0.024) |
| 100 | Data NN | S1: 1% of data | 1,2,3 | 0.464 (0.044) | 0.445 (0.046) | 0.355 (0.031) | 0.262 (0.024) |
| 100 | Data NN | S1: 10% of data | 1,2,3 | 0.587 (0.011) | 0.523 (0.031) | 0.345 (0.019) | 0.181 (0.013) |
| 100 | PINN | S2: 5,000 points | 1,2,3 | 1.26 (0.278) | 0.428 (0.134) | 0.028 (0.0063) | 0.020 (0.0048) |
| 100 | PINN | S2: 10,000 points | 1,2,3 | 0.130 (0.046) | 0.058 (0.017) | 0.019 (6.6e-04) | 0.014 (0.0031) |
| 100 | Data NN | S3: 4 x 50 | 1,2,3 | 0.508 (0.066) | 0.446 (0.074) | 0.239 (0.015) | 0.117 (0.011) |
| 100 | PINN | S3: 4 x 50 | 1,2,3 | 0.020 (0.0022) | 0.018 (0.0022) | 0.017 (0.0019) | 0.015 (0.0017) |
| 100 | Data NN | S3: 6 x 100 | 1,2,3 | 0.490 (0.025) | 0.449 (0.035) | 0.313 (0.037) | 0.188 (0.102) |
| 100 | PINN | S3: 6 x 100 | 1,2,3 | 0.410 (0.204) | 0.023 (0.0022) | 0.014 (0.0023) | 0.012 (0.0026) |
| 400 | Data NN | 20,000 steps | 1,2,3,4,5 | 0.548 (0.022) | 0.483 (0.035) | 0.314 (0.028) | 0.174 (0.019) |
| 400 | PINN | 20,000 steps | 1,2,3,4,5 | 0.029 (0.020) | 0.014 (0.0025) | 0.012 (0.0014) | 0.011 (0.0013) |
| 400 | Hybrid | 20,000 steps | 1,2,3,4,5 | 0.075 (0.070) | 0.033 (0.0065) | 0.027 (0.0067) | 0.025 (0.006) |
| 400 | Control | control | 1,2,3 | 0.326 (0.054) | 0.286 (0.048) | 0.207 (0.031) | 0.149 (0.020) |
| 400 | PINN | 100,000 steps (S5) | 1,2,3 | 1.55 (0.311) | 0.105 (0.086) | 0.0083 (5.2e-04) | 0.0056 (6.0e-04) |
| 400 | Hybrid | 100,000 steps (S5) | 1 | 0.613 | 0.100 | 0.020 | 0.015 |
| 400 | PINN | warm start (S6) | 1,2,3,4,5 | 1.32 (0.178) | 0.042 (0.018) | 0.010 (0.0017) | 0.0086 (0.0015) |
| 1000 | Data NN | 20,000 steps | 1,2,3,4,5 | 0.338 (0.024) | 0.244 (0.017) | 0.161 (0.013) | 0.115 (0.019) |
| 1000 | PINN | 20,000 steps | 1,2,3,4,5 | 0.012 (0.0016) | 0.011 (0.0018) | 0.010 (0.0019) | 0.0097 (0.002) |
| 1000 | Hybrid | 20,000 steps | 1,2,3,4,5 | 0.052 (0.0042) | 0.040 (0.0043) | 0.031 (0.0037) | 0.028 (0.0037) |
| 1000 | PINN | 100,000 steps (S5) | 1 | 0.821 | 0.229 | 0.0091 | 0.0063 |
| 1000 | PINN | warm start (S6) | 1,2,3 | 1.27 (0.794) | 0.112 (0.105) | 0.0083 (0.0019) | 0.007 (0.002) |

Continuity residual (RMS, automatic differentiation) at the 50,000 evaluation points, and without the points closer than $r$ to either lid corner (post-hoc, added after the S5 and S6 values were seen; all points: 50,000; $r \ge 0.05$: 49,793). Mean (s.d.). Source: `experiments/analysis/plots/analysis_corner_excluded.csv`.
<!-- /table:corner -->

**Figures.**
- `analysis_reynolds_trend.png` (lower panel): continuity residual against $Re$.
- Poster version: `v4_physics__*.png`.

**Caveats.**
- The residual measures how well a network satisfies the equations, not how close it is to the FD solution. At $Re = 400$ and 1000 the PINN has the smallest residuals and the largest errors.
- The grid-stencil value (central differences on the $64 \times 64$ grid, away from the walls) is about ten times smaller than the automatic-differentiation value for the data NN, because it averages over a grid spacing; for the PINN the two are similar. The FD reference itself has a stencil continuity of $1.5 \times 10^{-4}$ at $Re = 100$ [`poster_numbers.json`].
- The corner-excluded values were added after the S5 and S6 results were seen. The all-points value stays the primary result.

## 4.4 RQ3: Repeatability and robustness

**Finding.** Across seeds, all three networks are repeatable at their own level of accuracy; the largest spread is the hybrid's at $Re = 100$, caused by one run that ends on an error spike. Across Reynolds numbers, the data NN and the hybrid are robust and the PINN is not: every PINN seed at $Re = 400$ and 1000 converges to the same wrong flow, so the failure is systematic rather than a matter of initialization.

<!-- table:repeat -->
| $Re$ | Network | Seeds | Test A $u$, min–max | s.d. | s.d. / mean | Runs with a spike | Runs ending on a spike |
|---|---|---|---|---|---|---|---|
| 10 | Data NN | 1,2,3,4,5 | 0.017–0.022 | 0.0021 | 0.117 | 3 of 5 | 0 |
| 10 | PINN | 1,2,3,4,5 | 0.151–0.168 | 0.0073 | 0.045 | 0 of 5 | 0 |
| 10 | Hybrid | 1,2,3,4,5 | 0.056–0.076 | 0.0081 | 0.133 | 0 of 5 | 0 |
| 100 | Data NN | 1,2,3,4,5 | 0.015–0.018 | 0.001 | 0.061 | 2 of 5 | 0 |
| 100 | PINN | 1,2,3,4,5 | 0.072–0.100 | 0.011 | 0.131 | 0 of 5 | 0 |
| 100 | Hybrid | 1,2,3,4,5 | 0.037–0.118 | 0.032 | 0.533 | 1 of 5 | 1 |
| 400 | Data NN | 1,2,3,4,5 | 0.021–0.025 | 0.0015 | 0.064 | 2 of 5 | 0 |
| 400 | PINN | 1,2,3,4,5 | 0.324–0.371 | 0.019 | 0.053 | 0 of 5 | 0 |
| 400 | Hybrid | 1,2,3,4,5 | 0.054–0.080 | 0.012 | 0.183 | 0 of 5 | 0 |
| 1000 | Data NN | 1,2,3,4,5 | 0.014–0.020 | 0.0024 | 0.144 | 3 of 5 | 0 |
| 1000 | PINN | 1,2,3,4,5 | 0.404–0.428 | 0.0096 | 0.023 | 0 of 5 | 0 |
| 1000 | Hybrid | 1,2,3,4,5 | 0.069–0.099 | 0.012 | 0.145 | 0 of 5 | 0 |

Spike: an evaluation step $\ge$ 1,000 at which the test-A $u$ error is more than twice the best value at earlier steps (rule fixed before the hybrid runs). Sources: `experiments/results/experiment_results*.csv`, `experiments/analysis/plots/analysis_spikes.csv`.
<!-- /table:repeat -->


**Figures.**
- `analysis_repeatability.png`: test-A error against training step, every seed (data NN and PINN).
- `analysis_three_methods.png` ($Re = 100$, 400) and `analysis_three_methods_re10_re1000.png`: the same for the three networks, mean and range over seeds.
- `analysis_reynolds_trend.png`: every seed at every $Re$.

**Caveats.**
- The final-step protocol reports the network as it is at step 20,000, including a run that ends on a spike (`hybrid_re100_gpu_s5`: 0.118 at the last step, 0.041 at its best). No early stopping was used for any method.
- The spike rule (a value more than twice the best earlier value) was fixed before the hybrid runs; a different factor would give different counts. The data NN's spikes are short and it recovers from them.
- Five seeds are few for a standard deviation; the min–max range is the safer summary.

## 4.5 RQ4: Computational cost

**Finding.** For one flow on this grid, the FD solver is far cheaper than training any network. A trained network evaluates the full field faster than the FD solver, so training pays off only after many evaluations of the same flow.

<!-- table:cost -->
| $Re$ | Network (GPU) | Seeds with valid timing | Training time (s) | Time per step (s) | Full-field inference (s) | Peak VRAM (MiB) |
|---|---|---|---|---|---|---|
| 10 | Data NN | 4,5 | 144 (6.28) | 0.0071 (3.2e-04) | 0.010 (4.0e-04) | 122 (0) |
| 10 | PINN | 4,5 | 813 (24.1) | 0.040 (0.0012) | 0.010 (4.1e-04) | 706 (1.91) |
| 10 | Hybrid | 1,2,3,4,5 | 888 (14.6) | 0.044 (7.1e-04) | 0.011 (6.3e-04) | 712 (2.29) |
| 100 | Data NN | 1,2,3,4,5 | 137 (4.66) | 0.0068 (2.3e-04) | 0.010 (1.6e-04) | 122 (0) |
| 100 | PINN | 1,2,3,4,5 | 782 (2.22) | 0.039 (1.0e-04) | 0.010 (8.1e-04) | 708 (3.49) |
| 100 | Hybrid | 1,2,3,4,5 | 882 (14.9) | 0.044 (7.6e-04) | 0.011 (0.0012) | 711 (2.55) |
| 400 | Data NN | 1,2,3,4,5 | 148 (7.34) | 0.0073 (3.7e-04) | 0.010 (3.1e-04) | 122 (0) |
| 400 | PINN | 1,2,3,4,5 | 779 (3.02) | 0.039 (1.5e-04) | 0.010 (6.5e-04) | 707 (2.60) |
| 400 | Hybrid | 1,2,3,4,5 | 882 (11.5) | 0.044 (5.9e-04) | 0.011 (5.9e-04) | 710 (1.66) |
| 1000 | Data NN | 4,5 | 141 (3.64) | 0.007 (1.9e-04) | 0.012 (0.0022) | 122 (0) |
| 1000 | PINN | 4,5 | 828 (22.6) | 0.041 (0.0011) | 0.010 (1.8e-04) | 705 (0.206) |
| 1000 | Hybrid | 1,2,3,4,5 | 897 (0.981) | 0.045 (4.5e-05) | 0.010 (9.1e-04) | 713 (2.82) |

Only runs whose `config.json` has a valid timing (GPU not shared). Mean (s.d.). Source: `experiments/results/experiment_results*.csv`.

| Quantity ($Re = 100$) | Value |
|---|---|
| FD solve, $32^2$ / $64^2$ / $128^2$, CPU median (s) | 1.16 / 3.93 / 21.7 |
| FD scaling exponent (time vs grid points) | 1.06 |
| Data NN training, CPU (s) | 269 |
| PINN time per step, CPU (s) | 0.297 |
| Break-even field evaluations: data NN GPU / data NN CPU / PINN GPU / hybrid GPU | 36.0 / 70.6 / 200 / 226 |

Source: `experiments/results/poster_numbers.json`.
<!-- /table:cost -->

**Figures.** Poster version: `v4_cost__*.png`.

**Caveats.**
- All times come from one laptop (Intel i5-12450H, RTX 4060 Laptop GPU). FD ran on the CPU, the networks on the GPU.
- The data NN's cost to a usable field includes the FD solve that produces its data.
- Runs that shared the GPU are excluded. At $Re = 10$ and 1000 the data NN and PINN seeds 1–3 shared the GPU, so their cost is not reported; seeds 4–5 are timed on a free GPU.
- At $Re = 1000$ a fair cost must count the $127 \times 127$ solve behind the reference.
- The break-even counts repeated evaluations of the same flow. A network trained at one $Re$ does not replace an FD solve at another.

## 4.6 Hybrid network

**Finding.** Adding the FD data to the PINN loss gives a network that avoids the PINN's failure at $Re = 400$ and 1000, with an error of 6–8% in $u$ at every Reynolds number and residuals between those of the PINN and the data NN. It does not reach the data NN's accuracy. The data-only control shows that a large part of the gap to the data NN comes from the output units (physical instead of standardized), not from the physics loss.

Accuracy, physics and cost of the hybrid are in the tables of Sections 4.2–4.5. The control separates the two differences between the hybrid and the data NN:

<!-- table:control -->
| $Re$ | Network | Seeds | Test A $u$ | Test A $v$ | Test A $p$ | Continuity | x-momentum |
|---|---|---|---|---|---|---|---|
| 10 | Data NN | 1,2,3 | 0.019 (0.0027) | 0.018 (0.0014) | 0.119 (0.0016) | 0.584 (0.037) | 57.5 (14.9) |
| 10 | Control (data loss only, physical outputs) | 1,2,3 | 0.021 (0.0027) | 0.036 (0.012) | 0.125 (0.0034) | 0.688 (0.058) | 48.4 (7.64) |
| 10 | Hybrid | 1,2,3 | 0.058 (0.0012) | 0.054 (0.0072) | 0.216 (0.0038) | 0.083 (0.011) | 0.228 (0.069) |
| 100 | Data NN | 1,2,3 | 0.017 (0.0014) | 0.018 (0.0011) | 0.113 (0.0012) | 0.547 (0.076) | 5.62 (0.090) |
| 100 | Control (data loss only, physical outputs) | 1,2,3 | 0.030 (0.0059) | 0.039 (0.0042) | 0.126 (0.0031) | 0.448 (0.029) | 2.43 (0.272) |
| 100 | Hybrid | 1,2,3 | 0.050 (0.0036) | 0.053 (0.0062) | 0.169 (0.0023) | 0.173 (0.032) | 1.01 (0.745) |
| 400 | Data NN | 1,2,3 | 0.023 (0.002) | 0.025 (0.0016) | 0.158 (0.004) | 0.549 (0.030) | 2.08 (0.555) |
| 400 | Control (data loss only, physical outputs) | 1,2,3 | 0.053 (0.0014) | 0.066 (0.0094) | 0.201 (0.003) | 0.326 (0.054) | 0.686 (0.055) |
| 400 | Hybrid | 1,2,3 | 0.065 (0.011) | 0.074 (0.0071) | 0.231 (0.017) | 0.041 (0.0043) | 0.177 (0.188) |

Same seeds in the three rows of each $Re$. Sources: `experiments/results/control_re*.csv`, `experiment_results*.csv`.
<!-- /table:control -->

$Re = 10$ and 1000 controls: **[pending]**.

**Figures.**
- `analysis_hybrid_control.png`: data NN, control and hybrid, every seed.
- `analysis_three_methods.png`, `analysis_reynolds_trend.png`, `analysis_error_map_re400.png`.

**Caveats.**
- The control was added after a fairness review found the difference in output units, and after the first hybrid results had been seen. It changes no existing run.
- The loss weights were not tuned. All start at 1; the one-time check at step 2,000 never changed them, because the largest-to-smallest ratio of the weighted terms stayed below 100 in every hybrid run (each run's `config.json`).
- The hybrid costs slightly more to train than the PINN, because it computes both losses at every step.

## 4.7 Additional studies

**S1–S3: data, collocation points and network size ($Re = 100$).** With 1% of the training rows the data NN is less accurate than the PINN without data; with 10% it is between the two. With 5,000 collocation points the PINN's continuity residual at the evaluation points rises sixteenfold, mostly at the lid corners; away from them it is less than twice that of the 20,000-point runs (corner table, Section 4.3). A larger network helps neither method.

<!-- table:sensitivity -->
| Study | Network | Change | Seeds | Test A $u$ | Continuity |
|---|---|---|---|---|---|
| – | Data NN | standard (main runs) | 1,2,3 | 0.017 (0.0014) | 0.547 (0.076) |
| – | PINN | standard (main runs) | 1,2,3 | 0.084 (0.015) | 0.080 (0.039) |
| S1 | Data NN | 1% of training rows | 1,2,3 | 0.133 (0.0057) | 0.464 (0.044) |
| S1 | Data NN | 10% of training rows | 1,2,3 | 0.050 (0.013) | 0.587 (0.011) |
| S2 | PINN | 5,000 collocation points | 1,2,3 | 0.119 (0.012) | 1.26 (0.278) |
| S2 | PINN | 10,000 collocation points | 1,2,3 | 0.089 (0.014) | 0.130 (0.046) |
| S3 | Data NN | 4 x 50 network | 1,2,3 | 0.025 (0.002) | 0.508 (0.066) |
| S3 | Data NN | 6 x 100 network | 1,2,3 | 0.043 (0.051) | 0.490 (0.025) |
| S3 | PINN | 4 x 50 network | 1,2,3 | 0.128 (0.0065) | 0.020 (0.0022) |
| S3 | PINN | 6 x 100 network | 1,2,3 | 0.091 (0.020) | 0.410 (0.204) |

$Re = 100$. Sources: `experiments/results/sensitivity_re100.csv`, `experiment_results.csv` (standard runs, seeds 1–3).
<!-- /table:sensitivity -->

**Figures.** `analysis_s1_data.png`, `analysis_s2_collocation.png`, `analysis_s3_size.png`.

**S4: causal weighting (PINN, $Re = 400$).** No effect on the final error.

<!-- table:s4 -->
| $Re$ | Network | Runs | Seeds | Test A $u$ | Test A $v$ | Test A $p$ | Continuity | $u$ ratio per seed (new / old) |
|---|---|---|---|---|---|---|---|---|
| 400 | PINN | standard, same seeds | 1,2,3 | 0.349 (0.022) | 0.883 (0.098) | 0.859 (0.114) | 0.038 (0.023) |  |
| 400 | PINN | causal weighting | 1,2,3 | 0.349 (0.024) | 0.892 (0.090) | 0.880 (0.095) | 0.035 (0.019) | 0.99, 1.01, 1.00 |

Sources: `experiments/results/causal_re400.csv`, `experiment_results_re400.csv`.
<!-- /table:s4 -->

**Figure.** `analysis_s4_causal.png`.

**S5: training five times longer.** At $Re = 400$ the PINN's error falls in every seed but stays far above the data NN's; at $Re = 100$ the PINN barely changes and at $Re = 1000$ (one seed) it does not recover. Measured over all evaluation points, the long-trained networks violate continuity much more than after 20,000 steps. Nearly all of that comes from the two lid corners, where the wall velocity is discontinuous and the exact solution is singular: without the points within 0.05 of a lid corner, the long-trained PINNs satisfy continuity better than the 20,000-step PINNs at $Re = 400$ and as well at $Re = 1000$ (table in Section 4.3).

<!-- table:s5 -->
| $Re$ | Network | Runs | Seeds | Test A $u$ | Test A $v$ | Test A $p$ | Continuity | $u$ ratio per seed (new / old) |
|---|---|---|---|---|---|---|---|---|
| 400 | PINN | 20,000 steps, same seeds | 1,2,3 | 0.349 (0.022) | 0.883 (0.098) | 0.859 (0.114) | 0.038 (0.023) |  |
| 400 | PINN | 100,000 steps | 1,2,3 | 0.141 (0.013) | 0.242 (0.034) | 0.797 (0.444) | 1.55 (0.311) | 0.42, 0.34, 0.45 |
| 400 | Hybrid | 20,000 steps, same seeds | 1 | 0.076 | 0.082 | 0.250 | 0.038 |  |
| 400 | Hybrid | 100,000 steps | 1 | 0.044 | 0.049 | 0.203 | 0.613 | 0.57 |
| 1000 | PINN | 20,000 steps, same seeds | 1 | 0.413 | 0.993 | 1.05 | 0.010 |  |
| 1000 | PINN | 100,000 steps | 1 | 0.390 | 0.942 | 1.03 | 0.821 | 0.94 |

Scored at the final step (100,000) at 50,000 fresh points. Sources: `experiments/results/extended_re*.csv`, `experiment_results*.csv`.

| $Re$, network | Runs | $u$ at 20,000 | $u$ at 40,000 | $u$ at 100,000 |
|---|---|---|---|---|
| 100, Data NN | 3 | 0.017 | 0.014 | 0.011 |
| 100, PINN | 3 | 0.084 | 0.064 | 0.082 |
| 400, PINN | 3 | 0.349 | 0.212 | 0.141 |
| 400, Hybrid | 1 | 0.076 | 0.053 | 0.044 |
| 1000, PINN | 1 | 0.413 | 0.386 | 0.390 |

Test-A $u$ from the runs' evaluation log (mean over runs). Source: `experiments/results/poster_numbers.json`.
<!-- /table:s5 -->

$Re = 400$ hybrid and $Re = 1000$ PINN: **[1 seed; 3 pending]**; $Re = 1000$ hybrid: **[pending]**. The 100,000-step runs at $Re = 100$ have no physics evaluation yet.

**S6: starting from a trained PINN at a neighbouring Reynolds number.** At $Re = 400$ every seed started from the $Re = 100$ PINN ends at less than half the error of the random start. At $Re = 1000$, started from those $Re = 400$ runs, two of three seeds improve and one does not change. At $Re = 10$ there is no clear effect. The warm-start runs show the same lid-corner rise in the all-points residual as S5.

<!-- table:s6 -->
| $Re$ | Network | Runs | Seeds | Test A $u$ | Test A $v$ | Test A $p$ | Continuity | $u$ ratio per seed (new / old) |
|---|---|---|---|---|---|---|---|---|
| 400 | PINN | random start, same seeds | 1,2,3,4,5 | 0.356 (0.019) | 0.905 (0.076) | 0.889 (0.090) | 0.029 (0.020) |  |
| 400 | PINN | started from the $Re = 100$ PINN | 1,2,3,4,5 | 0.152 (0.010) | 0.291 (0.022) | 0.411 (0.027) | 1.32 (0.178) | 0.46, 0.40, 0.46, 0.44, 0.38 |
| 1000 | PINN | random start, same seeds | 1,2,3 | 0.410 (0.0052) | 0.983 (0.0091) | 1.02 (0.029) | 0.010 (6.0e-04) |  |
| 1000 | PINN | started from the $Re = 400$ PINN | 1,2,3 | 0.325 (0.076) | 0.705 (0.245) | 0.947 (0.144) | 1.27 (0.794) | 1.00, 0.67, 0.71 |
| 10 | PINN | random start, same seeds | 1,2,3 | 0.159 (0.0087) | 0.194 (0.0013) | 0.522 (7.1e-04) | 0.020 (7.8e-04) |  |
| 10 | PINN | started from the $Re = 100$ PINN | 1,2,3 | 0.155 (0.0058) | 0.197 (0.0027) | 0.503 (0.0019) | 0.016 (0.0018) | 0.89, 1.02, 1.02 |

The $Re = 1000$ runs start from the S6 $Re = 400$ runs of the same seed. Sources: `experiments/results/continuation_re*.csv`, `experiment_results*.csv`.
<!-- /table:s6 -->

**Figures.** `analysis_s6_continuation.png`, `analysis_s6_continuation_re1000.png`.

**Caveats for S4–S6.**
- S4–S6 were added after the first $Re = 400$ result. Each was planned before its own runs and none changed an existing run.
- S6 runs train for more steps in total than the main runs (40,000 at $Re = 400$ and 10, 60,000 at $Re = 1000$, counting the starting network). At the same total of 40,000 steps, the S5 runs at $Re = 400$ reach 0.212 (S5 table, from the evaluation log), against 0.152 for the warm start (5 seeds), so the warm start helps beyond the extra steps.
- Three seeds or fewer in every study.
