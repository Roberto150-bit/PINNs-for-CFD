# Findings summary

Lid-driven cavity, $(x, y, t) \rightarrow (u, v, p)$ on $t \in [0, 5]$. Three methods are compared against a finite-difference (FD) reference on the same $64 \times 64$ grid:
- a data-driven neural network (data NN), trained on FD data;
- a physics-informed neural network (PINN), trained on the equations only;
- a hybrid network, trained on both.

All networks use the same $5 \times 64$ architecture, optimizer and 20,000 training steps. Errors are test-A relative $L_2$ errors at the final training step. Means are over seeds 1–5 at every Reynolds number, unless a table says otherwise. Sources: `experiments/results/experiment_results*.csv` (main runs), `sensitivity_re100.csv`, `causal_re400.csv`, `continuation_re*.csv` and `poster_numbers.json`. The method is described in `notes/methodology_draft.md`.

## RQ1: Accuracy

**Finding.** The data NN reproduces the FD field to about 2% in $u$ at every Reynolds number. The PINN is about five times less accurate at $Re = 100$. With the same settings it converges to a wrong flow at $Re = 400$ and 1000.

| $Re$ | Data NN $u$ / $v$ / $p$ | PINN $u$ / $v$ / $p$ |
|---|---|---|
| 10 | 0.018 / 0.019 / 0.119 | 0.161 / 0.196 / 0.524 |
| 100 | 0.017 / 0.018 / 0.113 | 0.082 / 0.166 / 0.265 |
| 400 | 0.023 / 0.025 / 0.161 | 0.357 / 0.905 / 0.889 |
| 1000 | 0.016 / 0.019 / 0.020 | 0.416 / 0.984 / 1.022 |

At unseen times (test B, ten held-out snapshots) the errors are similar: 0.014 (data NN) and 0.078 (PINN) for $u$ at $Re = 100$.

**Figures.**
- `analysis_reynolds_trend.png` (error vs $Re$, every seed)
- `analysis_error_vs_t.png` (error over time)
- `analysis_error_map_re*.png` (where in the cavity the error is)

**Caveats.**
- The data NN is trained on the same FD solution it is scored against: test A interpolates between training points, and test B between training snapshots. It does not show how the data NN would do without a solver run.
- At $Re = 1000$ the reference is a $127 \times 127$ solve, because the $64 \times 64$ FD solution itself is 0.251 off [`experiment_results_re1000.csv`].
- Pressure errors are larger for both networks; the FD pressure has sharp peaks at the lid corners.

## RQ2: Physical consistency

**Finding.** The PINN satisfies the equations far better, even where it is inaccurate. The data NN is accurate but its fields are not divergence free. A small physics residual does not mean the flow is right: at $Re = 400$ and 1000 the PINN has its smallest residuals while its flow is wrong.

| $Re$ | Continuity residual (RMS), data NN | PINN | Ratio |
|---|---|---|---|
| 10 | 0.569 | 0.018 | 31× |
| 100 | 0.546 | 0.071 | 8× |
| 400 | 0.548 | 0.029 | 19× |
| 1000 | 0.338 | 0.012 | 29× |

- The x-momentum residual of the data NN is 5.6 at $Re = 100$ and 51.6 at $Re = 10$, against 0.096 and 0.054 for the PINN.
- On the grid stencil at $Re = 100$, the continuity RMS is 0.055 (data NN), 0.023 (PINN) and 0.00015 (FD reference) [`poster_numbers.json`].
- The reverse holds for the boundary conditions: the PINN's boundary error (RMS 0.017 at $Re = 100$, 0.039 at $Re = 10$) is larger than the data NN's (0.005).

**Figures.** `analysis_reynolds_trend.png` (lower panel) and the poster physics figure (`v4_physics__*.png`).

**Caveats.** The network residuals are computed by automatic differentiation at 50,000 points not used in training. They measure how well each network satisfies the equations, not how close it is to the FD solution. Being averages over random points, they can be dominated by the few points near the two lid corners, where the wall velocity jumps and the exact solution is singular. Without the points within 0.05 of a lid corner, the order PINN < hybrid < data NN is unchanged at every $Re$ [`experiments/analysis/plots/analysis_corner_excluded.csv`, added after the S5 and S6 results were seen].

## RQ3: Repeatability and robustness

**Finding.**
- Across seeds, both networks are repeatable at $Re = 100$:
  - data NN $u$ error 0.0154–0.0181 (s.d. 0.0010);
  - PINN 0.072–0.100 (s.d. 0.011).
- Across Reynolds numbers, only the data NN is robust: 0.016–0.023. The PINN goes from 0.082 at $Re = 100$ to 0.161 at $Re = 10$ and 0.36–0.42 at $Re \ge 400$. All seeds fail in the same way, so the failure is systematic, not bad luck.
- The data NN's test error shows occasional short-lived rises during training, from which it recovers. The PINN's error decreases smoothly. The spike table is in `experiments/analysis/plots/analysis_spikes.csv` (definition in the methodology draft, Section 5); its counts are not yet in the results tables.

**What changes the PINN at $Re = 400$** (seeds 1–3, $u$ error, same-seed standard runs 0.349):

| Study | Change | $u$ error |
|---|---|---|
| S4 | causal weighting in time | 0.349 (no effect) [`causal_re400.csv`] |
| S5 | 5× longer training (100,000 steps) | 0.212 at 40,000 steps, 0.141 at 100,000 [`poster_numbers.json`] |
| S6 | start from the trained $Re = 100$ PINN (40,000 steps in total) | 0.153 ($v$ 0.294, $p$ 0.425, against 0.883 and 0.859) [`poster_numbers.json`]; with all 5 seeds 0.152 against 0.356, every seed improving [`continuation_re400.csv`] |

At the same total of 40,000 steps, the warm start (0.153) is better than continued training from random weights (0.212). The Re 400 failure is therefore mostly an optimization problem.

Longer training raises the continuity residual measured at the 50,000 evaluation points, but only at the two lid corners. After 100,000 steps the three S5 PINNs at $Re = 400$ have residuals of 1.21, 1.82 and 1.63 there, against 0.015–0.060 after 20,000 steps [`extended_re400.csv`, `experiment_results_re400.csv`]; the three hybrids trained for 100,000 steps (seeds 1–3) go from 0.038–0.046 to 0.48–1.02, and the three 100,000-step PINNs at $Re = 1000$ from 0.010–0.011 to 0.39–1.87 [`extended_re1000.csv`]. The S6 warm-start runs show the same (1.11–1.49 at $Re = 400$) [`continuation_re400.csv`]. A check of where the residual lies showed that it is concentrated within about 0.03 of the two corners where the moving lid meets the side walls, where the boundary velocity jumps from 1 to 0 and the exact solution is singular. Excluding the points within 0.05 of either lid corner (0.4% of the domain; a check added after these values were seen), the long-trained PINNs have continuity residuals of 0.008–0.009 (S5) and 0.008–0.012 (S6), lower than the 20,000-step PINNs (0.011–0.013), and the data NN 0.29–0.36; the long-trained hybrids 0.018–0.020, against 0.023–0.030 after 20,000 steps [`experiments/analysis/plots/analysis_corner_excluded.csv`]. So longer training improves consistency in the flow and sharpens the corner singularity, which then dominates the average. The hybrid gains less accuracy from longer training than the PINN at $Re = 400$: mean $u$ 0.065 to 0.046, with per-seed ratios 0.57, 1.01 and 0.61 [`extended_re400.csv`, `experiment_results_re400.csv`]. At $Re = 1000$ the corner-excluded residual of the long-trained PINNs is about unchanged in seeds 1 and 2 (0.009 and 0.011, against 0.009) and higher in seed 3 (0.025 against 0.010), and they gain little accuracy ($u$ 0.410 to 0.375, per-seed ratios 0.94, 0.99, 0.81; $v$ 0.89, $p$ 1.04). The one hybrid trained for 100,000 steps at $Re = 1000$ (seed 1) improves strongly: $u$ 0.078 to 0.027, $v$ 0.105 to 0.030, $p$ 0.146 to 0.042, continuity without the lid corners 0.030 to 0.023 (one seed) [`extended_re1000.csv`, `experiment_results_re1000.csv`, `analysis_corner_excluded.csv`]. The S5 runs at $Re = 100$ have not been scored for physics.

Results at other $Re$:
- **S6 at $Re = 1000$:** final $u$ 0.412, 0.277 and 0.285, against 0.413, 0.413 and 0.404 from random weights. Two of three seeds improve [`continuation_re1000.csv`].
- **S6 at $Re = 10$:** no clear effect. The $u$ errors are 0.150, 0.155 and 0.161 against 0.168, 0.151 and 0.158 from random weights, so only one seed improves. $p$ is slightly lower and $v$ slightly higher in every seed [`continuation_re10.csv`, `experiment_results_re10.csv`].
- **S5 at $Re = 100$:** the data NN improves from 0.017 to 0.011 with 100,000 steps, the PINN barely (0.084 to 0.082) [`poster_numbers.json`].

**Sensitivity at $Re = 100$** (3 seeds, `sensitivity_re100.csv`):
- With 1% of the data the data NN (0.133) is worse than the PINN without data (0.082); with 10% it reaches 0.050.
- A PINN with 5,000 collocation points has a continuity residual of 1.26 at the evaluation points, mostly from the lid corners: without the points within 0.05 of a lid corner it is 0.028, against 0.017 with 20,000 points [`analysis_corner_excluded.csv`].
- A larger network ($6 \times 100$) does not help either method. One data NN seed ended on a loss spike (0.102).

**Figures.**
- `analysis_repeatability.png` and `analysis_three_methods.png` (error vs step, every seed)
- `analysis_s4_causal.png`, `analysis_s6_continuation.png`, `analysis_s6_continuation_re1000.png`
- `analysis_s1_data.png`, `analysis_s2_collocation.png`, `analysis_s3_size.png`

**Caveats.**
- Only three seeds in the studies.
- S4–S6 were added after the first $Re = 400$ result. Each was fixed before its own runs, and none changed an existing run.
- For S5 at $Re = 100$ only $u$ is available; the physics residuals were scored at $Re = 400$ only.

## RQ4: Cost ($Re = 100$, one laptop)

**Finding.** On this problem FD is much cheaper than training either network. Training only pays off when the trained network replaces many FD solves.

| | Time |
|---|---|
| FD, one $64 \times 64$ transient solve (CPU, median of repeats) | 3.93 s |
| Data NN training (GPU) | 137 s (CPU: 269 s) |
| PINN training (GPU) | 782 s |
| Full-field inference, either network (GPU) | 0.010 s |

- **Break-even:** about 36 full-field evaluations for the data NN (its cost includes the FD solve that produces its data) and 200 for the PINN [`poster_numbers.json`].
- **PINN step cost:** a PINN training step costs 0.039 s on the GPU and 0.297 s on the CPU.
- **FD scaling:** FD solve time grows about linearly with the number of grid points (exponent 1.06).

**Figures.** The poster cost figure (`v4_cost__*.png`).

**Caveats.**
- All timings come from a single laptop. FD ran on the CPU and the networks on the GPU.
- Runs that shared the GPU with another job are excluded from cost.
- At $Re = 1000$ a fair cost has to count the $127 \times 127$ solve the reference needs.
- The break-even counts only repeated evaluations of the same flow. A network trained for one $Re$ does not replace an FD solve at another.

## Hybrid network (data + physics loss)

**Finding.** At every Reynolds number the hybrid lies between the other two networks in accuracy. Unlike the PINN, its error hardly changes with $Re$. At $Re = 400$ and 1000, where the PINN converges to a wrong flow, the hybrid recovers the flow, and its continuity residual at the evaluation points stays close to the PINN's (away from the lid corners it is about twice the PINN's, 0.026 against 0.012 at $Re = 400$, seeds 1–3 [`analysis_corner_excluded.csv`]). It does not reach the data NN's accuracy at any Reynolds number.

Test-A error of $u$ and continuity residual, seeds 1–3, the same seeds for all three networks [`experiment_results*.csv`]:

| $Re$ | $u$: data NN / PINN / hybrid | Continuity: data NN / PINN / hybrid |
|---|---|---|
| 10 | 0.019 / 0.159 / 0.058 | 0.584 / 0.020 / 0.083 |
| 100 | 0.017 / 0.084 / 0.050 | 0.547 / 0.080 / 0.173 |
| 400 | 0.023 / 0.349 / 0.065 | 0.549 / 0.038 / 0.041 |
| 1000 | 0.017 / 0.410 / 0.079 | 0.352 / 0.010 / 0.054 |

All metrics at $Re = 100$ and 400:

| | $Re = 100$: data NN / PINN / hybrid | $Re = 400$: data NN / PINN / hybrid |
|---|---|---|
| Test A, $u$ | 0.017 / 0.084 / 0.050 | 0.023 / 0.349 / 0.065 |
| Test A, $v$ | 0.018 / 0.168 / 0.053 | 0.025 / 0.883 / 0.074 |
| Test A, $p$ | 0.113 / 0.266 / 0.170 | 0.158 / 0.859 / 0.231 |
| Test B, $u$ | 0.015 / 0.079 / 0.045 | 0.020 / 0.345 / 0.053 |
| Continuity residual | 0.547 / 0.080 / 0.173 | 0.549 / 0.038 / 0.041 |
| x-momentum residual | 5.62 / 0.110 / 1.01 | 2.08 / 0.034 / 0.177 |
| Boundary RMS | 0.005 / 0.017 / 0.017 | 0.005 / 0.030 / 0.022 |
| Training time, GPU (s) | 138 / 782 / 872 | 144 / 778 / 874 |

The hybrid costs slightly more to train than the PINN, because it computes both losses at every step.

**Figures.**
- `analysis_three_methods.png` (error vs training step)
- `analysis_error_vs_t.png` and `analysis_error_map_re100.png`, `analysis_error_map_re400.png`
- `analysis_reynolds_trend.png`
- `analysis_s4_causal.png` (the hybrid next to the causal PINN at $Re = 400$)

**Caveats.**
- Three seeds per Reynolds number in the tables. Of seeds 4–5 at $Re = 100$, seed 4 is in line (test-A $u$ 0.037). Seed 5 ends on a spike exactly at the last step (spike table), so its reported result, 0.118, is worse than the PINN's for that seed (0.076). At $Re = 400$ seeds 4 and 5 are in line (0.054, 0.080). Over all five seeds the hybrid's test-A $u$ error is 0.061 (s.d. 0.008) at $Re = 10$, 0.061 (s.d. 0.032) at $Re = 100$, 0.066 (s.d. 0.012) at $Re = 400$ and 0.082 (s.d. 0.012) at $Re = 1000$; at $Re = 10$ seed 5 is the highest (0.076) without a spike. At $Re = 100$ that is the largest seed-to-seed spread of the three networks (data NN s.d. 0.0010, PINN 0.011), entirely because of seed 5. [`experiment_results*.csv`; spike table `experiments/analysis/plots/analysis_spikes.csv`]
- The hybrid outputs physical $u$, $v$, $p$, while the data NN outputs standardized values. A data-only control (the hybrid code without the physics loss, seeds 1–5) separates the two effects [`control_re*.csv`; figures `analysis_hybrid_control.png`, `analysis_hybrid_control_re10_re1000.png`]:

| $Re$ | Test-A $u$: data NN / control / hybrid | Continuity: data NN / control / hybrid |
|---|---|---|
| 10 | 0.018 / 0.021 / 0.061 | 0.569 / 0.697 / 0.078 |
| 100 | 0.017 / 0.030 / 0.061 | 0.546 / 0.477 / 0.159 |
| 400 | 0.023 / 0.053 / 0.066 | 0.548 / 0.357 / 0.075 |
| 1000 | 0.016 / 0.092 / 0.082 | 0.338 / 0.257 / 0.052 |

  The cost of the output units grows with $Re$: the control's $u$ error is 1.2, 1.8, 2.3 and 5.6 times the data NN's at $Re = 10$, 100, 400 and 1000. The effect of the physics loss on accuracy shrinks with $Re$: against the control it multiplies the $u$ error by 2.9, 2.1 and 1.24 at $Re = 10$, 100 and 400, and at $Re = 1000$ it lowers it (0.90; lower in four of the five seeds, higher in seed 4: 0.099 against 0.082). At every $Re$ the physics loss lowers the continuity residual, by 3.0 to 9.0 times. The $Re = 100$ factor includes hybrid seed 5, which ends on a spike. With the first three seeds only, the factors were 1.1, 1.7, 2.3 and 5.3 (output units) and 2.8, 1.7, 1.25 and 0.86 (physics loss); seeds 4 and 5 of the control were run last, with the same settings. So at low $Re$ the hybrid's gap to the data NN comes almost entirely from the physics loss, at high $Re$ from the output units.
- The loss weights were not tuned: all start at 1, as in the PINN. In all twenty main runs the data term was the largest at step 2,000, 15 to 53 times the smallest term. That is below the rebalancing threshold of 100, so the weights never changed [each run's `config.json`].
