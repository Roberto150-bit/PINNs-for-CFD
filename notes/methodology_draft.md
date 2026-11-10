# Methodology (draft)

This draft describes how three neural-network (NN) methods were compared with a finite-difference (FD) reference on the lid-driven cavity: a data-driven NN trained on FD data, a physics-informed neural network (PINN) trained on the governing equations only, and a hybrid network trained on both. Every number is followed by the file it comes from, in square brackets. Paths are relative to the repository root.

## 1. Problem and reference solution

### 1.1 Flow problem

The test case is two-dimensional incompressible flow in a unit square cavity, $(x, y) \in [0, 1]^2$, driven by the top wall (the lid) [`models/common/config.py`]. The density is $\rho = 1$ and the lid speed $U = 1$, so the kinematic viscosity is $\nu = 1/Re$. The flow is solved as a time-dependent problem on $t \in [0, 5]$, starting from rest. The governing equations are the unsteady incompressible Navier–Stokes equations:

$$ \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0 $$

$$ \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) $$

$$ \frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{\partial p}{\partial y} + \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right) $$

All walls are no-slip. The lid velocity is ramped up smoothly instead of started impulsively,

$$ u_{lid}(t) = \tanh(10 t), \qquad v_{lid} = 0, $$

so that the initial condition is continuous and a smooth network can represent it. The FD solver uses the same ramp, so all methods solve exactly the same problem [`models/common/config.py`, `LID_RAMP_RATE`]. Four Reynolds numbers are studied: $Re = 10$, 100, 400 and 1000 [`models/common/config.py`, `EXPERIMENT_DIRS`]. $Re = 100$ is the main case, on which the method settings were fixed.

### 1.2 Finite-difference solver

The reference solutions come from an FD solver on a collocated uniform grid [`models/finite_difference/src/`]. The solver used for the comparison differs from the original course solver in five options, all selectable from the command line. The original behaviour can still be reproduced exactly [`models/finite_difference/README.md`]:

| Option | Original solver | Solver used for the comparison |
|---|---|---|
| Advection | first-order backward differences | second-order central differences |
| Pressure Poisson equation | 50 Jacobi iterations per step | direct sparse LU (lower-upper) factorization |
| Pressure boundary condition | $p = 0$ on the lid | $\partial p / \partial n = 0$ on all walls, zero mean pressure |
| Time stepping | explicit pressure-source scheme | incremental pressure-correction projection |
| Lid start | impulsive | $\tanh(10t)$ ramp |

The reasons for these changes are documented in `notes/week-04.md` (items 5 to 8). In short:
- the $p = 0$ lid condition converged to a wrong steady flow;
- the original time scheme had a steady state that depended on the time step;
- the Jacobi iterations left large Poisson residuals on fine grids.

Diffusion is treated explicitly, so the time step must satisfy $\Delta t \le \Delta x^2 / (4\nu)$ [`models/finite_difference/README.md`].

The reference grid is $64 \times 64$ points ($\Delta x = 1/63$). Fields are stored every 0.1 time units, giving 51 snapshots at $t = 0, 0.1, \dots, 5.0$ [`models/common/config.py`]. The reference runs use:

| $Re$ | Grid solved on | $\Delta t$ | Steps | Source |
|---|---|---|---|---|
| 10 | $64 \times 64$ | $2.5 \times 10^{-4}$ | 20,000 | `experiments/experiment_003_re10/finite_difference/fd_re10_n64_transient/config.json` |
| 100 | $64 \times 64$ | $10^{-3}$ | 5,000 | `experiments/experiment_001_re100/finite_difference/fd_re100_n64_transient/config.json` |
| 400 | $64 \times 64$ | $10^{-3}$ | 5,000 | `experiments/experiment_002_re400/finite_difference/fd_re400_n64_transient/config.json` |
| 1000 | $127 \times 127$, stored at the $64 \times 64$ points | $10^{-3}$ | 5,000 | `experiments/experiment_004_re1000/finite_difference/fd_re1000_n127to64_transient/config.json` |

At $Re = 1000$ the $64 \times 64$ grid is not converged: its steady centerline velocity deviates from the benchmark by up to 0.165, against 0.030 on the $127 \times 127$ grid [`experiments/results/fd_validation_re1000.json`]. The reference is therefore the $127 \times 127$ solution taken at every second grid point. Because $2 \times 63 + 1 = 127$, these points are exactly the $64 \times 64$ grid points, so no interpolation is involved [`models/finite_difference/src/subsample_reference.py`]. All methods are scored on the same $64 \times 64$ points at every $Re$.

### 1.3 Validation of the reference

The steady solution of the solver was compared with two published benchmarks:
- Marchi, Suero and Araki (2009): centerline velocities and primary-vortex centre at $Re = 10$, 100, 400 and 1000 [`experiments/benchmarks/marchi2009.csv`];
- Hou et al. (1995): vortex centre at $Re = 100$ and 400 [`experiments/benchmarks/hou1995.csv`].

On the $64 \times 64$ grid (the $127 \times 127$ grid at $Re = 1000$) the results are [`experiments/results/fd_validation_re{10,100,400,1000}.json`]:

| $Re$ | Grid | max $\lvert u - u_{Marchi} \rvert$ on $x = 0.5$ | max $\lvert v - v_{Marchi} \rvert$ on $y = 0.5$ | Vortex centre distance to Marchi | $\psi_{min}$ (Marchi) |
|---|---|---|---|---|---|
| 10 | 64 | 0.0008 | 0.0009 | 0.0003 | $-0.0997$ ($-0.1001$) |
| 100 | 64 | 0.0056 | 0.0065 | 0.0039 | $-0.1020$ ($-0.1035$) |
| 400 | 64 | 0.041 | 0.044 | 0.0110 | $-0.1069$ ($-0.1140$) |
| 1000 | 127 | 0.030 | 0.034 | 0.0037 | $-0.1137$ ($-0.1189$) |

At $Re = 100$ the vortex centre is also 0.0025 from Hou, and at $Re = 400$ 0.0061. All vortex centres are within one grid spacing ($\Delta x = 0.0159$) of the benchmarks. At $Re = 100$ the centerline error falls by factors of 3.6 and 2.3 from $32$ to $64$ to $128$ points, so the convergence is below second order [`experiments/results/fd_validation_re100.json`]. The likely causes are the first-order wall pressure condition and the singular lid corners.

Two further checks were made on the transient reference:
- **Time step:** halving $\Delta t$ changes the velocity at $t = 5$ by at most $4.3 \times 10^{-6}$ at $Re = 100$ and $3.2 \times 10^{-5}$ at $Re = 400$ [`experiments/results/fd_transient_convergence_re{100,400}.json`].
- **Grid:** the $64 \times 64$ and $128 \times 128$ transients at $Re = 100$ differ by at most 0.142 at $t = 5$, at the lid corners, with an RMS difference in $u$ of $4.6 \times 10^{-3}$ [`experiments/results/fd_transient_convergence_re100.json`].

## 2. Dataset and splits

One dataset is built per Reynolds number from the $64 \times 64$ transient reference. It has $64 \times 64 \times 51 = 208{,}896$ rows of $(x, y, t, u, v, p)$ [`experiments/data/dataset_re100_n64.json`; the same layout at every $Re$]. Two test sets are held out:

- **Test B (prediction at unseen times):** ten whole snapshots, $t = 0.5, 1.0, \dots, 5.0$, never used in training: 40,960 rows.
- **Test A (interpolation within the training snapshots):** 15% of the remaining rows, drawn at random.

The remaining rows (the 41 other snapshots) are split at random 70% / 15% / 15% into training, validation and test A. With split seed 0 the counts are 117,555 training, 25,190 validation, 25,191 test-A and 40,960 test-B rows [`experiments/data/dataset_re100_n64.json`]. The split is identical for every network, every seed and every $Re$. The training set is used only by the methods that learn from data (the data-driven NN and the hybrid). Inputs are scaled to $[-1, 1]$ in each coordinate [`models/common/dataset.py`]. The datasets contain no missing values or repeated points; their target statistics per split, checksums and the exact rows of the reduced training sets of S1 and S7 are listed in the dataset card [`experiments/data/README.md`; split checks in `tests/test_dataset.py`].

## 3. The three networks

### 3.1 Shared settings

The three networks differ only in their loss function. Everything else is shared [`models/common/config.py`, `models/*/src/config.py`; checked by `tests/test_hybrid.py`]:

| Setting | Value |
|---|---|
| Input / output | $(x, y, t)$ scaled to $[-1, 1]$ $\rightarrow$ $(u, v, p)$ |
| Architecture | fully connected, 5 hidden layers of 64 neurons, tanh |
| Trainable parameters | 17,091 |
| Initialization | Glorot uniform, seeded per layer from the run seed |
| Optimizer | Adam, learning rate $10^{-3}$, default Keras parameters |
| Training length | 20,000 optimizer steps, no early stopping (100,000 in study S5) |
| Seeds | 1 to 5 at every $Re$ (1 to 3 in the additional studies) |
| Data split / training points | fixed (seed 0) for every run, so seeds differ only in initialization and, for the data-using networks, batch order |

### 3.2 Data-driven neural network

The data-driven NN learns from the FD training rows only [`models/data_driven_nn/src/train.py`]. Each step uses a mini-batch of 4,096 rows; the order is reshuffled every epoch and the last partial batch is dropped. Targets are standardized with the training-set mean and standard deviation, and the loss is the mean of the three per-output mean squared errors:

$$ L_{data} = \frac{1}{3} \left( \mathrm{MSE}_u + \mathrm{MSE}_v + \mathrm{MSE}_p \right) \quad \text{(standardized)} $$

Predictions are converted back to physical units before evaluation.

### 3.3 Physics-informed neural network (PINN)

The PINN never sees FD data during training [`models/pinn/src/train.py`, `models/pinn/src/losses.py`]. Its loss is

$$ L_{PINN} = \lambda_{ic} L_{ic} + \lambda_{bc} L_{bc} + \lambda_c \overline{r_c^2} + \lambda_m \left( \overline{r_x^2} + \overline{r_y^2} \right) $$

where:
- $r_c$, $r_x$ and $r_y$ are the residuals of continuity and the two momentum equations, computed by automatic differentiation at 20,000 interior collocation points;
- $L_{ic}$ is the squared error of $u$ and $v$ at 4,096 points at $t = 0$;
- $L_{bc}$ is the squared error of $u$ and $v$ at 1,000 points per wall (4,000 in total).

All points are drawn once with a fixed seed and used in full at every step. The two lid corners are excluded, because they belong to both the lid and a side wall [`models/pinn/src/config.py`]. All weights start at $\lambda = 1$. At step 2,000 the weighted loss terms are compared once: if the largest is more than 100 times the smallest, all weights are rescaled so that every weighted term equals their median. This rule was fixed before any run and is recorded in each run's `config.json`.

### 3.4 Hybrid network

The hybrid network adds the two losses [`models/hybrid/src/train.py`]:

$$ L_{hybrid} = \lambda_{data} L_{data} + L_{PINN} $$

- **Data term:** the same training rows, mini-batch size, shuffling and standardized loss as the data-driven NN. For the same seed the batch sequence is identical [`tests/test_hybrid.py`].
- **Physics term:** the same points, loss functions and starting weights as the PINN, with $\lambda_{data} = 1$. The step-2,000 weight check is applied to all five terms.

Because the physics residuals need physical values, the hybrid network outputs $u$, $v$, $p$ in physical units, like the PINN. Its data term divides each output error by the training-set standard deviation, so for a given prediction it has the same value as the data NN loss [`tests/test_hybrid.py`]. The output parametrization, however, differs from the data NN, whose network outputs standardized values. The comparison between the hybrid and the data NN therefore changes both the loss and the output units. The comparison between the hybrid and the PINN changes only the loss.

To separate the two effects, a control run trains the hybrid code with the data loss only (`--data-only`: physics terms not computed, no weight check, physical outputs, same batches) [`models/hybrid/src/train.py`]. Hybrid against control isolates the physics loss; control against data NN isolates the output units. The control was added after the first two hybrid runs at $Re = 100$ had been seen in the training logs. It changes no existing run and is reported whatever it shows.

## 4. Training protocol

- Every run trains for exactly 20,000 steps (100,000 in study S5). **Results are taken at the final step**, also when a run ends on a temporary rise in error; there is no early stopping and no selection of the best checkpoint.
- The test-A error is logged every 500 steps and the loss terms every 10 steps (`eval_history.csv`, `history.csv` in each run folder). These logs are used for monitoring and for the training-curve figures only, never to choose a result.
- Deterministic operations on the graphics processing unit (GPU) are enabled (`tf.config.experimental.enable_op_determinism`). On one run per network this changed the time per step by a factor of 0.87 (data NN) and 1.00 (PINN) [`experiments/experiment_001_re100/determinism_cost/`].
- All main runs were trained on the GPU, one job at a time (the central processing unit, CPU, was used only for timing comparisons and the FD solves). Runs that shared the GPU with another job are flagged in their `config.json` (`"timing_valid": false`); their accuracy is used, their times are not.
- As a sanity check before the main runs, the data NN was trained on 1,000 training points to confirm that it can overfit them (loss reduced at least 100 times within 5,000 steps, MSE of $u$ and $v$ below $10^{-3}$).

## 5. Evaluation metrics

One script scores every run of every method in the same way [`experiments/analysis/evaluate_all.py`]:

- **Accuracy:** the relative $L_2$ error

  $$ e_\phi = \frac{\lVert \phi_{pred} - \phi_{FD} \rVert_2}{\lVert \phi_{FD} \rVert_2}, \qquad \phi \in \{u, v, p\}, $$

  on test A, test B and the full field. Pressure is defined only up to a constant, so each snapshot's mean pressure is subtracted from both fields first. The error is also reported per snapshot (51 values) to show how it develops in time.
- **Physical consistency:**
  - For the networks: the root mean square (RMS) of $r_c$, $r_x$ and $r_y$ by automatic differentiation at 50,000 random interior points that were not used in training (fixed seed).
  - For every method, FD included: the continuity residual by the same second-order central stencil on the $64 \times 64$ grid. It is computed over the full interior and over a core region three points from the walls.
- **Constraints:** the RMS boundary error on the wall nodes (lid corners excluded) over all snapshots, and the RMS initial-condition error of $u$ and $v$ at $t = 0$.
- **Flow structure:** the primary-vortex centre at $t = 5$. It is the minimum of the stream function $\psi(x, y) = \int_0^y u \, dy'$ (trapezoidal rule along each column), refined between grid points by a quadratic fit in each direction [`experiments/analysis/fd_validation.py`, `stream_function`, `vortex_center`]. Distances are given in grid spacings $\Delta x = 1/63$. $\psi_{min}$ measures the vortex strength.
- **Stability of training:** the number of spikes in the logged test-A error of $u$. A spike is an evaluation step ($\ge 1{,}000$) where the error is more than twice the best value reached at earlier steps. The definition was fixed before the hybrid results existed [`experiments/analysis/training_analysis.py`, `spike_stats`; table `experiments/analysis/plots/analysis_spikes.csv`].

The evaluator was checked on cases with known answers: FD scored against itself gives exactly 0, and FD with 1% added noise gives relative errors of 0.010 [`notes/week-04.md`, item 18].

## 6. Seeds and repeatability

Each configuration is trained with several seeds:
- five (1 to 5) for every network and the data-only control at every $Re$;
- three (1 to 3) in the additional studies, except where Section 8 lists fewer.

The seed sets the weight initialization and, for the data-using networks, the mini-batch order. The data split and the PINN training points are fixed for every run. Results are reported as the mean over seeds together with every individual seed (or the standard deviation), so that run-to-run variation is visible. With deterministic operations, two runs with the same seed and settings give identical results. For example, the first 20,000 steps of the 100,000-step S5 run at $Re = 400$ (seed 1) reproduce the 20,000-step main run with the same seed: the logged test-A errors of $u$, $v$ and $p$ agree exactly at all 41 evaluation steps (0.3565 for $u$ at step 20,000) [`experiments/experiment_002_re400/extended/pinn/pinn_re400_gpu_s1_steps100k/eval_history.csv`, `experiments/experiment_002_re400/pinn/pinn_re400_gpu_s1/eval_history.csv`].

## 7. Cost measurement

Cost is measured inside each run and stored in its `performance.json` [`models/common/run_record.py`, `models/common/profiling.py`]:

- **Training time:** wall-clock time of the optimizer steps, timed with `time.perf_counter`. The first step, which includes graph tracing and compilation, is reported separately. The periodic evaluations are excluded. Every 10 steps the loss values are copied from the GPU before the clock is read, which forces queued GPU work to finish, so asynchronous execution does not hide time.
- **Inference time:** the median of 20 calls after 3 warm-up calls, for the full $64 \times 64 \times 51$ field and for single points. GPU timings include copying the result to host memory.
- **Memory:** peak resident memory, sampled every 0.5 s, and peak GPU memory as reported by TensorFlow. GPU memory is allocated on demand, so the peak reflects actual use. GPU utilization is logged once per second with `nvidia-smi`.
- **FD cost:** the solve time of the transient FD run on the CPU, as the median of repeated runs [`experiments/results/poster_numbers.json`, `fd_n64_solve_seconds_median` = 3.93 s at $Re = 100$].
- **Cost to a usable field and break-even:**
  - The data NN's cost includes the FD solve that produces its training data; the PINN's cost is its training time [`experiments/results/poster_numbers.json`].
  - The break-even number of field evaluations is the cost to a usable field divided by the time saved per evaluation (FD solve time minus network inference time).

Times are only compared between runs on the same device that did not share it with another job.

## 8. Additional studies

All additional studies use the main settings except for the one factor studied. Each was fixed before its runs and is scored at the final step, against the main runs with the same seeds:

| Study | Question | Change | Runs | Folder |
|---|---|---|---|---|
| S1 | How much data does the data NN need? | 1% and 10% of the training rows | data NN, $Re = 100$, seeds 1–3 | `experiments/experiment_001_re100/sensitivity/` |
| S2 | How many collocation points does the PINN need? | 5,000 and 10,000 instead of 20,000 | PINN, $Re = 100$, seeds 1–3 | same |
| S3 | Does network size matter? | $4 \times 50$ and $6 \times 100$ instead of $5 \times 64$ | both networks, $Re = 100$, seeds 1–3 | same |
| S4 | Does causal weighting in time help the PINN? | residual weighted by time chunk (Wang, Sankaran and Perdikaris, 2022, Algorithm 1, published schedule) | PINN, $Re = 400$, seeds 1–3 | `experiments/experiment_002_re400/causal/` |
| S5 | Are the networks under-trained? | 100,000 steps instead of 20,000 | data NN and PINN at $Re = 100$, PINN at $Re = 400$, seeds 1–3; PINN at $Re = 1000$, seeds 1–3; hybrid at $Re = 400$, seeds 1–3, and at $Re = 1000$, seeds 1–2 (seed 3 planned, not run) | `experiments/experiment_00*/extended/` |
| S6 | Does starting from a solution at a lower $Re$ help the PINN? | initial weights from the trained PINN at the lower $Re$ (same seed), optimizer state fresh | $Re = 400$ from $Re = 100$, seeds 1–5; $Re = 1000$ from the S6 $Re = 400$ runs, seeds 1–3; $Re = 10$ from $Re = 100$, seeds 1–3 | `experiments/experiment_00*/continuation/` |
| S7 | Does the hybrid's physics loss make up for missing data? | 0.1% (118 rows) and 1% (1,176 rows) of the training rows | data NN and hybrid, $Re = 100$ and 400, seeds 1–3 | `experiments/experiment_00{1,2}_re{100,400}/sparse/` |

S4 uses 32 time chunks; the causality parameter steps through $10^{-2}, 10^{-1}, 1, 10, 100$ whenever every chunk weight exceeds 0.99 [`models/pinn/src/config.py`].

S6 runs have more training in total than the main runs: 40,000 steps at $Re = 400$ and 10, and 60,000 at $Re = 1000$. They are therefore also compared with S5 runs at the same total number of steps where those exist.

S7 trains on a fixed subset of the training rows. The subset is drawn with the split seed (0), not the run seed, so it is the same for every seed, both networks and every $Re$; the 0.1% and 1% subsets are separate draws (they share no row), and the 1% subset is the one used in S1 [`experiments/data/README.md`; `tests/test_dataset.py`]. The output scaling of the data loss (Section 3.2, 3.4) uses the mean and standard deviation of the subset. When the subset has fewer than 4,096 rows, every step uses the whole subset as its mini-batch. The hybrid's physics points are unchanged, and the validation and test sets are the full ones. The results at 0.1% and 1% are compared with the main runs (100%) with the same seeds, and with the PINN, which uses no data. The study was planned and its scoring fixed before its runs [`experiments/run_batch.py`, `sparse_jobs`].

The hybrid network (Section 3.4) is run like the other two networks, seeds 1–5 at every Reynolds number, and so is the data-only control (Section 3.4) [`experiments/experiment_00*/control/`].

A final set of runs completes the design so that every network has five seeds at every Reynolds number:
- seeds 4–5 of the data NN, the PINN and the hybrid at $Re = 10$ and 1000 (seeds 1–3 of the data NN and the PINN there shared the GPU, so only their accuracy is used; seeds 4–5 are timed on a free GPU);
- the data-only control at $Re = 10$ and 1000, seeds 1–3, and seeds 4–5 at all four Reynolds numbers;
- the 100,000-step runs listed under S5 for the hybrid and for the PINN at $Re = 1000$ (seeds 2–3).

These runs were planned and their scoring fixed before they started; no setting of an earlier run was changed. The PINN at $Re = 10$ with 100,000 steps, planned under S5, was not run. All of these runs are scored by the same evaluator as the main runs [`experiments/results/experiment_results*.csv`].

## 9. Hardware and software

All runs used one laptop [`experiments/experiment_001_re100/environment.json`]:

| Component | Version / model |
|---|---|
| CPU | Intel Core i5-12450H (8 cores: 4 performance, 4 efficiency; the Linux layer reports 6 physical / 12 logical) |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB, driver 610.78 (CUDA 13.3) |
| RAM | 8.2 GB visible to the Linux layer |
| Operating system | Linux (WSL2, kernel 6.18.54.1) on Windows |
| Python | 3.14.4 |
| TensorFlow | 2.22.0-rc0 |
| Keras | 3.16.0.dev2026110803 (nightly build, pinned in `requirements.txt`) |
| NumPy / SciPy | 2.5.4 / 1.18.1 |

## 10. Deviations and limitations a reader should know

1. The FD solver used for the reference differs from the original course solver in five options (Section 1.2). The original can still be reproduced exactly.
2. The FD reference converges below second order in space (Section 1.3), and the $64 \times 64$ and $128 \times 128$ transients differ by up to 0.142 at the lid corners. Errors of the networks below this level are not meaningful near the corners.
3. At $Re = 1000$ the reference comes from a $127 \times 127$ solve. The $64 \times 64$ FD solution itself has a test-A error of $u$ of 0.251 against this reference [`experiments/results/experiment_results_re1000.csv`]. A cost comparison at $Re = 1000$ has to count the finer solve.
4. The overfitting sanity check (Section 4) was relaxed after its first version failed. The original target was a loss of $10^{-4}$, which the pressure term could not reach because of the lid-corner pressure spikes.
5. Some network runs shared the GPU with another job (flagged in `config.json`). Their accuracy is valid, their times are not used. Some timed runs coincided with operating-system background updates, so their times may be slightly perturbed.
6. The Reynolds-number trend ($Re = 10$, 1000), studies S4–S6 and the hybrid network were added after the first results at $Re = 100$ and 400. Each was planned and its scoring fixed before its own runs. No setting of an existing run was changed after its results were seen. The same holds for the runs that complete five seeds at $Re = 10$ and 1000, the controls at those Reynolds numbers and the extra 100,000-step runs (Section 8), which were added last. The control was added after the first hybrid results had been seen (item 7).
7. The hybrid network outputs physical values, whereas the data NN outputs standardized values (Section 3.4). This is a second difference between the two besides the loss; the data-only control run measures its effect.
8. The weight check at step 2,000 never changed the PINN weights at $Re = 100$: the largest-to-smallest ratios were 21.8, 57.5, 18.3, 66.5 and 46.8 [`notes/week-04.md`, item 17]. Whether it changed weights in other runs is recorded in each `config.json`.
9. The partial differential equation (PDE) residuals are RMS values over 50,000 random points, so a few points with very large residuals can dominate them. This happens at the two corners where the moving lid meets the side walls: the wall velocity jumps from 1 to 0 there, and the exact solution is singular. For the networks trained longer than 20,000 steps (S5, and S6 at $Re = 400$ and 1000), at least 99.9% of the squared continuity residual of the PINNs, and 99.5–99.97% of that of the hybrids, comes from the 0.4% of the points that lie within 0.05 of these corners; for some 20,000-step runs the share is also large (91% for `pinn_re400_gpu_s1`, 99% for `hybrid_re400_gpu_s4`). A version of the residual without the points within 0.02, 0.05 and 0.1 of either lid corner is therefore reported next to the primary value (`experiments/analysis/plots/analysis_corner_excluded.csv`). It was added after the S5 and S6 residuals had been seen; the primary value is unchanged.
