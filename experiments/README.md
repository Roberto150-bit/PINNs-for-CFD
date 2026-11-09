# Experiments

This folder stores the experimental runs used to compare the finite-difference (FD) solver, the data-driven neural network, the PINN and the hybrid network (data loss + physics loss) on the lid-driven cavity.

Each experiment is a self-contained record: shared configuration, computing environment, method configurations, raw outputs, evaluation metrics and computational performance. The run folders are the source of all results; the CSV tables and figures are generated from them.

## Folders

| Folder | Content |
|---|---|
| `experiment_001_re100/` | $Re = 100$: main runs (all three networks, seeds 1-5), determinism cost, sensitivity studies S1-S3 (`sensitivity/`), S5 (`extended/`), hybrid data-only control (`control/`) |
| `experiment_002_re400/` | $Re = 400$: main runs (seeds 1-5), S5 (`extended/`, PINN and hybrid), causal weighting S4 (`causal/`), $Re$ continuation S6 (`continuation/`), hybrid control (`control/`) |
| `experiment_003_re10/` | $Re = 10$: Reynolds trend (FD and all three networks), S6 from $Re = 100$ (`continuation/`), hybrid control (`control/`) |
| `experiment_004_re1000/` | $Re = 1000$: Reynolds trend (FD and all three networks), S6 from the S6 $Re = 400$ runs (`continuation/`), S5 (`extended/`, PINN and hybrid), hybrid control (`control/`) |
| `benchmarks/` | Marchi et al. (2009) and Hou et al. (1995) tables |
| `data/` | space-time datasets built from the $64 \times 64$ FD transient, one per $Re$ |
| `analysis/` | evaluation and figure scripts; figures in `analysis/plots/` |
| `results/` | generated tables (CSV / JSON) |

Inside an experiment folder each method has its own subfolder (`finite_difference/`, `data_driven_nn/`, `pinn/`, `hybrid/`) with one folder per run, and `logs/` holds the batch logs. A network run folder contains `config.json`, `history.csv`, `eval_history.csv`, `metrics.json`, `performance.json`, `predictions.npz`, `model.keras` and, on the GPU, `gpu_monitor.csv`.

Run ids follow `<method>_re<Re>_<device>_s<seed>[_<tag>]`, e.g. `pinn_re400_gpu_s2` or `dnn_re100_gpu_s1_frac10` (S1, 10% of the training rows), `hybrid_re400_gpu_s1_dataonly` (hybrid control) or `pinn_re400_gpu_s1_from_re100` (S6). A study keeps its runs in a subfolder of the experiment, with the method subfolders inside it.

## Studies

| Study | Question | Runs | Folder |
|---|---|---|---|
| S1 | training-data size, data NN | 1% and 10% of the training rows, seeds 1-3, $Re = 100$ | `experiment_001_re100/sensitivity/` |
| S2 | collocation points, PINN | 5,000 and 10,000 points, seeds 1-3, $Re = 100$ | `experiment_001_re100/sensitivity/` |
| S3 | network size, data NN and PINN | 4 x 50 and 6 x 100, seeds 1-3, $Re = 100$ | `experiment_001_re100/sensitivity/` |
| S4 | causal weighting of the PDE residual, PINN | seeds 1-3, $Re = 400$ | `experiment_002_re400/causal/` |
| S5 | longer training (100,000 steps) | data NN and PINN at $Re = 100$, PINN at $Re = 400$ and $1000$, hybrid at $Re = 400$ and $1000$ | `experiment_00X/extended/` |
| S6 | $Re$ continuation: PINN started from the trained PINN at a neighbouring $Re$ | $Re = 400$ from $100$ (seeds 1-5), $Re = 1000$ from S6 $Re = 400$ and $Re = 10$ from $100$ (seeds 1-3) | `experiment_00X/continuation/` |
| control | hybrid with the data loss only, to separate the physics loss from the output units | seeds 1-3 at $Re = 100$ and $400$ (`--hybrid-control`) and at $Re = 10$ and $1000$ (`--day`) | `experiment_00X/control/` |


## Running

All commands run from the repository root.

```
python experiments/run_batch.py [--re 400] [--dry-run]   # main batch for one Re (data NN, PINN, FD timing)
python experiments/run_batch.py --hybrid                 # hybrid network, Re 10-1000
python experiments/run_batch.py --hybrid-control         # hybrid control (data loss only), Re 100 and 400
python experiments/run_batch.py --day                    # seeds 4-5 at Re 10 and 1000, controls at Re 10 and 1000, more S5 runs
python experiments/run_batch.py --sensitivity            # S1-S3 at Re 100
python experiments/run_batch.py --extended               # S5, 100,000 steps
python experiments/run_batch.py --extended-trend         # S5 extension: PINN at Re 1000 and 10
python experiments/run_batch.py --trend fd               # FD at Re 10 and 1000
python experiments/run_batch.py --trend nn [--trend-re 10]
python experiments/run_batch.py --causal                 # S4 at Re 400
python experiments/run_batch.py --continuation           # S6: Re 400 started from Re 100
python experiments/run_batch.py --continuation 1000      # S6: Re 1000 started from the S6 Re 400 runs
python experiments/run_batch.py --continuation 10        # S6: Re 10 started from Re 100
```

`--no-start-after HH:MM` starts no new job after that clock time.

A job is skipped when its run folder already holds a `performance.json`, so a batch can be restarted safely.

Runs started with `--shared-gpu` trained while another job used the same GPU. Their `config.json` records `"timing_valid": false`; their accuracy is valid, their times are not used as cost results.

At $Re = 1000$ the $64 \times 64$ FD grid is not converged, so the reference field is the $127 \times 127$ solution taken at every second grid point, which are exactly the $64 \times 64$ points:

```
python models/finite_difference/src/subsample_reference.py --re 1000 --source-n 127
```

## Evaluation and tables

```
python experiments/analysis/fd_validation.py --re 400 --exp-dir experiments/experiment_002_re400
python experiments/analysis/fd_transient_convergence.py --re 400 --exp-dir experiments/experiment_002_re400
python experiments/analysis/evaluate_all.py --re 100
python experiments/analysis/training_analysis.py [--only reynolds_trend]
```

A study folder is scored by pointing `evaluate_all.py` at it, for example

```
python experiments/analysis/evaluate_all.py --re 400 --exp-dir experiments/experiment_002_re400/continuation --out experiments/results/continuation_re400.csv
```

which is how `results/causal_re400.csv`, `continuation_re*.csv`, `control_re*.csv` and `extended_re*.csv` are made.

`evaluate_all.py` scores every run against the FD reference on test A, test B and the full field, and adds PDE residuals, boundary and initial-condition errors and cost. It writes `results/experiment_results.csv` for $Re = 100$ and `results/experiment_results_re<Re>.csv` otherwise. `training_analysis.py` writes the `analysis/plots/analysis_*.png` figures, each with a `.values.json` holding the plotted numbers, and `results/reynolds_trend.csv`.
