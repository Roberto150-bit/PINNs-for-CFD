# Experiments

This folder stores the experimental runs used to compare the finite-difference (FD) solver, the data-driven neural network and the PINN on the lid-driven cavity.

Each experiment is a self-contained record: shared configuration, computing environment, method configurations, raw outputs, evaluation metrics and computational performance. The run folders are the source of all results; the CSV tables and figures are generated from them.

## Folders

| Folder | Content |
|---|---|
| `experiment_001_re100/` | $Re = 100$: main runs, determinism cost, sensitivity studies S1-S3 (`sensitivity/`), longer training S5 (`extended/`) |
| `experiment_002_re400/` | $Re = 400$: main runs, S5 (`extended/`), causal weighting S4 (`causal/`), $Re$ continuation S6 (`continuation/`) |
| `experiment_003_re10/` | $Re = 10$: Reynolds trend (FD, data NN and PINN, seeds 1-3) |
| `experiment_004_re1000/` | $Re = 1000$: Reynolds trend (FD, data NN and PINN, seeds 1-3), $Re$ continuation S6 from $Re = 400$ (`continuation/`, seeds 1-2) |
| `benchmarks/` | Marchi et al. (2009) and Hou et al. (1995) tables |
| `data/` | space-time datasets built from the $64 \times 64$ FD transient, one per $Re$ |
| `analysis/` | evaluation and figure scripts; figures in `analysis/plots/` |
| `results/` | generated tables (CSV / JSON) |

Inside an experiment folder each method has its own subfolder (`finite_difference/`, `data_driven_nn/`, `pinn/`) with one folder per run, and `logs/` holds the batch logs. A network run folder contains `config.json`, `history.csv`, `eval_history.csv`, `metrics.json`, `performance.json`, `predictions.npz`, `model.keras` and, on the GPU, `gpu_monitor.csv`.

Run ids follow `<method>_re<Re>_<device>_s<seed>[_<tag>]`, e.g. `pinn_re400_gpu_s2` or `dnn_re100_gpu_s1_frac10` (S1, 10% of the training rows).

## Running

All commands run from the repository root.

```
python experiments/run_batch.py [--re 400] [--dry-run]   # main batch for one Re
python experiments/run_batch.py --sensitivity            # S1-S3 at Re 100
python experiments/run_batch.py --extended               # S5, 100,000 steps
python experiments/run_batch.py --trend fd               # FD at Re 10 and 1000
python experiments/run_batch.py --trend nn [--trend-re 10]
python experiments/run_batch.py --causal                 # S4 at Re 400
python experiments/run_batch.py --continuation           # S6: Re 400 started from Re 100
python experiments/run_batch.py --continuation 1000      # S6: Re 1000 started from the S6 Re 400 runs
```

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

`evaluate_all.py` scores every run against the FD reference on test A, test B and the full field, and adds PDE residuals, boundary and initial-condition errors and cost. It writes `results/experiment_results.csv` for $Re = 100$ and `results/experiment_results_re<Re>.csv` otherwise. `training_analysis.py` writes the `analysis/plots/analysis_*.png` figures, each with a `.values.json` holding the plotted numbers, and `results/reynolds_trend.csv`.
