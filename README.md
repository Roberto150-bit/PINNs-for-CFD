# PINNs-for-CFD

Senior thesis project exploring computational physics and machine learning, with a focus on comparing traditional numerical methods, data-driven neural networks, and physics-informed neural networks (PINNs).

The project studies two-dimensional incompressible Navier–Stokes flow using the lid-driven cavity problem as a controlled computational benchmark. Fluid is modeled within a square domain where the top wall moves horizontally while the remaining walls remain stationary, producing a characteristic circulating vortex flow.

The thesis compares three approaches to modeling this system: a traditional finite-difference numerical solver, a data-driven neural network trained on numerical reference data, and a physics-informed neural network that incorporates the governing Navier–Stokes equations and physical constraints into training. A fourth model, a hybrid network trained on both the reference data and the physics loss under the same settings, tests whether the two sources of information can be combined. The goal is to evaluate how these approaches differ in their ability to reproduce the underlying flow field and to investigate the strengths and limitations of PINNs as an alternative or complement to conventional computational fluid dynamics methods.

## Repository Structure

- ```models/```: the finite-difference solver (`finite_difference/`), the data-driven neural network (`data_driven_nn/`), the PINN (`pinn/`), the hybrid network (`hybrid/`) and the code they share (`common/`: network, dataset, settings, residuals). Each model folder has a README with its options and outputs.
- ```experiments/```: the experiment runs (one folder per Reynolds number), the batch runner, evaluation and figure scripts, and the generated result tables. See `experiments/README.md`.
- ```tests/```: checks of the network, the PDE residuals, the PINN losses, the causal weights, the evaluator and the hybrid's shared settings (`python -m tests.test_hybrid`, etc.).
- ```notebooks/```: Exploratory analysis, testing, visualization, and computational work.
- ```notes/```: Weekly research notes, the methodology draft and the findings summary.
- ```thesis/```: Thesis drafts and final written materials.

## Reproducing the results

From the repository root, with the environment recorded in `experiments/experiment_001_re100/environment.json`:

```
python models/finite_difference/src/run_solver.py --n 64 --re 100 --mode transient   # FD reference
python -m models.common.dataset --re 100 --build          # dataset from the 64 x 64 FD transient
python experiments/run_batch.py --re 100                  # data NN and PINN, seeds 1-5, and FD timing
python experiments/run_batch.py --hybrid                  # hybrid network
python experiments/analysis/evaluate_all.py --re 100      # score every run
python experiments/analysis/aggregate.py                  # summary tables
python experiments/analysis/training_analysis.py          # analysis figures
```

The other Reynolds numbers, the studies S1-S6 and the hybrid control are listed with their commands in `experiments/README.md`.

**NOTE:** This repository will evolve as the thesis progress.

