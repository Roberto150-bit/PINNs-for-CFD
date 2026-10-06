# Notes

### Setup:
1. Set up the role of the data-driven neural network for the thesis. It will learn only from CFD data, while the finite-difference solver stays as the baseline and the PINN will learn from the physics equations.

### Neural Networks Setup:
2. Both neural-network models (data and PINN) will use TensorFlow through tf.keras, with model definition, training, evaluation, and configuration kept in separate files. 

### Data-driven NN `config.py`:
3. Categories setup:
Training settings
- EPOCHS
- BATCH_SIZE
- LEARNING_RATE

Model settings
- number of hidden layers
- neurons per layer
- activation function

Reproducibility
- RANDOM_SEED

File/output settings
- where training data comes from
- where results/models are saved

### NNs Setup:
4. Both the data-driven NN and future PINN should save AND create a subfolder in experiments folder each time a run is done:
- trained model (model.keras)
- training history (history.csv)
- predictions (predictions.npz)
- evaluation metrics (metrics.json)
- configuration used for that run (config.json)

5. For model comparisons, runs should use the same shared configuration wherever possible. Before comparing results, the configs should be checked to make sure the problem setup, physical parameters, model architecture, training settings, and evaluation conditions match. 

6. Later, the side-by-side comparison process should automatically verify the shared configuration values before running or comparing experiments so mismatched runs are not compared by mistake.


### Data-Driven Inputs & Outputs:
7. For current solver, we already know the experiment is fixed at Re = 100 and the finite-difference solver produces the CFD solution on the cavity grid. Thus the setup for the data-driven nn is: 

$$
(x, y) 
\rightarrow
(u, v, p)
$$

we are not adding more inputs yet. Since we are setting the baseline for experiments and testing.

8. One input vector is:

$$
\mathbf{x}_{input}
=
[x, y]
$$

9. One output vector is:

$$
\mathbf{y}_{output}
=
[u, v, p]
$$

10. For the current $21 \times 21$ grid: $441$ $samples$

$$
\bold{X}_{shape} 
= 
(441, 2) 
= 
\begin{bmatrix} 
x_1 & y_1 \\ 
x_2 & y_2 \\
\vdots & \vdots \\
x_441 & y_441 \\
\end{bmatrix} 
$$


$$
\bold{Y}_{shape}
=
(441, 3)
=
\begin{bmatrix}
u_1 & v_1 & p_1 \\ 
u_2 & v_2 & p_2 \\
\vdots & \vdots \\
u_441 & v_441 & p_441 \\
\end{bmatrix}
$$

11. Training data includes all grid points, including boundaries.

12. The data model will use: $x$, $y$, $u$, $v$, $p$ from `solver_results.npz`

13. The number that TensorFlow will use to make the network learn is the mean squared error (MSE):

$$
\text{MSE}
=
\frac{1}{N}
\sum_{i=1}^{N}
(y_i - \hat{y})^2
$$

14. Each output component will be separately calculated:

$$
\text{MSE}_u, 
\qquad
\text{MSE}_v,
\qquad
\text{MSE}_p
$$

During trianing the neural network will learn through the combination of the errors:

$$
\mathcal{L}_{data}
=
\frac{MSE_u + MSE_v + MSE_p}{3}
$$

15. Since the three values differ they will add more weight to the loss function in general. For example:

$$
\text{MSE}_u=0.001,
\qquad 
\text{MSE}_v=0.001,
\qquad 
\text{MSE}_p=0.1
$$

Then:

$$
\mathcal{L}_{data}
=
\frac{0.001+0.001+0.1}{3}
=0.034
$$

Thus, $u$, $v$, and $p$ will be calculated in the $MSE_{loss}$, with output normalization used to prevent differences from causing one variable to dominate everything.

16. Evaluation measurements, telling me how well it performed after training:

$MSE$: Emphasizes larger errors and directly connects to out training loss.

$$
MSE
=
\frac{1}{N}
\sum_{i=1}^{N}
(y_i - \hat{y})^2
$$

$MAE$ (Mean Absolute Error): Tells us the average magnitude of the error in a more intuitive way. This is different than the MSE although similar formulas. It's easier to interpreted as the model's typical prediction error.

$$
\text{MAE}
=
\frac{1}{N}
\sum_{i=1}^{N}|y_i-\hat y_i|
$$

$Relative error$: Tells us how large the error is relative to the actual CFD solution, which can make comparisons across $u$, $v$, $p$ easier. 

$$
\text{Relative Error (\%)}
=
\frac{|y-\hat y|}{|y|}\times100
$$

17. Make sure that the data-driven neural network satisfy physical law. For incompressible flow, it should satisfy:

$$
\nabla
\cdot
\mathbf{u}
=
\frac{\partial u}{\partial x}
+
\frac{\partial v}{\partial y}
=0
$$

Thus, we can measure how close the neural network's predicted velocity field comes to satisfying it.

$$
\text{Continuity error}
=
\frac{\partial \hat u}{\partial x}
+
\frac{\partial \hat v}{\partial y}
$$

Ideally
$$
\nabla
\cdot
\hat{\mathbf{u}}
\approx
0
$$

This measurement will not be used to train the model.

For boundary-contidions, we can calculate errors. Our CFD problem is:

- Top wall: $u = 1$, $v = 0$
- Bottom, left, right wall: $u = 0$, $v = 0$

Therefore for physical consistency, we'll keep track of:

- Continuity error
- Boundary-Condition error

18. For each neural-network experiment, we'll record as well:

- Training Time: How long it takes TensorFlow to train the model.
- Inference Time: After it's already trained, how long it takes the model to generate the complete $u$, $v$, $p$ solution for the $21 \times 21$ grid.

SUMMARY:

- Training: Normalized combined MSE
- Accuracy: MSE, MAE, & Relative Error
- Physical Consistency: Continuity & Boundary Error.
- Performance: Training Time & Inference Time.

### Experiments setup
19. Main rule: An experiment folder should contain enough information to understand, reproduce, and re-analyze that experiment without relying on another experiment. My thesis is based on comparing PINN agains a baseline data-driven neural network and producing indicators that help settle this.

```python
experiment_001/
├── experiment_config.json
├── environment.json
│
├── finite_difference/
│   ├── config.json
│   ├── solution.npz
│   └── performance.json
│
├── data_driven_nn/
│   ├── config.json
|   ├── model.keras
|   ├── history.csv
│   ├── predictions.npz
│   ├── metrics.json
│   └── performance.json
│
└── pinn/
    ├── config.json
    ├── model.keras
    ├── history.csv
    ├── predictions.npz
    ├── metrics.json
    └── performance.json
```

- `experiment_config.json`: defines the experiment. This is the shared benchmark.
- Each method keeps its own complete record. This means that when running the experiments we can ask if these models are solving the same experiment. Then analyzing the details that are comparable.
- We save first and calculate later. This means that is important to see how the neural networks work and produce the experiments result as a last step.
- Preserve computational evidence as well. This is provided by `environment.json` that tells us what hardware/software was available. And each model run will have `performance.json` that says what this specific method actaully used. Overview, `environment.json` is the environment where I run the experiment (specifications of the laptop) and `performance.json` will be what amount of resource used. 

20. Setting up schemas

`experiment_config.json`

Define the shared problem that every method in this experiment is solving. 

```json
{
  "experiment_id": "experiment_001",
  "problem": "lid_driven_cavity",
  "description": "Baseline comparison at Re=100",

  "domain": {
    "x_min": 0.0,
    "x_max": 1.0,
    "y_min": 0.0,
    "y_max": 1.0
  },

  "grid": {
    "nx": 21,
    "ny": 21
  },

  "physics": {
    "reynolds_number": 100,
    "rho": 1.0,
    "nu": 0.01
  },

  "boundary_conditions": {
    "lid_velocity": 1.0,
    "wall_velocity": 0.0
  },

  "problem_type": {
    "time_dependent": false
  },

  "evaluation": {
    "outputs": ["u", "v", "p"],
    "reference_method": "finite_difference"
  },

  "methods": [
    "finite_difference",
    "data_driven_nn",
    "pinn"
  ]
}
```

`environment.json`

Record the environment available when the experiment ran. 

```json
{
  "timestamp": "YYYY-MM-DDTHH:MM:SS",

  "software": {
    "python_version": "",
    "tensorflow_version": "",
    "numpy_version": ""
  },

  "system": {
    "operating_system": "",
    "cpu": "",
    "cpu_count": 0,
    "total_ram_bytes": 0
  },

  "gpu": {
    "available": false,
    "device": "",
    "total_vram_bytes": null
  }
}
```

`finite_difference/config.json`

Record how it was configured for this experiment

```json
{
  "numerical_method": "finite_difference",

  "time_step": 0.001,
  "max_steps": 6000,

  "pressure_solver": {
    "method": "jacobi",
    "iterations": 0
  },

  "convergence": {
    "enabled": true,
    "tolerance": 0.00001
  },

  "initial_conditions": {
    "u": 0.0,
    "v": 0.0,
    "p": 0.0
  }
}
```

`data_driven_nn/config.json`

Records how the data-driven nn was configured and trained for that experiment.

```json
{
  "method": "data_driven_nn",

  "model": {
    "hidden_layers": 0,
    "neurons_per_layer": 0,
    "activation": "",
    "output_size": 3
  },

  "training": {
    "epochs": 0,
    "batch_size": 0,
    "learning_rate": 0.0,
    "optimizer": "",
    "loss_function": "mse"
  },

  "data": {
    "input_variables": ["x", "y"],
    "target_variables": ["u", "v", "p"],
    "training_samples": 0
  },

  "reproducibility": {
    "random_seed": 0
  }
}
```

`pinn/config.json`

Record settings of the PINN for that experiment.

```json
{
  "method": "pinn",

  "model": {
    "hidden_layers": 0,
    "neurons_per_layer": 0,
    "activation": "",
    "output_size": 3
  },

  "training": {
    "epochs": 0,
    "learning_rate": 0.0,
    "optimizer": ""
  },

  "physics": {
    "equations": [
      "continuity",
      "x_momentum",
      "y_momentum"
    ],
    "collocation_points": 0
  },

  "loss": {
    "continuity_weight": 0.0,
    "x_momentum_weight": 0.0,
    "y_momentum_weight": 0.0,
    "boundary_weight": 0.0
  },

  "reproducibility": {
    "random_seed": 0
  }
}
```

`performance.json`

Record the computational resources that method actually used during the experiment.

```json
{
  "timing": {
    "total_runtime_seconds": 0.0,
    "training_time_seconds": null,
    "inference_time_seconds": null
  },

  "memory": {
    "peak_ram_bytes": 0,
    "peak_vram_bytes": null
  },

  "storage": {
    "model_size_bytes": null,
    "output_size_bytes": 0,
    "total_run_size_bytes": 0
  }
}
```

`metrics.json`

Records how well the method performed. 

```json
{
  "accuracy": {
    "mse_u": 0.0,
    "mse_v": 0.0,
    "mse_p": 0.0,

    "relative_l2_u": 0.0,
    "relative_l2_v": 0.0,
    "relative_l2_p": 0.0
  },

  "physical_consistency": {
    "continuity_residual": 0.0,
    "x_momentum_residual": 0.0,
    "y_momentum_residual": 0.0
  }
}
```

`predictions.npz` / `solution.npz`

Preserve the actual fields produced by each method so we can calculate new metrics later without rerunning the experiment.

`history.csv`

Preserve how training changed from epoch to epoch.

`experiment_results.csv`

Generated overview that pulls the most useful comparison information from all experiment folders.

Columns that will be used:
```text
experiment_id
problem
reynolds_number
grid_nx
grid_ny
time_dependent
method

mse_u
mse_v
mse_p
relative_l2_u
relative_l2_v
relative_l2_p

continuity_residual
x_momentum_residual
y_momentum_residual

total_runtime_seconds
training_time_seconds
inference_time_seconds
peak_ram_bytes
peak_vram_bytes
model_size_bytes
output_size_bytes
total_run_size_bytes
```

Graphs that will be produced:

| Category | Graph | Purpose |
|---|---|---|
| Accuracy | MSE by method for $u$, $v$, $p$ | Compare prediction accuracy |
| Accuracy | Relative $L_2$ by method for $u$, $v$, $p$ | Compare normalized error |
| Physics | Physics residuals by method | Compare continuity and momentum consistency |
| Compute | Runtime by method | Compare computational time |
| Compute | Peak RAM by method | Compare memory requirements |
| Compute | Peak VRAM by method | Compare GPU memory requirements |
| Compute | Storage by method | Compare model/run storage |
| Training | Loss vs epoch | Show training behavior |
| Fields | $u$, $v$, $p$ field plots | Visually compare solutions |
| Error | $u$, $v$, $p$ error fields | Show where predictions differ from the finite-difference reference |


### Cavity Benchmark:

21. Kamel (2020) was selected as the peer-reviewed journal article for checking the finite-difference solver at $Re = 100$. The paper uses the same one-sided lid-driven cavity boundary conditions as the current solver: the top wall moves with $u = 1$, $v = 0$, while the other three walls are stationary.

This also showed that the solver's previous convergence test only checked convergence over time. A separate grid-independence test will eventually be needed to check how spatial resolution affects the finite-difference solution.

22. The primary vortex-center location will be used to check the finite-difference solution at $Re = 100$. Kamel (2020) reports the primary vortex center at: $(x, y) = (0.6133, 0.7400)$. The finite-difference solver's vortex center will later be calculated from its velocity field and compared against this value. Since the current $21 \times 21$ grid is much coarser than Kamel's $151 \times 151$ grid, an exact coordinate match is not expected.

