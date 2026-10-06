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