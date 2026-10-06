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

