# Week 2

## Goal

Implement the finite-difference baseline and begin the shared
architecture for the neural-network models.

## Mathematical Notes

The one-dimensional central-difference approximation is:

$$
\frac{\partial u}{\partial x}
\approx
\frac{u_{i+1} - u_{i-1}}{2\Delta x}
$$

For the second derivative:

$$
\frac{\partial^2 u}{\partial x^2}
\approx
\frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2}
$$

### Why this matters

I am using the finite-difference solution as the numerical baseline
against which the neural-network approaches will eventually be compared.

## Implementation Notes

The initial boundary-condition implementation is:

```python
u[:, 0] = 0
u[:, -1] = 0