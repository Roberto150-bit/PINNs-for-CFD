# PINNs-for-CFD

Senior thesis project exploring computational physics and machine learning, with a focus on comparing traditional numerical methods, data-driven neural networks, and physics-informed neural networks (PINNs).

The project studies two-dimensional incompressible Navier–Stokes flow using the lid-driven cavity problem as a controlled computational benchmark. Fluid is modeled within a square domain where the top wall moves horizontally while the remaining walls remain stationary, producing a characteristic circulating vortex flow.

The thesis compares three approaches to modeling this system: a traditional finite-difference numerical solver, a data-driven neural network trained on numerical reference data, and a physics-informed neural network that incorporates the governing Navier–Stokes equations and physical constraints into training. The goal is to evaluate how these approaches differ in their ability to reproduce the underlying flow field and to investigate the strengths and limitations of PINNs as an alternative or complement to conventional computational fluid dynamics methods.

## Repository Structure

- ```models/```: Implementations of the finite-difference solver, data-driven neural network, and PINN.
- ```notebooks/```: Explotary analysis, testing, visualization, and computational work.
- ```experiments/```: Structured experiments and experimental results
- ```notes/```: Weekly research notes, mathematical work, tehcnical decisions, observations, and progress.
- ```thesis/```: Thesis drafts and final written materials.

**NOTE:** This repository will evolve as the thesis progress.

