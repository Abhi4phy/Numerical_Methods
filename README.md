# Numerical Methods for Physics

A comprehensive collection of **58 numerical methods** commonly used in computational physics and applied mathematics. Each method is implemented in a self-contained Python file with detailed theory, "Where to Start" guidance, working implementations, and demonstrations.

---

## 📖 Where to Start — Learning Roadmap

If you're a computational physics scholar, here is the recommended order. Each stage builds on the previous one.

### Stage 1: Foundations (Start Here)
| Order | File | Why |
|-------|------|-----|
| 1 | `05_Root_Finding/bisection_method.py` | Simplest algorithm — builds intuition |
| 2 | `05_Root_Finding/newton_raphson.py` | Introduces convergence rates |
| 3 | `01_Linear_Algebra/lu_decomposition.py` | Core of all linear solvers |
| 4 | `01_Linear_Algebra/eigenvalue_eigenvector.py` | Essential for quantum mechanics |
| 5 | `03_Numerical_Integration/trapezoidal_rule.py` | Simplest numerical integration |
| 6 | `03_Numerical_Integration/gaussian_quadrature.py` | Optimal integration — used everywhere |
| 7 | `09_Error_Analysis_Stability/truncation_roundoff_error.py` | Understand YOUR results' accuracy |

### Stage 2: Differential Equations (The Heart of Physics)
| Order | File | Why |
|-------|------|-----|
| 8 | `02_Differential_Equations/euler_method.py` | Simplest ODE solver — see its limitations |
| 9 | `02_Differential_Equations/runge_kutta_rk4.py` | The workhorse ODE solver |
| 10 | `02_Differential_Equations/adaptive_step_size.py` | Real-world ODE solving (RK45) |
| 11 | `02_Differential_Equations/finite_difference_method.py` | PDEs — the bread and butter |
| 12 | `09_Error_Analysis_Stability/stability_courant.py` | Critical: CFL condition for PDEs |
| 13 | `02_Differential_Equations/symplectic_integrators.py` | Energy-conserving — essential for physics |

### Stage 3: Spectral & Transform Methods
| Order | File | Why |
|-------|------|-----|
| 14 | `07_Numerical_Linear_Systems/fast_fourier_transform.py` | FFT — ubiquitous in physics |
| 15 | `02_Differential_Equations/spectral_methods.py` | Exponential convergence for smooth problems |
| 16 | `13_Signal_Processing/wavelets.py` | Time-frequency analysis |

### Stage 4: Large-Scale Problems
| Order | File | Why |
|-------|------|-----|
| 17 | `01_Linear_Algebra/conjugate_gradient.py` | Iterative solver for big systems |
| 18 | `01_Linear_Algebra/sparse_matrices.py` | Real physics = sparse matrices |
| 19 | `01_Linear_Algebra/krylov_methods.py` | GMRES, BiCGSTAB for non-symmetric systems |
| 20 | `01_Linear_Algebra/svd.py` | Data compression, pseudoinverse, PCA |
| 21 | `07_Numerical_Linear_Systems/multigrid_method.py` | Optimal O(N) solver |

### Stage 5: Statistical & Stochastic Methods
| Order | File | Why |
|-------|------|-----|
| 22 | `03_Numerical_Integration/monte_carlo_integration.py` | High-dimensional integration |
| 23 | `08_Stochastic_Statistical/monte_carlo_metropolis.py` | Sampling from complex distributions |
| 24 | `08_Stochastic_Statistical/mcmc.py` | Bayesian inference |
| 25 | `08_Stochastic_Statistical/parallel_tempering.py` | Escape local minima |
| 26 | `02_Differential_Equations/stochastic_de.py` | Brownian motion, Langevin dynamics |

### Stage 6: Advanced Physics Methods
| Order | File | Why |
|-------|------|-----|
| 27 | `10_Quantum_Methods/split_operator_schrodinger.py` | Quantum dynamics |
| 28 | `10_Quantum_Methods/dmrg.py` | Many-body quantum physics |
| 29 | `11_Fluid_Dynamics/finite_volume_method.py` | Conservation laws, shock capturing |
| 30 | `11_Fluid_Dynamics/lattice_boltzmann.py` | Mesoscale fluid dynamics |
| 31 | `12_Particle_Methods/nbody_methods.py` | Barnes-Hut, particle-mesh |
| 32 | `12_Particle_Methods/ewald_summation.py` | Long-range forces in periodic systems |

### Stage 7: Specialized & Advanced Topics
| Order | File | Why |
|-------|------|-----|
| 33 | `14_Automatic_Differentiation/automatic_differentiation.py` | Exact derivatives, ML connection |
| 34 | `15_Interface_Methods/level_set.py` | Moving boundaries, multiphase flows |
| 35 | `15_Interface_Methods/phase_field.py` | Diffuse interface modeling |
| 36 | `16_Advanced_Techniques/tensor_decomposition.py` | High-dimensional data & quantum |
| 37 | `16_Advanced_Techniques/pml_absorbing_bc.py` | Wave simulations — absorbing boundaries |
| 38 | `04_Interpolation_Approximation/pade_approximants.py` | Resum divergent series |

> **Tip:** Each file has a docstring with theory, equations, and a "Where to start" section. Read the docstring first, then run the file to see the demo output.

---

## Structure

### 1. Linear Algebra (`01_Linear_Algebra/`) — 10 files
- **LU Decomposition** – Factoring a matrix into Lower and Upper triangular matrices
- **QR Decomposition** – Factoring using orthogonal and upper triangular matrices
- **Cholesky Decomposition** – For symmetric positive-definite matrices
- **Eigenvalue/Eigenvector Problems** – Power iteration and QR algorithm
- **Jacobi Iterative Solver** – Iterative method for diagonally dominant systems
- **Gauss-Seidel Solver** – Improved iterative solver
- **Conjugate Gradient** – For large sparse symmetric positive-definite systems
- **Singular Value Decomposition (SVD)** ⭐ – Matrix factorization, pseudoinverse, PCA, low-rank approximation
- **Krylov Subspace Methods** ⭐ – GMRES, BiCGSTAB for non-symmetric systems
- **Sparse Matrix Methods** ⭐ – COO/CSR formats, sparse solvers, bandwidth reduction

### 2. Differential Equations (`02_Differential_Equations/`) — 9 files
- **Euler Method** – Simplest ODE solver
- **Runge-Kutta (RK4)** – Fourth-order accurate ODE solver
- **Adaptive Step-Size Methods** – RK45 with error control
- **Finite Difference Method (FDM)** – Discretizing PDEs on grids
- **Finite Element Method (FEM)** – Weak-form PDE solutions
- **Spectral Methods** – Fourier/Chebyshev basis for PDEs
- **Boundary Element Method (BEM)** – Surface-based PDE approach
- **Symplectic Integrators** ⭐ – Verlet, leapfrog, Yoshida-4 — energy-conserving for Hamiltonian systems
- **Stochastic Differential Equations** ⭐ – Euler-Maruyama, Milstein, Langevin dynamics

### 3. Numerical Integration (`03_Numerical_Integration/`) — 4 files
- **Trapezoidal Rule** – Linear approximation of integrals
- **Simpson's Rule** – Quadratic approximation of integrals
- **Gaussian Quadrature** – Optimal node placement for integration
- **Monte Carlo Integration** – Stochastic integration for high dimensions

### 4. Interpolation & Approximation (`04_Interpolation_Approximation/`) — 5 files
- **Lagrange Interpolation** – Polynomial through data points
- **Newton Interpolation** – Divided-difference form
- **Cubic Spline Interpolation** – Smooth piecewise polynomials
- **Least-Squares Fitting** – Best-fit curves to data
- **Padé Approximants** ⭐ – Rational function approximation, series resummation

### 5. Root-Finding (`05_Root_Finding/`) — 4 files
- **Bisection Method** – Guaranteed convergence by interval halving
- **Newton-Raphson Method** – Quadratic convergence using derivatives
- **Secant Method** – Derivative-free quasi-Newton approach
- **Fixed-Point Iteration** – Iterative function evaluation

### 6. Optimization (`06_Optimization/`) — 4 files
- **Gradient Descent** – First-order iterative optimization
- **Conjugate Gradient Optimization** – Efficient for quadratic objectives
- **Linear Programming** – Simplex method for LP problems
- **Variational Methods** – Euler-Lagrange equation approach

### 7. Numerical Linear Systems in Physics (`07_Numerical_Linear_Systems/`) — 3 files
- **Fast Fourier Transform (FFT)** – Efficient spectral decomposition
- **Green's Function Methods** – Solving inhomogeneous differential equations
- **Multigrid Methods** – Hierarchical solvers for large-scale PDEs

### 8. Stochastic & Statistical Methods (`08_Stochastic_Statistical/`) — 4 files
- **Monte Carlo (Metropolis-Hastings)** – MCMC sampling algorithm
- **Markov Chain Monte Carlo (MCMC)** – Bayesian inference with MCMC
- **Random Sampling** – RNG and statistical sampling techniques
- **Parallel Tempering** ⭐ – Replica exchange MCMC for multimodal distributions

### 9. Error Analysis & Stability (`09_Error_Analysis_Stability/`) — 3 files
- **Truncation & Round-off Error** – Understanding numerical precision
- **Stability Analysis (Courant Condition)** – CFL condition for PDEs
- **Convergence Criteria** – Measuring and ensuring convergence

### 10. Quantum Methods (`10_Quantum_Methods/`) ⭐ — 2 files
- **Split-Operator Schrödinger** – FFT-based time-dependent quantum dynamics, tunneling, imaginary time
- **DMRG** – Density Matrix Renormalization Group for 1D quantum many-body systems

### 11. Fluid Dynamics (`11_Fluid_Dynamics/`) ⭐ — 2 files
- **Finite Volume Method** – Conservative schemes for Euler equations, Burgers, shock capturing
- **Lattice Boltzmann Method** – D2Q9 BGK for incompressible flows, lid-driven cavity

### 12. Particle Methods (`12_Particle_Methods/`) ⭐ — 2 files
- **N-Body Methods** – Direct summation, Barnes-Hut tree, particle-mesh (PM) method
- **Ewald Summation** – Long-range periodic electrostatics, Madelung constants

### 13. Signal Processing (`13_Signal_Processing/`) ⭐ — 1 file
- **Wavelet Transform** – DWT/CWT, multi-resolution analysis, denoising, compression

### 14. Automatic Differentiation (`14_Automatic_Differentiation/`) ⭐ — 1 file
- **Automatic Differentiation** – Forward mode (dual numbers), reverse mode (backpropagation), Jacobians

### 15. Interface Methods (`15_Interface_Methods/`) ⭐ — 2 files
- **Level Set Methods** – Implicit interface tracking, signed distance functions, CSG operations
- **Phase-Field Methods** – Allen-Cahn, Cahn-Hilliard, spinodal decomposition

### 16. Advanced Techniques (`16_Advanced_Techniques/`) ⭐ — 2 files
- **Tensor Decomposition** – CP, Tucker (HOSVD), Tensor Train for high-dimensional data
- **Perfectly Matched Layer (PML)** – Absorbing boundary conditions for wave equations

> ⭐ = New advanced methods

---

## Requirements

```
numpy
matplotlib
scipy (for comparison/validation only)
```

## Usage

Each file is standalone. Run any file directly:
```bash
python Numerical_Methods/01_Linear_Algebra/lu_decomposition.py
```

Every file includes:
- **Module docstring** with theory, equations, and key references
- **"Where to start" section** with step-by-step guidance
- **Implementation** with clear function signatures and docstrings
- **`__main__` demo** with numerical output and optional plots
