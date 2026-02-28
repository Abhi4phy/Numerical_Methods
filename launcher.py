"""
Numerical Methods Launcher
===========================
Interactive launcher with both Terminal (CLI) and GUI modes.

Usage:
    python launcher.py          → auto-detect (GUI if available, else CLI)
    python launcher.py --cli    → force terminal mode
    python launcher.py --gui    → force GUI mode

Architecture:
    This file is a backward-compatible entry point.
    The application is organized as the ``launcher_app`` package:

        launcher_app/
        ├── __init__.py     Package metadata
        ├── __main__.py     python -m launcher_app
        ├── config.py       Paths, colours, fonts, constants
        ├── catalog.py      CATEGORIES, LEARNING_PATH
        ├── equations.py    EQUATIONS database, is_math_line()
        ├── utils.py        get_file_docstring(), run_file()
        ├── cli.py          Terminal (CLI) launcher
        ├── gui.py          Tkinter GUI launcher
        └── main.py         Mode detection & dispatch

    All public symbols are re-exported here for backward
    compatibility:  ``from launcher import CATEGORIES, EQUATIONS``
"""

import os
import sys
import subprocess
import textwrap

# ============================================================
# Method Catalog
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CATEGORIES = [
    {
        "id": "01",
        "name": "Linear Algebra",
        "folder": "01_Linear_Algebra",
        "icon": "🔢",
        "files": [
            ("lu_decomposition.py",       "LU Decomposition",       "Factor A = LU for solving linear systems"),
            ("qr_decomposition.py",       "QR Decomposition",       "Orthogonal factorization A = QR"),
            ("cholesky_decomposition.py",  "Cholesky Decomposition", "For symmetric positive-definite matrices"),
            ("eigenvalue_eigenvector.py",  "Eigenvalues & Eigenvectors", "Power iteration and QR algorithm"),
            ("jacobi_iterative.py",       "Jacobi Iterative Solver", "Iterative solver for diagonally dominant systems"),
            ("gauss_seidel.py",           "Gauss-Seidel Solver",    "Improved iterative linear solver"),
            ("conjugate_gradient.py",     "Conjugate Gradient",     "Krylov solver for SPD systems"),
            ("svd.py",                    "SVD",                    "Singular Value Decomposition, PCA, pseudoinverse"),
            ("krylov_methods.py",         "Krylov Methods",         "GMRES, BiCGSTAB for non-symmetric systems"),
            ("sparse_matrices.py",        "Sparse Matrices",        "COO/CSR formats, sparse solvers"),
        ]
    },
    {
        "id": "02",
        "name": "Differential Equations",
        "folder": "02_Differential_Equations",
        "icon": "📈",
        "files": [
            ("euler_method.py",           "Euler Method",           "Simplest ODE solver — start here!"),
            ("runge_kutta_rk4.py",        "Runge-Kutta RK4",       "The workhorse 4th-order ODE solver"),
            ("adaptive_step_size.py",     "Adaptive Step Size",     "RK45 with automatic error control"),
            ("finite_difference_method.py","Finite Difference (FDM)","Discretize PDEs on grids"),
            ("finite_element_method.py",  "Finite Element (FEM)",   "Weak-form PDE solutions"),
            ("spectral_methods.py",       "Spectral Methods",       "Fourier/Chebyshev for exponential convergence"),
            ("boundary_element_method.py","Boundary Element (BEM)", "Surface-based PDE approach"),
            ("symplectic_integrators.py", "Symplectic Integrators", "Energy-conserving for Hamiltonian systems"),
            ("stochastic_de.py",          "Stochastic DEs",         "Euler-Maruyama, Milstein, Langevin"),
        ]
    },
    {
        "id": "03",
        "name": "Numerical Integration",
        "folder": "03_Numerical_Integration",
        "icon": "∫",
        "files": [
            ("trapezoidal_rule.py",       "Trapezoidal Rule",       "Linear approximation of integrals"),
            ("simpsons_rule.py",          "Simpson's Rule",         "Quadratic approximation of integrals"),
            ("gaussian_quadrature.py",    "Gaussian Quadrature",    "Optimal node placement — high accuracy"),
            ("monte_carlo_integration.py","Monte Carlo Integration","Stochastic integration for high dimensions"),
        ]
    },
    {
        "id": "04",
        "name": "Interpolation & Approximation",
        "folder": "04_Interpolation_Approximation",
        "icon": "📐",
        "files": [
            ("lagrange_interpolation.py", "Lagrange Interpolation", "Polynomial through data points"),
            ("newton_interpolation.py",   "Newton Interpolation",   "Divided-difference form"),
            ("cubic_spline.py",           "Cubic Spline",           "Smooth piecewise polynomials"),
            ("least_squares_fitting.py",  "Least-Squares Fitting",  "Best-fit curves to data"),
            ("pade_approximants.py",      "Padé Approximants",      "Rational function approximation"),
        ]
    },
    {
        "id": "05",
        "name": "Root-Finding",
        "folder": "05_Root_Finding",
        "icon": "🎯",
        "files": [
            ("bisection_method.py",       "Bisection Method",       "Guaranteed convergence by halving"),
            ("newton_raphson.py",         "Newton-Raphson",         "Quadratic convergence with derivatives"),
            ("secant_method.py",          "Secant Method",          "Derivative-free quasi-Newton"),
            ("fixed_point_iteration.py",  "Fixed-Point Iteration",  "Iterative x = g(x) approach"),
        ]
    },
    {
        "id": "06",
        "name": "Optimization",
        "folder": "06_Optimization",
        "icon": "⚡",
        "files": [
            ("gradient_descent.py",                "Gradient Descent",       "First-order iterative optimization"),
            ("conjugate_gradient_optimization.py",  "CG Optimization",       "Efficient for quadratic objectives"),
            ("linear_programming.py",              "Linear Programming",     "Simplex method for LP"),
            ("variational_methods.py",             "Variational Methods",    "Euler-Lagrange equation approach"),
        ]
    },
    {
        "id": "07",
        "name": "Numerical Linear Systems",
        "folder": "07_Numerical_Linear_Systems",
        "icon": "🌊",
        "files": [
            ("fast_fourier_transform.py", "Fast Fourier Transform",  "FFT — ubiquitous in physics"),
            ("greens_function.py",        "Green's Functions",       "Solving inhomogeneous DEs"),
            ("multigrid_method.py",       "Multigrid Methods",       "Optimal O(N) PDE solvers"),
        ]
    },
    {
        "id": "08",
        "name": "Stochastic & Statistical",
        "folder": "08_Stochastic_Statistical",
        "icon": "🎲",
        "files": [
            ("monte_carlo_metropolis.py", "Metropolis-Hastings",    "MCMC sampling algorithm"),
            ("mcmc.py",                   "MCMC",                   "Bayesian inference with MCMC"),
            ("random_sampling.py",        "Random Sampling",        "RNG and sampling techniques"),
            ("parallel_tempering.py",     "Parallel Tempering",     "Replica exchange for multimodal distributions"),
        ]
    },
    {
        "id": "09",
        "name": "Error Analysis & Stability",
        "folder": "09_Error_Analysis_Stability",
        "icon": "📊",
        "files": [
            ("truncation_roundoff_error.py","Truncation & Round-off", "Understanding numerical precision"),
            ("stability_courant.py",       "Stability (CFL)",        "Courant condition for PDEs"),
            ("convergence_criteria.py",    "Convergence Criteria",   "Measuring convergence"),
        ]
    },
    {
        "id": "10",
        "name": "Quantum Methods",
        "folder": "10_Quantum_Methods",
        "icon": "⚛️",
        "files": [
            ("split_operator_schrodinger.py", "Split-Operator Schrödinger", "FFT-based quantum dynamics"),
            ("dmrg.py",                       "DMRG",                       "Density Matrix Renormalization Group"),
        ]
    },
    {
        "id": "11",
        "name": "Fluid Dynamics",
        "folder": "11_Fluid_Dynamics",
        "icon": "💧",
        "files": [
            ("finite_volume_method.py",  "Finite Volume Method",    "Conservative PDE discretization"),
            ("lattice_boltzmann.py",     "Lattice Boltzmann",       "D2Q9 BGK for fluid simulations"),
        ]
    },
    {
        "id": "12",
        "name": "Particle Methods",
        "folder": "12_Particle_Methods",
        "icon": "🔵",
        "files": [
            ("nbody_methods.py",         "N-Body Methods",          "Barnes-Hut tree, particle-mesh"),
            ("ewald_summation.py",       "Ewald Summation",         "Long-range periodic electrostatics"),
        ]
    },
    {
        "id": "13",
        "name": "Signal Processing",
        "folder": "13_Signal_Processing",
        "icon": "📡",
        "files": [
            ("wavelets.py",              "Wavelet Transform",       "DWT/CWT, denoising, compression"),
        ]
    },
    {
        "id": "14",
        "name": "Automatic Differentiation",
        "folder": "14_Automatic_Differentiation",
        "icon": "🔄",
        "files": [
            ("automatic_differentiation.py", "Automatic Differentiation", "Forward & reverse mode AD"),
        ]
    },
    {
        "id": "15",
        "name": "Interface Methods",
        "folder": "15_Interface_Methods",
        "icon": "🔲",
        "files": [
            ("level_set.py",             "Level Set Methods",       "Implicit interface tracking, CSG"),
            ("phase_field.py",           "Phase-Field Methods",     "Allen-Cahn, Cahn-Hilliard"),
        ]
    },
    {
        "id": "16",
        "name": "Advanced Techniques",
        "folder": "16_Advanced_Techniques",
        "icon": "🚀",
        "files": [
            ("tensor_decomposition.py",  "Tensor Decomposition",    "CP, Tucker, Tensor Train"),
            ("pml_absorbing_bc.py",      "PML Absorbing BC",        "Perfectly Matched Layer for waves"),
        ]
    },
]

# Recommended learning path (file paths in order)
LEARNING_PATH = [
    ("Stage 1: Foundations", [
        ("05_Root_Finding", "bisection_method.py",              "Simplest algorithm — builds intuition"),
        ("05_Root_Finding", "newton_raphson.py",                "Introduces convergence rates"),
        ("01_Linear_Algebra", "lu_decomposition.py",            "Core of all linear solvers"),
        ("01_Linear_Algebra", "eigenvalue_eigenvector.py",      "Essential for quantum mechanics"),
        ("03_Numerical_Integration", "trapezoidal_rule.py",     "Simplest numerical integration"),
        ("03_Numerical_Integration", "gaussian_quadrature.py",  "Optimal integration — used everywhere"),
        ("09_Error_Analysis_Stability", "truncation_roundoff_error.py", "Understand YOUR results' accuracy"),
    ]),
    ("Stage 2: Differential Equations", [
        ("02_Differential_Equations", "euler_method.py",             "Simplest ODE solver"),
        ("02_Differential_Equations", "runge_kutta_rk4.py",          "The workhorse ODE solver"),
        ("02_Differential_Equations", "adaptive_step_size.py",       "Real-world ODE solving (RK45)"),
        ("02_Differential_Equations", "finite_difference_method.py", "PDEs — the bread and butter"),
        ("09_Error_Analysis_Stability", "stability_courant.py",      "CFL condition for PDEs"),
        ("02_Differential_Equations", "symplectic_integrators.py",   "Energy-conserving for Hamiltonian systems"),
    ]),
    ("Stage 3: Spectral & Transform Methods", [
        ("07_Numerical_Linear_Systems", "fast_fourier_transform.py", "FFT — ubiquitous in physics"),
        ("02_Differential_Equations", "spectral_methods.py",         "Exponential convergence"),
        ("13_Signal_Processing", "wavelets.py",                      "Time-frequency analysis"),
    ]),
    ("Stage 4: Large-Scale Problems", [
        ("01_Linear_Algebra", "conjugate_gradient.py",  "Iterative solver for big systems"),
        ("01_Linear_Algebra", "sparse_matrices.py",     "Real physics = sparse matrices"),
        ("01_Linear_Algebra", "krylov_methods.py",      "GMRES, BiCGSTAB"),
        ("01_Linear_Algebra", "svd.py",                 "Data compression, PCA"),
        ("07_Numerical_Linear_Systems", "multigrid_method.py", "Optimal O(N) solver"),
    ]),
    ("Stage 5: Statistical & Stochastic", [
        ("03_Numerical_Integration", "monte_carlo_integration.py",  "High-dimensional integration"),
        ("08_Stochastic_Statistical", "monte_carlo_metropolis.py",  "Sampling complex distributions"),
        ("08_Stochastic_Statistical", "mcmc.py",                    "Bayesian inference"),
        ("08_Stochastic_Statistical", "parallel_tempering.py",      "Escape local minima"),
        ("02_Differential_Equations", "stochastic_de.py",           "Brownian motion, Langevin"),
    ]),
    ("Stage 6: Advanced Physics", [
        ("10_Quantum_Methods", "split_operator_schrodinger.py", "Quantum dynamics"),
        ("10_Quantum_Methods", "dmrg.py",                       "Many-body quantum"),
        ("11_Fluid_Dynamics", "finite_volume_method.py",         "Shock capturing"),
        ("11_Fluid_Dynamics", "lattice_boltzmann.py",            "Mesoscale fluids"),
        ("12_Particle_Methods", "nbody_methods.py",              "Barnes-Hut, PM"),
        ("12_Particle_Methods", "ewald_summation.py",            "Periodic electrostatics"),
    ]),
    ("Stage 7: Specialized Topics", [
        ("14_Automatic_Differentiation", "automatic_differentiation.py", "Exact derivatives"),
        ("15_Interface_Methods", "level_set.py",                  "Moving boundaries"),
        ("15_Interface_Methods", "phase_field.py",                "Diffuse interfaces"),
        ("16_Advanced_Techniques", "tensor_decomposition.py",     "High-dimensional data"),
        ("16_Advanced_Techniques", "pml_absorbing_bc.py",         "Absorbing boundaries"),
        ("04_Interpolation_Approximation", "pade_approximants.py","Resum divergent series"),
    ]),
]


def get_file_docstring(filepath):
    """Extract the module-level docstring from a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()  # Read full file for complete docstring
        # Find triple-quoted docstring
        for quote in ['"""', "'''"]:
            idx = content.find(quote)
            if idx != -1:
                end = content.find(quote, idx + 3)
                if end != -1:
                    return content[idx + 3:end].strip()
        return "No description available."
    except Exception:
        return "Could not read file."


def run_file(filepath):
    """Run a Python file as a subprocess."""
    print(f"\n{'='*60}")
    print(f"  Running: {os.path.basename(filepath)}")
    print(f"{'='*60}\n")
    try:
        result = subprocess.run(
            [sys.executable, filepath],
            cwd=os.path.dirname(filepath),
            timeout=120
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print("\n⚠️  Script timed out after 120 seconds.")
        return -1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return -1


# ============================================================
# Supplementary LaTeX Equations (rendered in GUI)
# ============================================================

EQUATIONS = {
    "lu_decomposition.py": [
        ("Matrix Factorization", r"A = LU"),
        ("Forward Substitution", r"y_i = b_i - \sum_{j=1}^{i-1} \ell_{ij}\, y_j"),
        ("Back Substitution", r"x_i = \frac{1}{u_{ii}} \left( y_i - \sum_{j=i+1}^{n} u_{ij}\, x_j \right)"),
        ("Computational Cost", r"\mathcal{O}\!\left(\tfrac{2}{3}n^3\right)"),
    ],
    "eigenvalue_eigenvector.py": [
        ("Eigenvalue Problem", r"A\mathbf{v} = \lambda \mathbf{v}"),
        ("Characteristic Polynomial", r"\det(A - \lambda I) = 0"),
        ("Power Iteration", r"\mathbf{v}^{(k+1)} = \frac{A\mathbf{v}^{(k)}}{\|A\mathbf{v}^{(k)}\|}"),
        ("Rayleigh Quotient", r"\lambda \approx \frac{\mathbf{v}^T A \mathbf{v}}{\mathbf{v}^T \mathbf{v}}"),
    ],
    "svd.py": [
        ("SVD Factorization", r"A = U \Sigma V^T"),
        ("Low-Rank Approximation", r"A_k = \sum_{i=1}^{k} \sigma_i\, \mathbf{u}_i \mathbf{v}_i^T"),
        ("Pseudoinverse", r"A^+ = V \Sigma^+ U^T"),
        ("Eckart–Young Theorem", r"\min_{\mathrm{rank}(B)=k} \|A-B\|_F = \sqrt{\sum_{i>k}\sigma_i^2}"),
    ],
    "conjugate_gradient.py": [
        ("CG Update", r"\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k"),
        ("Step Size", r"\alpha_k = \frac{\mathbf{r}_k^T \mathbf{r}_k}{\mathbf{p}_k^T A\, \mathbf{p}_k}"),
        ("Convergence", r"\|e_k\|_A \leq 2 \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^{\!k} \|e_0\|_A"),
    ],
    "euler_method.py": [
        ("Euler Step", r"y_{n+1} = y_n + h\, f(t_n,\, y_n)"),
        ("Local Truncation Error", r"\tau_n = \frac{h^2}{2}\, y''(\xi_n) = \mathcal{O}(h^2)"),
        ("Global Error", r"|e_n| \leq \frac{hM}{2L}\bigl(e^{L(t_n-t_0)}-1\bigr) = \mathcal{O}(h)"),
    ],
    "runge_kutta_rk4.py": [
        ("Stage 1", r"k_1 = f(t_n,\; y_n)"),
        ("Stage 2", r"k_2 = f\!\left(t_n+\tfrac{h}{2},\; y_n+\tfrac{h}{2}k_1\right)"),
        ("Stage 3", r"k_3 = f\!\left(t_n+\tfrac{h}{2},\; y_n+\tfrac{h}{2}k_2\right)"),
        ("Stage 4", r"k_4 = f(t_n+h,\; y_n+h\,k_3)"),
        ("RK4 Update", r"y_{n+1} = y_n + \frac{h}{6}\bigl(k_1 + 2k_2 + 2k_3 + k_4\bigr)"),
        ("Error Order", r"\tau = \mathcal{O}(h^5),\quad E_{\text{global}} = \mathcal{O}(h^4)"),
    ],
    "finite_difference_method.py": [
        ("Forward Difference", r"f'(x) \approx \frac{f(x+h) - f(x)}{h} + \mathcal{O}(h)"),
        ("Central Difference", r"f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} + \mathcal{O}(h^2)"),
        ("Second Derivative", r"f''(x) \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2}"),
        ("2D Laplacian", r"\nabla^2 u \approx \frac{u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{i,j}}{h^2}"),
    ],
    "spectral_methods.py": [
        ("Fourier Expansion", r"u(x) = \sum_{k=-N/2}^{N/2} \hat{u}_k\, e^{ikx}"),
        ("Spectral Differentiation", r"\widehat{u'}_k = ik\,\hat{u}_k"),
        ("Exponential Convergence", r"\|u - u_N\| \leq C\, e^{-\alpha N}\;\text{(smooth }u\text{)}"),
    ],
    "symplectic_integrators.py": [
        ("Hamiltonian", r"H(q,p) = \frac{p^2}{2m} + V(q)"),
        ("Velocity Verlet (position)", r"q_{n+1} = q_n + h\,p_n/m + \frac{h^2}{2m}F(q_n)"),
        ("Velocity Verlet (momentum)", r"p_{n+1} = p_n + \frac{h}{2}\bigl[F(q_n)+F(q_{n+1})\bigr]"),
        ("Symplecticity", r"\det\frac{\partial(q_{n+1},p_{n+1})}{\partial(q_n,p_n)} = 1"),
    ],
    "gaussian_quadrature.py": [
        ("Quadrature Rule", r"\int_{-1}^{1} f(x)\,dx \approx \sum_{i=1}^{n} w_i\, f(x_i)"),
        ("Exactness", r"\text{n-point Gauss: exact for polynomials of degree }\leq 2n-1"),
    ],
    "monte_carlo_integration.py": [
        ("MC Estimator", r"I \approx \frac{V}{N}\sum_{i=1}^{N} f(\mathbf{x}_i)"),
        ("Error Rate", r"\sigma_I = \frac{\sigma_f}{\sqrt{N}}\;\text{(dimension-independent)}"),
        ("Importance Sampling", r"I \approx \frac{1}{N}\sum_{i=1}^{N} \frac{f(x_i)}{p(x_i)},\quad x_i\sim p"),
    ],
    "fast_fourier_transform.py": [
        ("DFT", r"X_k = \sum_{n=0}^{N-1} x_n\, e^{-2\pi i\, kn/N}"),
        ("Inverse DFT", r"x_n = \frac{1}{N}\sum_{k=0}^{N-1} X_k\, e^{2\pi i\, kn/N}"),
        ("FFT Complexity", r"\mathcal{O}(N \log N)\;\text{vs}\;\mathcal{O}(N^2)\text{ for DFT}"),
        ("Parseval's Theorem", r"\sum_{n}|x_n|^2 = \frac{1}{N}\sum_{k}|X_k|^2"),
    ],
    "newton_raphson.py": [
        ("Newton Step", r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}"),
        ("Quadratic Convergence", r"|e_{n+1}| \leq C\,|e_n|^2"),
        ("Multidimensional", r"\mathbf{x}_{n+1} = \mathbf{x}_n - J^{-1}(\mathbf{x}_n)\,\mathbf{F}(\mathbf{x}_n)"),
    ],
    "gradient_descent.py": [
        ("Update Rule", r"\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha\,\nabla f(\mathbf{x}_k)"),
        ("Convergence (convex)", r"f(\mathbf{x}_k) - f^* \leq \frac{\|\mathbf{x}_0-\mathbf{x}^*\|^2}{2\alpha k}"),
    ],
    "monte_carlo_metropolis.py": [
        ("Acceptance Ratio", r"\alpha = \min\!\left(1,\; \frac{P(x')}{P(x)}\right)"),
        ("Boltzmann Weight", r"P(E) \propto e^{-E/k_BT}"),
        ("Detailed Balance", r"P(x)\,T(x{\to}x') = P(x')\,T(x'{\to}x)"),
    ],
    "split_operator_schrodinger.py": [
        ("Schrödinger Equation", r"i\hbar\frac{\partial\psi}{\partial t} = \left[-\frac{\hbar^2}{2m}\nabla^2 + V\right]\psi"),
        ("Split Operator", r"e^{-i\hat{H}\Delta t/\hbar} \approx e^{-i\hat{V}\Delta t/2\hbar}\, e^{-i\hat{T}\Delta t/\hbar}\, e^{-i\hat{V}\Delta t/2\hbar}"),
    ],
    "automatic_differentiation.py": [
        ("Dual Numbers", r"f(a+\varepsilon b) = f(a) + \varepsilon\, f'(a)\,b,\;\; \varepsilon^2=0"),
        ("Chain Rule (forward)", r"\dot{y} = \frac{\partial f}{\partial x}\,\dot{x}"),
        ("Chain Rule (reverse)", r"\bar{x} = \frac{\partial f}{\partial x}^{\!T}\bar{y}"),
    ],
    "wavelets.py": [
        ("CWT", r"W(a,b) = \frac{1}{\sqrt{a}}\int f(t)\,\psi^*\!\left(\frac{t-b}{a}\right)dt"),
        ("DWT Filter Bank", r"cA[n] = \sum_k h[k]\, x[2n+k],\;\; cD[n] = \sum_k g[k]\, x[2n+k]"),
    ],
    "phase_field.py": [
        ("Ginzburg-Landau Energy", r"F[\phi] = \int\!\left[\frac{(\phi^2-1)^2}{4} + \frac{\varepsilon^2}{2}|\nabla\phi|^2\right]dV"),
        ("Allen-Cahn", r"\frac{\partial\phi}{\partial t} = M\left[\varepsilon^2\nabla^2\phi - \phi^3 + \phi\right]"),
        ("Cahn-Hilliard", r"\frac{\partial\phi}{\partial t} = \nabla\cdot\!\left[M\,\nabla\!\left(\phi^3-\phi-\varepsilon^2\nabla^2\phi\right)\right]"),
    ],
    "level_set.py": [
        ("Level Set Equation", r"\frac{\partial\phi}{\partial t} + \mathbf{v}\cdot\nabla\phi = 0"),
        ("Normal Vector", r"\hat{n} = \frac{\nabla\phi}{|\nabla\phi|}"),
        ("Curvature", r"\kappa = \nabla\cdot\!\left(\frac{\nabla\phi}{|\nabla\phi|}\right)"),
        ("Eikonal (SDF)", r"|\nabla\phi| = 1"),
    ],
    "pml_absorbing_bc.py": [
        ("Damped Wave Equation", r"\frac{\partial^2 u}{\partial t^2} + \sigma\frac{\partial u}{\partial t} = c^2\frac{\partial^2 u}{\partial x^2}"),
        ("Damping Profile", r"\sigma(x) = \sigma_{\max}\!\left(\frac{d}{L_{\text{PML}}}\right)^{\!p}"),
        ("Optimal \u03c3_max", r"\sigma_{\max} = -\frac{(p+1)\,c\,\ln R}{2\,L_{\text{PML}}}"),
    ],
    "tensor_decomposition.py": [
        ("CP Decomposition", r"\mathcal{T} \approx \sum_{r=1}^{R} \lambda_r\, \mathbf{a}_r \otimes \mathbf{b}_r \otimes \mathbf{c}_r"),
        ("Tucker / HOSVD", r"\mathcal{T} \approx \mathcal{G} \times_1 U_1 \times_2 U_2 \times_3 U_3"),
        ("Tensor Train", r"\mathcal{T}(i_1,\ldots,i_d) = G_1[i_1]\, G_2[i_2]\cdots G_d[i_d]"),
    ],
    "finite_volume_method.py": [
        ("Conservation Law", r"\frac{\partial u}{\partial t} + \frac{\partial F(u)}{\partial x} = 0"),
        ("FVM Update", r"u_i^{n+1} = u_i^n - \frac{\Delta t}{\Delta x}\left[\hat{F}_{i+1/2} - \hat{F}_{i-1/2}\right]"),
        ("CFL Condition", r"\mathrm{CFL} = \frac{|a|\,\Delta t}{\Delta x} \leq 1"),
    ],
    "lattice_boltzmann.py": [
        ("Lattice Boltzmann Eq.", r"f_i(\mathbf{x}+\mathbf{c}_i\Delta t,\, t+\Delta t) = f_i + \Omega_i"),
        ("BGK Collision", r"\Omega_i = -\frac{f_i - f_i^{\text{eq}}}{\tau}"),
    ],
    "ewald_summation.py": [
        ("Ewald Split", r"E = E_{\text{real}} + E_{\text{recip}} + E_{\text{self}}"),
        ("Real Space", r"E_{\text{real}} = \frac{1}{2}\sum_{i\neq j} q_i q_j\, \frac{\mathrm{erfc}(\alpha r_{ij})}{r_{ij}}"),
    ],
    "stochastic_de.py": [
        ("SDE", r"dX_t = a(X_t)\,dt + b(X_t)\,dW_t"),
        ("Euler-Maruyama", r"X_{n+1} = X_n + a\,\Delta t + b\,\Delta W_n"),
        ("Itô's Lemma", r"df = \left(\frac{\partial f}{\partial t} + a\frac{\partial f}{\partial x} + \frac{b^2}{2}\frac{\partial^2 f}{\partial x^2}\right)dt + b\frac{\partial f}{\partial x}dW"),
    ],
}


def is_math_line(line):
    """Detect if a line likely contains a mathematical equation."""
    s = line.strip()
    if not s or len(s) < 3:
        return False
    # Skip obvious prose
    skip = ('**', '#', 'def ', 'class ', 'import ', 'from ', 'Note', 'See ',
            'Use ', 'Run ', 'The ', 'This ', 'For ', 'In ', 'It ', 'A ',
            'An ', 'Each ', 'Start', 'Prerequisite', 'Best ', 'Cost:',
            'Model', 'Where', 'Need', 'Can ', 'Only', 'Common', 'More',
            'Require', 'That', 'We ', 'You', 'Both', 'Prefer', 'Instead',
            'Useful', 'Peak', 'What', 'Instead', 'Here', 'Equivalent',
            'Surround', 'Result', 'When', 'Save', 'Works', 'Return')
    if s.startswith(skip):
        return False
    # Unicode math symbols strongly indicate equations
    MATH = set('\u2202\u2207\u222b\u2211\u220f\u221a\u221e\u2248\u2260\u2264\u2265\u00b1\u00d7\u00b7\u00f7\u2208\u2209\u2282\u2283\u222a\u2229\u2200\u2203\u03c6\u03c8\u03c9\u03a9\u03c0\u03b8\u03ba\u03b1\u03b2\u03b3\u0394\u03a3\u03a0\u00b2\u00b3\u2074\u207b\u00b9\u210f\u03b5\u03b4\u03bb\u03bc\u03c3\u03c4')
    n_math = sum(1 for c in s if c in MATH)
    if n_math >= 2:
        return True
    if n_math == 1 and ('=' in s or '\u2192' in s):
        return True
    return False


# ============================================================
# Terminal (CLI) Launcher
# ============================================================

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def cli_header():
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║        NUMERICAL METHODS FOR PHYSICS — LAUNCHER          ║")
    print("║        57 methods across 16 categories                   ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()


def cli_main_menu():
    """Show main menu and return choice."""
    cli_header()
    print("  Choose a mode:\n")
    print("    [B]  Browse by Category")
    print("    [L]  Learning Path (recommended order)")
    print("    [S]  Search methods")
    print("    [R]  Run a specific file")
    print("    [Q]  Quit")
    print()
    return input("  → ").strip().lower()


def cli_browse():
    """Browse methods by category."""
    while True:
        clear_screen()
        cli_header()
        print("  ── Categories ──\n")
        for i, cat in enumerate(CATEGORIES):
            n = len(cat["files"])
            print(f"    {i+1:2d}. {cat['icon']}  {cat['name']:30s} ({n} files)")
        print(f"\n    {'':2s}  [B] Back to main menu")
        print()
        
        choice = input("  Select category (1-16) → ").strip().lower()
        if choice == 'b' or choice == '':
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(CATEGORIES):
                cli_category(CATEGORIES[idx])
        except ValueError:
            pass


def cli_category(cat):
    """Show files in a category."""
    while True:
        clear_screen()
        cli_header()
        print(f"  ── {cat['icon']}  {cat['name']} ──\n")
        
        for i, (fname, title, desc) in enumerate(cat["files"]):
            filepath = os.path.join(BASE_DIR, cat["folder"], fname)
            exists = "✓" if os.path.exists(filepath) else "✗"
            print(f"    {i+1:2d}. [{exists}] {title:35s} — {desc}")
        
        print(f"\n    {'':2s}  [B] Back to categories")
        print()
        
        choice = input("  Select file (number) or [V]iew/[R]un → ").strip().lower()
        if choice == 'b' or choice == '':
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(cat["files"]):
                fname, title, desc = cat["files"][idx]
                filepath = os.path.join(BASE_DIR, cat["folder"], fname)
                cli_file_action(filepath, title, desc)
        except ValueError:
            pass


def cli_file_action(filepath, title, desc):
    """Show file actions."""
    while True:
        clear_screen()
        cli_header()
        print(f"  ── {title} ──")
        print(f"  {desc}\n")
        print(f"  File: {os.path.relpath(filepath, BASE_DIR)}")
        
        if os.path.exists(filepath):
            print(f"  Status: ✓ Available")
        else:
            print(f"  Status: ✗ File not found")
            input("\n  Press Enter to go back...")
            return
        
        print(f"\n    [V]  View description (docstring)")
        print(f"    [R]  Run the demo")
        print(f"    [O]  Open in editor")
        print(f"    [B]  Back")
        print()
        
        choice = input("  → ").strip().lower()
        
        if choice == 'b' or choice == '':
            return
        elif choice == 'v':
            clear_screen()
            cli_header()
            print(f"  ── {title} ── Description ──\n")
            docstring = get_file_docstring(filepath)
            # Word-wrap nicely
            for line in docstring.split('\n'):
                if len(line) > 76:
                    for wrapped in textwrap.wrap(line, 76):
                        print(f"  {wrapped}")
                else:
                    print(f"  {line}")
            print()
            input("  Press Enter to continue...")
        elif choice == 'r':
            run_file(filepath)
            print()
            input("  Press Enter to continue...")
        elif choice == 'o':
            try:
                if os.name == 'nt':
                    os.startfile(filepath)
                else:
                    subprocess.Popen(['xdg-open', filepath])
            except Exception:
                print(f"  Could not open file. Path: {filepath}")
                input("  Press Enter to continue...")


def cli_learning_path():
    """Show the recommended learning path."""
    while True:
        clear_screen()
        cli_header()
        print("  ── 📚 Recommended Learning Path ──\n")
        
        file_num = 0
        stages = []
        for stage_name, files in LEARNING_PATH:
            stages.append((stage_name, file_num, len(files)))
            file_num += len(files)
        
        for i, (stage_name, start, count) in enumerate(stages):
            print(f"    {i+1}. {stage_name:40s} ({count} methods)")
        
        print(f"\n    [B] Back")
        print()
        
        choice = input("  Select stage (1-7) → ").strip().lower()
        if choice == 'b' or choice == '':
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(LEARNING_PATH):
                cli_learning_stage(LEARNING_PATH[idx])
        except ValueError:
            pass


def cli_learning_stage(stage_data):
    """Show files in a learning stage."""
    stage_name, files = stage_data
    while True:
        clear_screen()
        cli_header()
        print(f"  ── 📚 {stage_name} ──\n")
        
        for i, (folder, fname, desc) in enumerate(files):
            filepath = os.path.join(BASE_DIR, folder, fname)
            exists = "✓" if os.path.exists(filepath) else "✗"
            display_name = fname.replace('.py', '').replace('_', ' ').title()
            print(f"    {i+1:2d}. [{exists}] {display_name:35s} — {desc}")
        
        print(f"\n    [B] Back")
        print()
        
        choice = input("  Select (number to view/run) → ").strip().lower()
        if choice == 'b' or choice == '':
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                folder, fname, desc = files[idx]
                filepath = os.path.join(BASE_DIR, folder, fname)
                title = fname.replace('.py', '').replace('_', ' ').title()
                cli_file_action(filepath, title, desc)
        except ValueError:
            pass


def cli_search():
    """Search for methods by keyword."""
    clear_screen()
    cli_header()
    print("  ── 🔍 Search Methods ──\n")
    query = input("  Search: ").strip().lower()
    
    if not query:
        return
    
    results = []
    for cat in CATEGORIES:
        for fname, title, desc in cat["files"]:
            # Search in title, description, and filename
            searchable = f"{title} {desc} {fname}".lower()
            if query in searchable:
                results.append((cat, fname, title, desc))
    
    if not results:
        # Also search docstrings
        for cat in CATEGORIES:
            for fname, title, desc in cat["files"]:
                filepath = os.path.join(BASE_DIR, cat["folder"], fname)
                if os.path.exists(filepath):
                    docstring = get_file_docstring(filepath).lower()
                    if query in docstring:
                        results.append((cat, fname, title, desc))
    
    if not results:
        print(f"\n  No results for '{query}'.")
        input("  Press Enter to continue...")
        return
    
    print(f"\n  Found {len(results)} result(s):\n")
    for i, (cat, fname, title, desc) in enumerate(results):
        print(f"    {i+1:2d}. [{cat['name']}] {title:30s} — {desc}")
    
    print(f"\n    [B] Back")
    print()
    
    choice = input("  Select (number) → ").strip().lower()
    if choice == 'b' or choice == '':
        return
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(results):
            cat, fname, title, desc = results[idx]
            filepath = os.path.join(BASE_DIR, cat["folder"], fname)
            cli_file_action(filepath, title, desc)
    except ValueError:
        pass


def cli_run_direct():
    """Run a file by path."""
    clear_screen()
    cli_header()
    print("  ── Run File ──\n")
    print("  Enter relative path (e.g., 01_Linear_Algebra/svd.py)")
    print()
    path = input("  Path: ").strip()
    
    if not path:
        return
    
    filepath = os.path.join(BASE_DIR, path)
    if os.path.exists(filepath):
        run_file(filepath)
        input("\n  Press Enter to continue...")
    else:
        print(f"\n  File not found: {filepath}")
        input("  Press Enter to continue...")


def cli_launcher():
    """Main CLI event loop."""
    while True:
        clear_screen()
        choice = cli_main_menu()
        
        if choice == 'b':
            cli_browse()
        elif choice == 'l':
            cli_learning_path()
        elif choice == 's':
            cli_search()
        elif choice == 'r':
            cli_run_direct()
        elif choice in ('q', 'quit', 'exit'):
            print("\n  Goodbye! Happy computing 🚀\n")
            break


# ============================================================
# GUI Launcher (tkinter)
# ============================================================

def gui_launcher():
    """Launch the tkinter GUI."""
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    import threading

    root = tk.Tk()
    root.title("Numerical Methods for Physics")
    root.geometry("1050x700")
    root.configure(bg="#1e1e2e")
    root.minsize(900, 600)

    # ── Color scheme ──
    BG       = "#1e1e2e"
    BG2      = "#282840"
    BG3      = "#313150"
    FG       = "#cdd6f4"
    FG_DIM   = "#7f849c"
    ACCENT   = "#89b4fa"
    ACCENT2  = "#a6e3a1"
    ACCENT3  = "#f9e2af"
    RED      = "#f38ba8"
    FONT     = ("Consolas", 10)
    FONT_B   = ("Consolas", 11, "bold")
    FONT_H   = ("Consolas", 14, "bold")
    FONT_SM  = ("Consolas", 9)

    style = ttk.Style()
    style.theme_use('clam')
    style.configure("Cat.TButton", font=FONT, padding=6,
                    background=BG2, foreground=FG)
    style.map("Cat.TButton",
              background=[("active", BG3)],
              foreground=[("active", ACCENT)])
    style.configure("Run.TButton", font=FONT_B, padding=8,
                    background="#45475a", foreground=ACCENT2)
    style.map("Run.TButton",
              background=[("active", "#585b70")])
    style.configure("Nav.TButton", font=FONT, padding=4,
                    background=BG2, foreground=ACCENT)
    style.map("Nav.TButton",
              background=[("active", BG3)])

    # ── Header ──
    header = tk.Frame(root, bg=BG, pady=8)
    header.pack(fill="x")
    tk.Label(header, text="⚛  Numerical Methods for Physics",
             font=FONT_H, bg=BG, fg=ACCENT).pack(side="left", padx=15)
    tk.Label(header, text="57 methods · 16 categories",
             font=FONT_SM, bg=BG, fg=FG_DIM).pack(side="left", padx=10)

    # ── Navigation tabs ──
    nav_frame = tk.Frame(root, bg=BG)
    nav_frame.pack(fill="x", padx=10)
    
    current_view = tk.StringVar(value="browse")

    def switch_view(view):
        current_view.set(view)
        for btn in nav_buttons:
            btn.configure(style="Nav.TButton")
        if view == "browse":
            nav_buttons[0].configure(style="Run.TButton")
            show_browse()
        elif view == "learn":
            nav_buttons[1].configure(style="Run.TButton")
            show_learning()
        elif view == "search":
            nav_buttons[2].configure(style="Run.TButton")
            show_search()

    nav_buttons = []
    for text, view in [("📂 Browse", "browse"), ("📚 Learning Path", "learn"), ("🔍 Search", "search")]:
        btn = ttk.Button(nav_frame, text=text, style="Nav.TButton",
                        command=lambda v=view: switch_view(v))
        btn.pack(side="left", padx=3, pady=4)
        nav_buttons.append(btn)

    # ── Main content area ──
    content = tk.PanedWindow(root, orient="horizontal", bg=BG,
                             sashwidth=4, sashrelief="flat")
    content.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # Left panel: category/file list
    left_frame = tk.Frame(content, bg=BG2, width=360)
    content.add(left_frame, minsize=280)

    left_header = tk.Label(left_frame, text="Categories", font=FONT_B,
                           bg=BG2, fg=ACCENT, anchor="w", padx=10, pady=6)
    left_header.pack(fill="x")

    list_canvas = tk.Canvas(left_frame, bg=BG2, highlightthickness=0)
    list_scrollbar = ttk.Scrollbar(left_frame, orient="vertical",
                                    command=list_canvas.yview)
    list_inner = tk.Frame(list_canvas, bg=BG2)
    
    list_inner.bind("<Configure>",
                    lambda e: list_canvas.configure(scrollregion=list_canvas.bbox("all")))
    list_canvas.create_window((0, 0), window=list_inner, anchor="nw")
    list_canvas.configure(yscrollcommand=list_scrollbar.set)
    
    list_scrollbar.pack(side="right", fill="y")
    list_canvas.pack(fill="both", expand=True)

    # Enable mousewheel scrolling
    def _on_mousewheel(event):
        list_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    list_canvas.bind_all("<MouseWheel>", _on_mousewheel)

    # Right panel: details and output
    right_frame = tk.Frame(content, bg=BG)
    content.add(right_frame, minsize=400)

    detail_header = tk.Label(right_frame, text="Select a method to view details",
                             font=FONT_B, bg=BG, fg=ACCENT3,
                             anchor="w", padx=10, pady=6)
    detail_header.pack(fill="x")

    # Description area
    desc_frame = tk.Frame(right_frame, bg=BG)
    desc_frame.pack(fill="both", expand=True)

    desc_text = scrolledtext.ScrolledText(desc_frame, font=("Consolas", 10),
                                          bg=BG2, fg=FG, insertbackground=FG,
                                          wrap="word", relief="flat",
                                          padx=12, pady=10,
                                          spacing1=2, spacing3=2)
    desc_text.pack(fill="both", expand=True, padx=5, pady=2)

    # ── Rich-text tags ──
    desc_text.tag_configure("title", font=("Consolas", 15, "bold"),
                           foreground=ACCENT, spacing1=4, spacing3=2)
    desc_text.tag_configure("heading", font=("Consolas", 12, "bold"),
                           foreground=ACCENT2, spacing1=8, spacing3=2)
    desc_text.tag_configure("subheading", font=("Consolas", 10, "bold"),
                           foreground=ACCENT3, spacing1=4)
    desc_text.tag_configure("equation", font=("Consolas", 10),
                           foreground="#cba6f7", background=BG3,
                           lmargin1=30, lmargin2=30,
                           spacing1=2, spacing3=2)
    desc_text.tag_configure("eq_header", font=("Consolas", 13, "bold"),
                           foreground="#cba6f7", spacing1=6, spacing3=4)
    desc_text.tag_configure("eq_label", font=("Consolas", 10, "italic"),
                           foreground="#f5c2e7", lmargin1=20, spacing1=4)
    desc_text.tag_configure("bullet", font=("Consolas", 10),
                           foreground=FG, lmargin1=20, lmargin2=35)
    desc_text.tag_configure("start_section", font=("Consolas", 12, "bold"),
                           foreground=ACCENT2, background="#1e3a2e",
                           spacing1=10, spacing3=4, lmargin1=5, lmargin2=5)
    desc_text.tag_configure("separator", foreground="#45475a")
    desc_text.tag_configure("normal", font=("Consolas", 10), foreground=FG,
                           lmargin1=10)
    desc_text.tag_configure("dim", font=("Consolas", 9), foreground=FG_DIM,
                           lmargin1=10)
    desc_text.tag_configure("code", font=("Consolas", 9),
                           foreground="#fab387", background=BG3,
                           lmargin1=30, lmargin2=30)

    # Image refs (prevent garbage collection of PhotoImages)
    _eq_images = []

    def _render_latex(latex_str, fontsize=14):
        """Render LaTeX string to a tkinter PhotoImage via matplotlib."""
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            import io, base64

            fig = Figure(figsize=(8, 0.5), dpi=110)
            fig.patch.set_facecolor(BG2)
            fig.text(0.02, 0.5, f"${latex_str}$",
                     fontsize=fontsize, color="#cba6f7",
                     va='center', ha='left')

            canvas = FigureCanvasAgg(fig)
            canvas.draw()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=110,
                        bbox_inches='tight', pad_inches=0.06,
                        facecolor=BG2, edgecolor='none')
            buf.seek(0)

            img_data = base64.b64encode(buf.read()).decode('ascii')
            return tk.PhotoImage(data=img_data)
        except Exception:
            return None

    def render_rich_text(content, filepath=None):
        """Render formatted text with highlighted equations & LaTeX images."""
        _eq_images.clear()

        desc_text.configure(state="normal")
        desc_text.delete("1.0", "end")

        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""

            # Title (followed by ===)
            if nxt and len(nxt) >= 3 and all(c == '=' for c in nxt):
                desc_text.insert("end", stripped + "\n", "title")
                desc_text.insert("end", "═" * min(len(stripped), 55) + "\n\n",
                                 "separator")
                i += 2; continue

            # Heading (followed by --- or ━━━)
            if nxt and len(nxt) >= 3 and all(c in '-━' for c in nxt):
                desc_text.insert("end", "\n" + stripped + "\n", "heading")
                desc_text.insert("end", "─" * min(len(stripped), 50) + "\n",
                                 "separator")
                i += 2; continue

            # "Where to start" section
            if stripped.lower().startswith('where to start'):
                desc_text.insert("end", "\n ▶ " + stripped + "\n",
                                 "start_section")
                i += 1; continue

            # Bold markers  **text** / **text:**
            if stripped.startswith("**"):
                end_b = stripped.find("**", 2)
                if end_b > 0:
                    bold = stripped[2:end_b]
                    rest = stripped[end_b + 2:]
                    desc_text.insert("end", "  " + bold, "subheading")
                    if rest:
                        desc_text.insert("end", rest, "normal")
                    desc_text.insert("end", "\n")
                    i += 1; continue

            # Equation lines (Unicode math)
            if is_math_line(stripped):
                desc_text.insert("end", "  " + stripped + "\n", "equation")
                i += 1; continue

            # Bullet points
            if stripped.startswith("- ") or stripped.startswith("* "):
                desc_text.insert("end", "  • " + stripped[2:] + "\n", "bullet")
                i += 1; continue

            # Numbered items (1. / 1) / 2. etc.)
            if stripped and stripped[0].isdigit() and len(stripped) > 2:
                if (stripped[1] in '.)' and len(stripped) > 2 and stripped[2] == ' '):
                    desc_text.insert("end", "  " + stripped + "\n", "bullet")
                    i += 1; continue
                if stripped[1].isdigit() and len(stripped) > 3 and stripped[2] in '.)':
                    desc_text.insert("end", "  " + stripped + "\n", "bullet")
                    i += 1; continue

            # Separator lines
            if stripped and len(stripped) >= 3 and all(c in '═─━-=~' for c in stripped):
                desc_text.insert("end", "─" * 40 + "\n", "separator")
                i += 1; continue

            # Empty line
            if not stripped:
                desc_text.insert("end", "\n")
                i += 1; continue

            # Normal text
            desc_text.insert("end", "  " + stripped + "\n", "normal")
            i += 1

        # ---- Rendered LaTeX equations (if available) ----
        if filepath:
            fname = os.path.basename(filepath)
            if fname in EQUATIONS:
                desc_text.insert("end", "\n\n")
                desc_text.insert("end", "━" * 45 + "\n", "separator")
                desc_text.insert("end",
                                 "  📐  Key Equations (Rendered)\n", "eq_header")
                desc_text.insert("end", "━" * 45 + "\n\n", "separator")

                for label, latex in EQUATIONS[fname]:
                    if label:
                        desc_text.insert("end", f"  {label}:\n", "eq_label")
                    photo = _render_latex(latex)
                    if photo:
                        _eq_images.append(photo)
                        desc_text.insert("end", "     ")
                        desc_text.image_create("end", image=photo)
                        desc_text.insert("end", "\n\n")
                    else:
                        desc_text.insert("end",
                                         f"      {latex}\n\n", "equation")

        desc_text.configure(state="disabled")

    # ── Welcome screen ──
    render_rich_text(
        "Numerical Methods for Physics\n"
        "==============================\n\n"
        "Welcome!  Browse categories on the left, or use the\n"
        "Learning Path for a recommended study order.\n\n"
        "Select any method to read its theory & equations, then\n"
        "click  \u25b6 Run Demo  to see it in action.\n\n"
        "Features\n"
        "--------\n"
        "- Rich formatted theory with highlighted equations\n"
        "- Rendered LaTeX equations (via matplotlib)\n"
        "- Runnable demos for every method\n"
        "- 7-stage learning roadmap\n"
        "- 16 categories \u00b7 58 implementations")

    # Button bar
    btn_bar = tk.Frame(right_frame, bg=BG, pady=6)
    btn_bar.pack(fill="x", padx=5)

    selected_file = tk.StringVar(value="")

    def run_selected():
        fpath = selected_file.get()
        if not fpath or not os.path.exists(fpath):
            messagebox.showwarning("No file", "Please select a method first.")
            return
        
        desc_text.configure(state="normal")
        desc_text.delete("1.0", "end")
        desc_text.insert("1.0", f"Running {os.path.basename(fpath)}...\n\n")
        desc_text.configure(state="disabled")
        run_btn.configure(state="disabled")
        
        def _run():
            try:
                result = subprocess.run(
                    [sys.executable, fpath],
                    capture_output=True, text=True,
                    cwd=os.path.dirname(fpath),
                    timeout=120
                )
                output = result.stdout
                if result.stderr:
                    output += "\n── stderr ──\n" + result.stderr
            except subprocess.TimeoutExpired:
                output = "⚠️ Script timed out after 120 seconds."
            except Exception as e:
                output = f"❌ Error: {e}"
            
            def _update():
                desc_text.configure(state="normal")
                desc_text.delete("1.0", "end")
                desc_text.insert("1.0", f"── Output: {os.path.basename(fpath)} ──\n\n")
                desc_text.insert("end", output)
                desc_text.configure(state="disabled")
                run_btn.configure(state="normal")
            
            root.after(0, _update)
        
        threading.Thread(target=_run, daemon=True).start()

    def view_docstring():
        fpath = selected_file.get()
        if not fpath or not os.path.exists(fpath):
            return
        docstring = get_file_docstring(fpath)
        render_rich_text(docstring, fpath)

    def show_equations():
        """Show only the rendered LaTeX equations for the selected method."""
        fpath = selected_file.get()
        if not fpath or not os.path.exists(fpath):
            messagebox.showinfo("No file", "Please select a method first.")
            return
        fname = os.path.basename(fpath)
        if fname not in EQUATIONS:
            desc_text.configure(state="normal")
            desc_text.delete("1.0", "end")
            _eq_images.clear()
            desc_text.insert("end",
                "\n  No rendered equations available for this method.\n\n",
                "normal")
            desc_text.insert("end",
                "  Click '📄 View Theory' to see the full description\n",
                "dim")
            desc_text.insert("end",
                "  with inline equations from the source code.\n", "dim")
            desc_text.configure(state="disabled")
            return
        _eq_images.clear()
        desc_text.configure(state="normal")
        desc_text.delete("1.0", "end")
        title = fname.replace('.py', '').replace('_', ' ').title()
        desc_text.insert("end",
                         f"\n  📐  {title} \u2014 Key Equations\n", "eq_header")
        desc_text.insert("end", "━" * 50 + "\n\n", "separator")
        for label, latex in EQUATIONS[fname]:
            if label:
                desc_text.insert("end", f"  {label}\n", "eq_label")
            photo = _render_latex(latex, fontsize=16)
            if photo:
                _eq_images.append(photo)
                desc_text.insert("end", "     ")
                desc_text.image_create("end", image=photo)
                desc_text.insert("end", "\n\n")
            else:
                desc_text.insert("end", f"    {latex}\n\n", "equation")
        desc_text.configure(state="disabled")

    run_btn = ttk.Button(btn_bar, text="▶  Run Demo", style="Run.TButton",
                        command=run_selected)
    run_btn.pack(side="left", padx=4)

    view_btn = ttk.Button(btn_bar, text="📄 View Theory", style="Nav.TButton",
                         command=view_docstring)
    view_btn.pack(side="left", padx=4)

    eq_btn = ttk.Button(btn_bar, text="📐 Equations", style="Nav.TButton",
                        command=show_equations)
    eq_btn.pack(side="left", padx=4)

    # ── View builders ──
    def clear_list():
        for w in list_inner.winfo_children():
            w.destroy()

    def make_file_button(parent, title, desc, filepath, indent=0):
        """Create a clickable file entry."""
        frame = tk.Frame(parent, bg=BG2, pady=1)
        frame.pack(fill="x", padx=(5 + indent*15, 5))
        
        exists = os.path.exists(filepath)
        mark = "✓" if exists else "✗"
        mark_color = ACCENT2 if exists else RED
        
        tk.Label(frame, text=mark, font=FONT_SM, bg=BG2,
                fg=mark_color, width=2).pack(side="left")
        
        btn = tk.Button(frame, text=title, font=FONT, bg=BG2, fg=FG,
                       activebackground=BG3, activeforeground=ACCENT,
                       relief="flat", anchor="w", cursor="hand2",
                       command=lambda: select_file(filepath, title, desc))
        btn.pack(side="left", fill="x", expand=True)
        
        tk.Label(frame, text=desc[:40], font=FONT_SM, bg=BG2,
                fg=FG_DIM, anchor="e").pack(side="right", padx=5)

    def select_file(filepath, title, desc):
        selected_file.set(filepath)
        detail_header.configure(text=f"  {title}")

        if os.path.exists(filepath):
            docstring = get_file_docstring(filepath)
            render_rich_text(docstring, filepath)
        else:
            desc_text.configure(state="normal")
            desc_text.delete("1.0", "end")
            desc_text.insert("1.0", "  File not found.", "normal")
            desc_text.configure(state="disabled")

    def show_browse():
        clear_list()
        left_header.configure(text="  Categories")
        
        for cat in CATEGORIES:
            # Category header
            cat_frame = tk.Frame(list_inner, bg=BG3, pady=4)
            cat_frame.pack(fill="x", padx=3, pady=(6, 1))
            
            tk.Label(cat_frame, text=f"  {cat['icon']}  {cat['name']}",
                    font=FONT_B, bg=BG3, fg=ACCENT,
                    anchor="w").pack(fill="x", padx=5)
            
            # Files
            for fname, title, desc in cat["files"]:
                filepath = os.path.join(BASE_DIR, cat["folder"], fname)
                make_file_button(list_inner, title, desc, filepath, indent=1)

    def show_learning():
        clear_list()
        left_header.configure(text="  📚 Learning Path")
        
        step = 1
        for stage_name, files in LEARNING_PATH:
            # Stage header
            stage_frame = tk.Frame(list_inner, bg=BG3, pady=4)
            stage_frame.pack(fill="x", padx=3, pady=(8, 1))
            tk.Label(stage_frame, text=f"  {stage_name}",
                    font=FONT_B, bg=BG3, fg=ACCENT3,
                    anchor="w").pack(fill="x", padx=5)
            
            for folder, fname, desc in files:
                filepath = os.path.join(BASE_DIR, folder, fname)
                title = f"{step}. {fname.replace('.py','').replace('_',' ').title()}"
                make_file_button(list_inner, title, desc, filepath, indent=1)
                step += 1

    def show_search():
        clear_list()
        left_header.configure(text="  🔍 Search")
        
        search_frame = tk.Frame(list_inner, bg=BG2, pady=8)
        search_frame.pack(fill="x", padx=5)
        
        tk.Label(search_frame, text="Search:", font=FONT,
                bg=BG2, fg=FG).pack(side="left", padx=5)
        
        search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=search_var,
                               font=FONT, bg=BG3, fg=FG,
                               insertbackground=FG, relief="flat")
        search_entry.pack(side="left", fill="x", expand=True, padx=5)
        search_entry.focus_set()
        
        results_frame = tk.Frame(list_inner, bg=BG2)
        results_frame.pack(fill="x")
        
        def do_search(*_):
            for w in results_frame.winfo_children():
                w.destroy()
            
            query = search_var.get().strip().lower()
            if len(query) < 2:
                return
            
            count = 0
            for cat in CATEGORIES:
                for fname, title, desc in cat["files"]:
                    searchable = f"{title} {desc} {fname} {cat['name']}".lower()
                    if query in searchable:
                        filepath = os.path.join(BASE_DIR, cat["folder"], fname)
                        make_file_button(results_frame,
                                        f"[{cat['id']}] {title}",
                                        desc, filepath)
                        count += 1
            
            if count == 0:
                # Search docstrings
                for cat in CATEGORIES:
                    for fname, title, desc in cat["files"]:
                        filepath = os.path.join(BASE_DIR, cat["folder"], fname)
                        if os.path.exists(filepath):
                            docstring = get_file_docstring(filepath).lower()
                            if query in docstring:
                                make_file_button(results_frame,
                                                f"[{cat['id']}] {title}",
                                                desc, filepath)
                                count += 1
            
            if count == 0:
                tk.Label(results_frame, text=f"  No results for '{query}'",
                        font=FONT, bg=BG2, fg=FG_DIM).pack(padx=10, pady=10)
        
        search_var.trace_add("write", do_search)
        search_entry.bind("<Return>", do_search)

    # ── Initialize ──
    switch_view("browse")

    # ── Status bar ──
    status = tk.Frame(root, bg=BG3, height=24)
    status.pack(fill="x", side="bottom")
    tk.Label(status, text="  Numerical Methods for Physics · Python + NumPy · 2024-2026",
             font=FONT_SM, bg=BG3, fg=FG_DIM, anchor="w").pack(fill="x", padx=5)

    root.mainloop()


# ============================================================
# Entry Point  (delegates to launcher_app package)
# ============================================================

# Re-export package symbols for backward compatibility
try:
    from launcher_app.config import BASE_DIR as _BASE_DIR
    from launcher_app.catalog import CATEGORIES as _CAT, LEARNING_PATH as _LP
    from launcher_app.equations import EQUATIONS as _EQ, is_math_line as _IML
    from launcher_app.utils import get_file_docstring as _GFD, run_file as _RF
    from launcher_app.cli import cli_launcher as _CLI
    from launcher_app.gui import gui_launcher as _GUI
except ImportError:
    pass  # package not available; monolithic launcher.py still works

if __name__ == "__main__":
    # Prefer the package if available; fall back to monolithic code
    try:
        from launcher_app.main import main
        main()
    except ImportError:
        # Fallback: use the functions defined above in this file
        mode = None
        if "--cli" in sys.argv:
            mode = "cli"
        elif "--gui" in sys.argv:
            mode = "gui"

        if mode == "cli":
            cli_launcher()
        elif mode == "gui":
            gui_launcher()
        else:
            try:
                import tkinter
                test_root = tkinter.Tk()
                test_root.withdraw()
                test_root.destroy()
                gui_launcher()
            except Exception:
                print("  GUI not available, starting terminal mode...\n")
                cli_launcher()
