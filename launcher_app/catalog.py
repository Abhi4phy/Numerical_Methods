"""
Method Catalog — categories, file lists, and learning path.
=============================================================
All 16 categories with their file metadata, plus the
recommended 7-stage learning roadmap.
"""

# ============================================================
# Method Catalog  (16 categories, 58 files)
# ============================================================

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


# ============================================================
# Recommended 7-stage learning path
# ============================================================

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
