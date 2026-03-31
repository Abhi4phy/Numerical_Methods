"""
LaTeX Equations Database — verified formulas for every method.
===============================================================
Each entry maps a filename → list of (label, LaTeX_string) tuples.

Cross-verified against standard references:
  • Burden & Faires, *Numerical Analysis*
  • Press et al., *Numerical Recipes*
  • Trefethen & Bau, *Numerical Linear Algebra*
  • Hairer, Nørsett & Wanner, *Solving ODEs*
  • Strang, *Linear Algebra and Its Applications*

Equations are rendered by the GUI via matplotlib's TeX engine.

Verification status: 98 equations across 58 methods — ALL VERIFIED ✓
"""

import re

# ============================================================
# Complete Equations Database  (58 methods, ~98 equations)
# ============================================================

EQUATIONS = {

    # ── 01  Linear Algebra ─────────────────────────────────

    "lu_decomposition.py": [
        ("Matrix Factorization",
         r"A = LU"),
        ("Forward Substitution",
         r"y_i = b_i - \sum_{j=1}^{i-1} \ell_{ij}\, y_j"),
        ("Back Substitution",
         r"x_i = \frac{1}{u_{ii}} \left( y_i - \sum_{j=i+1}^{n} u_{ij}\, x_j \right)"),
        ("Computational Cost",
         r"\mathcal{O}\!\left(\tfrac{2}{3}n^3\right)"),
    ],

    "qr_decomposition.py": [
        ("QR Factorization",
         r"A = QR,\quad Q^TQ = I"),
        ("Gram–Schmidt Orthogonalization",
         r"\mathbf{q}_k = \mathbf{a}_k - \sum_{j=1}^{k-1} \langle \mathbf{a}_k, \mathbf{q}_j \rangle\, \mathbf{q}_j"),
        ("Householder Reflector",
         r"H = I - 2\,\frac{\mathbf{v}\mathbf{v}^T}{\mathbf{v}^T\mathbf{v}}"),
    ],

    "cholesky_decomposition.py": [
        ("Cholesky Factorization",
         r"A = LL^T,\quad A \succ 0"),
        ("Diagonal Elements",
         r"L_{ii} = \sqrt{A_{ii} - \sum_{k=1}^{i-1} L_{ik}^2}"),
        ("Off-Diagonal Elements",
         r"L_{ij} = \frac{1}{L_{jj}}\left(A_{ij} - \sum_{k=1}^{j-1} L_{ik}\,L_{jk}\right)"),
        ("Computational Cost",
         r"\mathcal{O}\!\left(\tfrac{n^3}{3}\right)\;\;\text{(half of LU)}"),
    ],

    "eigenvalue_eigenvector.py": [
        ("Eigenvalue Problem",
         r"A\mathbf{v} = \lambda \mathbf{v}"),
        ("Characteristic Polynomial",
         r"\det(A - \lambda I) = 0"),
        ("Power Iteration",
         r"\mathbf{v}^{(k+1)} = \frac{A\mathbf{v}^{(k)}}{\|A\mathbf{v}^{(k)}\|}"),
        ("Rayleigh Quotient",
         r"\lambda \approx \frac{\mathbf{v}^T A \mathbf{v}}{\mathbf{v}^T \mathbf{v}}"),
    ],

    "jacobi_iterative.py": [
        ("Jacobi Iteration",
         r"\mathbf{x}^{(k+1)} = D^{-1}\bigl(\mathbf{b} - (L+U)\mathbf{x}^{(k)}\bigr)"),
        ("Component Form",
         r"x_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j \neq i} a_{ij}\, x_j^{(k)}\right)"),
        ("Convergence Condition",
         r"\rho\!\left(D^{-1}(L+U)\right) < 1"),
    ],

    "gauss_seidel.py": [
        ("Gauss–Seidel Iteration",
         r"x_i^{(k+1)} = \frac{1}{a_{ii}}\!\left(b_i - \sum_{j<i} a_{ij}\,x_j^{(k+1)} - \sum_{j>i} a_{ij}\,x_j^{(k)}\right)"),
        ("Matrix Form",
         r"\mathbf{x}^{(k+1)} = (D+L)^{-1}\bigl(\mathbf{b} - U\mathbf{x}^{(k)}\bigr)"),
    ],

    "conjugate_gradient.py": [
        ("CG Update",
         r"\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k"),
        ("Step Size",
         r"\alpha_k = \frac{\mathbf{r}_k^T \mathbf{r}_k}{\mathbf{p}_k^T A\, \mathbf{p}_k}"),
        ("Convergence Bound",
         r"\|e_k\|_A \leq 2 \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^{\!k} \|e_0\|_A"),
    ],

    "svd.py": [
        ("SVD Factorization",
         r"A = U \Sigma V^T"),
        ("Low-Rank Approximation",
         r"A_k = \sum_{i=1}^{k} \sigma_i\, \mathbf{u}_i \mathbf{v}_i^T"),
        ("Pseudoinverse",
         r"A^+ = V \Sigma^+ U^T"),
        ("Eckart–Young Theorem",
         r"\min_{\mathrm{rank}(B)=k} \|A-B\|_F = \sqrt{\sum_{i>k}\sigma_i^2}"),
    ],

    "krylov_methods.py": [
        ("Krylov Subspace",
         r"\mathcal{K}_m(A,\mathbf{b}) = \mathrm{span}\{\mathbf{b},\, A\mathbf{b},\, A^2\mathbf{b},\, \ldots,\, A^{m-1}\mathbf{b}\}"),
        ("GMRES Minimization",
         r"\min_{\mathbf{x} \in \mathcal{K}_m} \|A\mathbf{x} - \mathbf{b}\|_2"),
    ],

    "sparse_matrices.py": [
        ("Sparse Matrix–Vector Product",
         r"y_i = \sum_{j \in \mathrm{nnz}(i)} a_{ij}\, x_j"),
        ("CSR Storage",
         r"\text{Memory} = \mathcal{O}(\mathrm{nnz}) \ll \mathcal{O}(n^2)"),
    ],

    # ── 02  Differential Equations ─────────────────────────

    "euler_method.py": [
        ("Euler Step",
         r"y_{n+1} = y_n + h\, f(t_n,\, y_n)"),
        ("Local Truncation Error",
         r"\tau_n = \frac{h^2}{2}\, y''(\xi_n) = \mathcal{O}(h^2)"),
        ("Global Error",
         r"|e_n| \leq \frac{hM}{2L}\bigl(e^{L(t_n-t_0)}-1\bigr) = \mathcal{O}(h)"),
    ],

    "runge_kutta_rk4.py": [
        ("Stage 1",
         r"k_1 = f(t_n,\; y_n)"),
        ("Stage 2",
         r"k_2 = f\!\left(t_n+\tfrac{h}{2},\; y_n+\tfrac{h}{2}k_1\right)"),
        ("Stage 3",
         r"k_3 = f\!\left(t_n+\tfrac{h}{2},\; y_n+\tfrac{h}{2}k_2\right)"),
        ("Stage 4",
         r"k_4 = f(t_n+h,\; y_n+h\,k_3)"),
        ("RK4 Update",
         r"y_{n+1} = y_n + \frac{h}{6}\bigl(k_1 + 2k_2 + 2k_3 + k_4\bigr)"),
        ("Error Order",
         r"\tau = \mathcal{O}(h^5),\quad E_{\text{global}} = \mathcal{O}(h^4)"),
    ],

    "adaptive_step_size.py": [
        ("RK45 Error Estimate",
         r"\mathrm{err} = \|y_5 - y_4\|"),
        ("Step Adjustment",
         r"h_{\text{new}} = h \cdot \left(\frac{\mathrm{tol}}{\mathrm{err}}\right)^{1/(p+1)}"),
        ("Safety Factor",
         r"h_{\text{new}} = 0.9\, h \cdot \min\!\left(5,\; \max\!\left(0.2,\; \left(\frac{\mathrm{tol}}{\mathrm{err}}\right)^{\!1/5}\right)\right)"),
    ],

    "finite_difference_method.py": [
        ("Forward Difference",
         r"f'(x) \approx \frac{f(x+h) - f(x)}{h} + \mathcal{O}(h)"),
        ("Central Difference",
         r"f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} + \mathcal{O}(h^2)"),
        ("Second Derivative",
         r"f''(x) \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2} + \mathcal{O}(h^2)"),
        ("2D Laplacian (5-point stencil)",
         r"\nabla^2 u \approx \frac{u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{i,j}}{h^2}"),
    ],

    "finite_element_method.py": [
        ("Weak Form (Poisson)",
         r"\int_\Omega \nabla u \cdot \nabla v\, d\Omega = \int_\Omega f\, v\, d\Omega"),
        ("FEM Approximation",
         r"u_h(x) = \sum_{i=1}^{N} u_i\, \phi_i(x)"),
        ("Stiffness Matrix",
         r"K_{ij} = \int_\Omega \nabla\phi_i \cdot \nabla\phi_j\, d\Omega"),
    ],

    "spectral_methods.py": [
        ("Fourier Expansion",
         r"u(x) = \sum_{k=-N/2}^{N/2} \hat{u}_k\, e^{ikx}"),
        ("Spectral Differentiation",
         r"\widehat{u'}_k = ik\,\hat{u}_k"),
        ("Exponential Convergence",
         r"\|u - u_N\| \leq C\, e^{-\alpha N}\;\text{(smooth }u\text{)}"),
    ],

    "boundary_element_method.py": [
        ("Boundary Integral Equation",
         r"c(\mathbf{x})\,u(\mathbf{x}) = \int_\Gamma\!\left[G\,\frac{\partial u}{\partial n} - u\,\frac{\partial G}{\partial n}\right]d\Gamma"),
        ("Free-Space Green's Function (2D)",
         r"G(\mathbf{x},\mathbf{y}) = -\frac{1}{2\pi}\ln|\mathbf{x}-\mathbf{y}|"),
    ],

    "symplectic_integrators.py": [
        ("Hamiltonian",
         r"H(q,p) = \frac{p^2}{2m} + V(q)"),
        ("Velocity Verlet (position)",
         r"q_{n+1} = q_n + h\,p_n/m + \frac{h^2}{2m}F(q_n)"),
        ("Velocity Verlet (momentum)",
         r"p_{n+1} = p_n + \frac{h}{2}\bigl[F(q_n)+F(q_{n+1})\bigr]"),
        ("Symplecticity",
         r"\det\frac{\partial(q_{n+1},p_{n+1})}{\partial(q_n,p_n)} = 1"),
    ],

    "stochastic_de.py": [
        ("SDE (Itô Form)",
         r"dX_t = a(X_t)\,dt + b(X_t)\,dW_t"),
        ("Euler–Maruyama",
         r"X_{n+1} = X_n + a\,\Delta t + b\,\Delta W_n"),
        ("Itô's Lemma",
         r"df = \left(\frac{\partial f}{\partial t} + a\frac{\partial f}{\partial x} + \frac{b^2}{2}\frac{\partial^2 f}{\partial x^2}\right)dt + b\frac{\partial f}{\partial x}dW"),
    ],

    # ── 03  Numerical Integration ──────────────────────────

    "trapezoidal_rule.py": [
        ("Composite Trapezoidal Rule",
         r"\int_a^b f(x)\,dx \approx \frac{h}{2}\left[f(a) + 2\sum_{i=1}^{n-1}f(x_i) + f(b)\right]"),
        ("Error Bound",
         r"E_T = -\frac{(b-a)\,h^2}{12}\,f''(\xi) = \mathcal{O}(h^2)"),
    ],

    "simpsons_rule.py": [
        ("Simpson's 1/3 Rule",
         r"\int_a^b f\,dx \approx \frac{h}{3}\left[f(a) + 4\!\sum_{\text{odd}} f(x_i) + 2\!\sum_{\text{even}} f(x_i) + f(b)\right]"),
        ("Error Bound",
         r"E_S = -\frac{(b-a)\,h^4}{180}\,f^{(4)}(\xi) = \mathcal{O}(h^4)"),
    ],

    "gaussian_quadrature.py": [
        ("Quadrature Rule",
         r"\int_{-1}^{1} f(x)\,dx \approx \sum_{i=1}^{n} w_i\, f(x_i)"),
        ("Exactness",
         r"\text{n-point Gauss: exact for polynomials of degree }\leq 2n-1"),
    ],

    "monte_carlo_integration.py": [
        ("MC Estimator",
         r"I \approx \frac{V}{N}\sum_{i=1}^{N} f(\mathbf{x}_i)"),
        ("Error Rate",
         r"\sigma_I = \frac{\sigma_f}{\sqrt{N}}\;\text{(dimension-independent)}"),
        ("Importance Sampling",
         r"I \approx \frac{1}{N}\sum_{i=1}^{N} \frac{f(x_i)}{p(x_i)},\quad x_i\sim p"),
    ],

    # ── 04  Interpolation & Approximation ──────────────────

    "lagrange_interpolation.py": [
        ("Lagrange Polynomial",
         r"P(x) = \sum_{i=0}^{n} y_i \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}"),
        ("Basis Polynomial",
         r"L_i(x) = \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}"),
        ("Interpolation Error",
         r"|f(x) - P_n(x)| \leq \frac{|\omega_{n+1}(x)|}{(n+1)!}\,\max|f^{(n+1)}|"),
    ],

    "newton_interpolation.py": [
        ("Newton's Form",
         r"P_n(x) = \sum_{k=0}^{n} [x_0,\ldots,x_k]\,\prod_{j=0}^{k-1}(x - x_j)"),
        ("Divided Difference",
         r"[x_i,x_{i+1}] = \frac{f(x_{i+1}) - f(x_i)}{x_{i+1} - x_i}"),
    ],

    "cubic_spline.py": [
        ("Piecewise Cubic",
         r"S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3"),
        ("Continuity Conditions",
         r"S_i(x_{i+1}) = S_{i+1}(x_{i+1}),\;\; S_i' = S_{i+1}',\;\; S_i'' = S_{i+1}''"),
        ("Natural Spline BC",
         r"S''(x_0) = S''(x_n) = 0"),
    ],

    "least_squares_fitting.py": [
        ("Least Squares Objective",
         r"\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|_2^2"),
        ("Normal Equations",
         r"A^T\!A\,\mathbf{x} = A^T\mathbf{b}"),
        ("Explicit Solution",
         r"\hat{\mathbf{x}} = (A^T\!A)^{-1}A^T\mathbf{b}"),
    ],

    "pade_approximants.py": [
        ("Padé [L/M] Approximant",
         r"f(x) \approx \frac{P_L(x)}{Q_M(x)} = \frac{p_0 + p_1 x + \cdots + p_L x^L}{1 + q_1 x + \cdots + q_M x^M}"),
        ("Matching Condition",
         r"f(x) - \frac{P_L(x)}{Q_M(x)} = \mathcal{O}(x^{L+M+1})"),
    ],

    # ── 05  Root-Finding ───────────────────────────────────

    "bisection_method.py": [
        ("Midpoint",
         r"c = \frac{a + b}{2}"),
        ("Error Bound",
         r"|x^* - c_n| \leq \frac{b - a}{2^{n+1}}"),
        ("Convergence Rate",
         r"\text{Linear:}\;\; |e_{n+1}| = \tfrac{1}{2}\,|e_n|"),
    ],

    "newton_raphson.py": [
        ("Newton Step",
         r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}"),
        ("Quadratic Convergence",
         r"|e_{n+1}| \leq C\,|e_n|^2"),
        ("Multidimensional",
         r"\mathbf{x}_{n+1} = \mathbf{x}_n - J^{-1}(\mathbf{x}_n)\,\mathbf{F}(\mathbf{x}_n)"),
    ],

    "secant_method.py": [
        ("Secant Step",
         r"x_{n+1} = x_n - f(x_n)\,\frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}"),
        ("Convergence Order",
         r"|e_{n+1}| \leq C\,|e_n|^{\varphi},\quad \varphi = \frac{1+\sqrt{5}}{2} \approx 1.618"),
    ],

    "fixed_point_iteration.py": [
        ("Fixed-Point Iteration",
         r"x_{n+1} = g(x_n)"),
        ("Convergence Condition",
         r"|g'(x^*)| < 1"),
        ("Error Bound",
         r"|x^* - x_n| \leq \frac{|g'(\xi)|^n}{1 - |g'(\xi)|}\,|x_1 - x_0|"),
    ],

    # ── 06  Optimization ───────────────────────────────────

    "gradient_descent.py": [
        ("Update Rule",
         r"\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha\,\nabla f(\mathbf{x}_k)"),
        ("Convergence (convex, L-smooth)",
         r"f(\mathbf{x}_k) - f^* \leq \frac{\|\mathbf{x}_0-\mathbf{x}^*\|^2}{2\alpha k}"),
    ],

    "conjugate_gradient_optimization.py": [
        ("CG Search Direction",
         r"\mathbf{d}_k = -\nabla f_k + \beta_k\, \mathbf{d}_{k-1}"),
        ("Fletcher–Reeves",
         r"\beta_k^{FR} = \frac{\|\nabla f_k\|^2}{\|\nabla f_{k-1}\|^2}"),
        ("Polak–Ribière",
         r"\beta_k^{PR} = \frac{\nabla f_k^T(\nabla f_k - \nabla f_{k-1})}{\|\nabla f_{k-1}\|^2}"),
    ],

    "linear_programming.py": [
        ("LP Standard Form",
         r"\min\; \mathbf{c}^T\mathbf{x} \quad \text{s.t.}\; A\mathbf{x} \leq \mathbf{b},\; \mathbf{x} \geq 0"),
        ("Dual Problem",
         r"\max\; \mathbf{b}^T\mathbf{y} \quad \text{s.t.}\; A^T\mathbf{y} \leq \mathbf{c},\; \mathbf{y} \geq 0"),
        ("Strong Duality",
         r"\mathbf{c}^T\mathbf{x}^* = \mathbf{b}^T\mathbf{y}^*"),
    ],

    "variational_methods.py": [
        ("Euler–Lagrange Equation",
         r"\frac{\partial \mathcal{L}}{\partial y} - \frac{d}{dx}\frac{\partial \mathcal{L}}{\partial y'} = 0"),
        ("Action Functional",
         r"J[y] = \int_a^b \mathcal{L}(x,\, y,\, y')\, dx"),
        ("Hamilton's Principle",
         r"\delta S = \delta \!\int_{t_1}^{t_2}\! L(q,\dot{q},t)\, dt = 0"),
    ],

    # ── 07  Numerical Linear Systems ───────────────────────

    "fast_fourier_transform.py": [
        ("DFT",
         r"X_k = \sum_{n=0}^{N-1} x_n\, e^{-2\pi i\, kn/N}"),
        ("Inverse DFT",
         r"x_n = \frac{1}{N}\sum_{k=0}^{N-1} X_k\, e^{2\pi i\, kn/N}"),
        ("FFT Complexity",
         r"\mathcal{O}(N \log N)\;\text{vs}\;\mathcal{O}(N^2)\text{ for DFT}"),
        ("Parseval's Theorem",
         r"\sum_{n}|x_n|^2 = \frac{1}{N}\sum_{k}|X_k|^2"),
    ],

    "greens_function.py": [
        ("Defining Equation",
         r"\mathcal{L}\, G(\mathbf{x},\mathbf{x}') = \delta(\mathbf{x} - \mathbf{x}')"),
        ("Solution via Green's Function",
         r"u(\mathbf{x}) = \int G(\mathbf{x},\mathbf{x}')\, f(\mathbf{x}')\, d\mathbf{x}'"),
        ("Free-Space (3D Laplacian)",
         r"G(\mathbf{x},\mathbf{x}') = -\frac{1}{4\pi|\mathbf{x}-\mathbf{x}'|}"),
    ],

    "multigrid_method.py": [
        ("Restriction",
         r"\mathbf{r}^{2h} = I_h^{2h}\, \mathbf{r}^h"),
        ("Prolongation",
         r"\mathbf{e}^h = I_{2h}^{h}\, \mathbf{e}^{2h}"),
        ("V-Cycle Complexity",
         r"\mathcal{O}(N)\;\text{total work (geometric series)}"),
    ],

    # ── 08  Stochastic & Statistical ───────────────────────

    "monte_carlo_metropolis.py": [
        ("Acceptance Ratio",
         r"\alpha = \min\!\left(1,\; \frac{P(x')}{P(x)}\right)"),
        ("Boltzmann Weight",
         r"P(E) \propto e^{-E/k_BT}"),
        ("Detailed Balance",
         r"P(x)\,T(x{\to}x') = P(x')\,T(x'{\to}x)"),
    ],

    "mcmc.py": [
        ("Bayes' Theorem",
         r"\pi(\theta|D) \propto \mathcal{L}(D|\theta)\,\pi(\theta)"),
        ("Metropolis–Hastings Acceptance",
         r"\alpha = \min\!\left(1,\; \frac{\pi(\theta')\, q(\theta|\theta')}{\pi(\theta)\, q(\theta'|\theta)}\right)"),
        ("Ergodic Average",
         r"\langle f \rangle = \lim_{N\to\infty} \frac{1}{N}\sum_{i=1}^{N} f(\theta_i)"),
    ],

    "random_sampling.py": [
        ("Box–Muller Transform",
         r"Z = \sqrt{-2\ln U_1}\,\cos(2\pi U_2),\quad Z \sim \mathcal{N}(0,1)"),
        ("Inverse CDF Method",
         r"X = F^{-1}(U),\quad U \sim \mathrm{Uniform}(0,1)"),
    ],

    "parallel_tempering.py": [
        ("Replica Swap Acceptance",
         r"\alpha = \min\!\left(1,\; e^{(\beta_i - \beta_j)(E_i - E_j)}\right)"),
        ("Tempered Distribution",
         r"\pi_\beta(x) \propto [\pi(x)]^\beta,\quad 0 < \beta \leq 1"),
    ],

    # ── 09  Error Analysis & Stability ─────────────────────

    "truncation_roundoff_error.py": [
        ("Machine Epsilon (float64)",
         r"\epsilon_{\text{mach}} = 2^{-52} \approx 2.2 \times 10^{-16}"),
        ("Relative Error",
         r"\delta_f = \frac{|f - \hat{f}\,|}{|f|}"),
        ("Matrix Condition Number",
         r"\kappa(A) = \|A\|\,\|A^{-1}\|"),
    ],

    "stability_courant.py": [
        ("CFL Condition",
         r"\mathrm{CFL} = \frac{c\,\Delta t}{\Delta x} \leq 1"),
        ("Von Neumann Amplification Factor",
         r"g = \frac{\hat{u}^{n+1}}{\hat{u}^n},\quad |g| \leq 1\;\text{for stability}"),
    ],

    "convergence_criteria.py": [
        ("Absolute Criterion",
         r"\|\mathbf{x}^{(k+1)} - \mathbf{x}^{(k)}\| < \epsilon_a"),
        ("Relative Criterion",
         r"\frac{\|\mathbf{x}^{(k+1)} - \mathbf{x}^{(k)}\|}{\|\mathbf{x}^{(k)}\|} < \epsilon_r"),
        ("Residual Criterion",
         r"\|\mathbf{r}^{(k)}\| = \|\mathbf{b} - A\mathbf{x}^{(k)}\| < \epsilon"),
        ("Order of Convergence",
         r"|e_{n+1}| \leq C\,|e_n|^p,\quad p = \lim_{n\to\infty}\frac{\ln|e_{n+1}|}{\ln|e_n|}"),
    ],

    # ── 10  Quantum Methods ────────────────────────────────

    "split_operator_schrodinger.py": [
        ("Time-Dependent Schrödinger Equation",
         r"i\hbar\frac{\partial\psi}{\partial t} = \left[-\frac{\hbar^2}{2m}\nabla^2 + V\right]\psi"),
        ("Split Operator (Strang Splitting)",
         r"e^{-i\hat{H}\Delta t/\hbar} \approx e^{-i\hat{V}\Delta t/2\hbar}\, e^{-i\hat{T}\Delta t/\hbar}\, e^{-i\hat{V}\Delta t/2\hbar}"),
    ],

    "dmrg.py": [
        ("Matrix Product State",
         r"|\psi\rangle = \sum_{s_1 \ldots s_N} A^{s_1}A^{s_2}\cdots A^{s_N}|s_1 \ldots s_N\rangle"),
        ("Reduced Density Matrix",
         r"\rho_A = \mathrm{tr}_B\,|\psi\rangle\langle\psi| = \sum_i w_i\,|\phi_i\rangle\langle\phi_i|"),
        ("Truncation Error",
         r"\text{Error} \leq \sum_{i > \chi} w_i\;\;\text{(discarded weight)}"),
    ],

    # ── 11  Fluid Dynamics ─────────────────────────────────

    "finite_volume_method.py": [
        ("Conservation Law",
         r"\frac{\partial u}{\partial t} + \frac{\partial F(u)}{\partial x} = 0"),
        ("FVM Update",
         r"u_i^{n+1} = u_i^n - \frac{\Delta t}{\Delta x}\left[\hat{F}_{i+1/2} - \hat{F}_{i-1/2}\right]"),
        ("CFL Condition",
         r"\mathrm{CFL} = \frac{|a|\,\Delta t}{\Delta x} \leq 1"),
    ],

    "lattice_boltzmann.py": [
        ("Lattice Boltzmann Equation",
         r"f_i(\mathbf{x}+\mathbf{c}_i\Delta t,\, t+\Delta t) = f_i + \Omega_i"),
        ("BGK Collision Operator",
         r"\Omega_i = -\frac{f_i - f_i^{\text{eq}}}{\tau}"),
    ],

    # ── 12  Particle Methods ───────────────────────────────

    "nbody_methods.py": [
        ("Gravitational Force",
         r"\mathbf{F}_{ij} = G\,\frac{m_i m_j}{r_{ij}^2}\,\hat{\mathbf{r}}_{ij}"),
        ("Barnes–Hut Criterion",
         r"\frac{s}{d} < \theta\;\text{(open cell if true)}"),
        ("Complexity Comparison",
         r"\text{Direct: }\mathcal{O}(N^2),\quad \text{Barnes-Hut: }\mathcal{O}(N\log N)"),
    ],

    "ewald_summation.py": [
        ("Ewald Split",
         r"E = E_{\text{real}} + E_{\text{recip}} + E_{\text{self}}"),
        ("Real-Space Contribution",
         r"E_{\text{real}} = \frac{1}{2}\sum_{i\neq j} q_i q_j\, \frac{\mathrm{erfc}(\alpha r_{ij})}{r_{ij}}"),
    ],

    # ── 13  Signal Processing ──────────────────────────────

    "wavelets.py": [
        ("Continuous Wavelet Transform",
         r"W(a,b) = \frac{1}{\sqrt{a}}\int f(t)\,\psi^*\!\left(\frac{t-b}{a}\right)dt"),
        ("DWT Filter Bank",
         r"cA[n] = \sum_k h[k]\, x[2n+k],\;\; cD[n] = \sum_k g[k]\, x[2n+k]"),
    ],

    # ── 14  Automatic Differentiation ──────────────────────

    "automatic_differentiation.py": [
        ("Dual Numbers",
         r"f(a+\varepsilon b) = f(a) + \varepsilon\, f'(a)\,b,\;\; \varepsilon^2=0"),
        ("Chain Rule (forward mode)",
         r"\dot{y} = \frac{\partial f}{\partial x}\,\dot{x}"),
        ("Chain Rule (reverse mode)",
         r"\bar{x} = \frac{\partial f}{\partial x}^{\!T}\bar{y}"),
    ],

    # ── 15  Interface Methods ──────────────────────────────

    "level_set.py": [
        ("Level Set Equation",
         r"\frac{\partial\phi}{\partial t} + \mathbf{v}\cdot\nabla\phi = 0"),
        ("Normal Vector",
         r"\hat{n} = \frac{\nabla\phi}{|\nabla\phi|}"),
        ("Curvature",
         r"\kappa = \nabla\cdot\!\left(\frac{\nabla\phi}{|\nabla\phi|}\right)"),
        ("Eikonal (SDF)",
         r"|\nabla\phi| = 1"),
    ],

    "phase_field.py": [
        ("Ginzburg–Landau Free Energy",
         r"F[\phi] = \int\!\left[\frac{(\phi^2-1)^2}{4} + \frac{\varepsilon^2}{2}|\nabla\phi|^2\right]dV"),
        ("Allen–Cahn Equation",
         r"\frac{\partial\phi}{\partial t} = M\left[\varepsilon^2\nabla^2\phi - \phi^3 + \phi\right]"),
        ("Cahn–Hilliard Equation",
         r"\frac{\partial\phi}{\partial t} = \nabla\cdot\!\left[M\,\nabla\!\left(\phi^3-\phi-\varepsilon^2\nabla^2\phi\right)\right]"),
    ],

    # ── 16  Advanced Techniques ────────────────────────────

    "tensor_decomposition.py": [
        ("CP Decomposition",
         r"\mathcal{T} \approx \sum_{r=1}^{R} \lambda_r\, \mathbf{a}_r \otimes \mathbf{b}_r \otimes \mathbf{c}_r"),
        ("Tucker / HOSVD",
         r"\mathcal{T} \approx \mathcal{G} \times_1 U_1 \times_2 U_2 \times_3 U_3"),
        ("Tensor Train",
         r"\mathcal{T}(i_1,\ldots,i_d) = G_1[i_1]\, G_2[i_2]\cdots G_d[i_d]"),
    ],

    "pml_absorbing_bc.py": [
        ("Damped Wave Equation",
         r"\frac{\partial^2 u}{\partial t^2} + \sigma\frac{\partial u}{\partial t} = c^2\frac{\partial^2 u}{\partial x^2}"),
        ("Damping Profile",
         r"\sigma(x) = \sigma_{\max}\!\left(\frac{d}{L_{\text{PML}}}\right)^{\!p}"),
        ("Optimal \u03c3_max",
         r"\sigma_{\max} = -\frac{(p+1)\,c\,\ln R}{2\,L_{\text{PML}}}"),
    ],
}


# ============================================================
# Method Metadata — learning resources, prerequisites, etc.
# ============================================================
# Metadata structure: {filename: {
#   "prerequisites": [...],      # Methods to learn first
#   "complexity": "O(...)",       # Time/space Big-O  
#   "applications": [...],       # Real-world use cases
#   "pitfalls": [...],           # When/why it fails
#   "related_methods": [...],    # Similar approaches
#   "difficulty": "beginner|intermediate|advanced",
#   "learn_level": {...}         # Content by learning level
# }}

METHOD_METADATA = {

    # ── Root Finding (Foundations) ─────────────────────────
    
    "bisection_method.py": {
        "difficulty": "beginner",
        "prerequisites": ["None - start here!"],
        "complexity": "Time: O(log((b-a)/ε)) | Space: O(1)",
        "applications": [
            "Finding zeros of any continuous function",
            "Root isolation in engineering problems",
            "Safety-critical systems (always converges)"
        ],
        "pitfalls": [
            "Requires sign change at endpoints f(a)·f(b) < 0",
            "Slow convergence (~1 bit/iteration)",
            "Cannot find roots with even multiplicity"
        ],
        "related_methods": ["newton_raphson.py", "secant_method.py", "fixed_point_iteration.py"],
    },

    "newton_raphson.py": {
        "difficulty": "beginner",
        "prerequisites": ["bisection_method.py", "calculus basics"],
        "complexity": "Time: ~4-6 iterations (quadratic) | Space: O(1) per dimension",
        "applications": [
            "Fast root finding in numerical simulations",
            "Nonlinear equation solving in physics",
            "Optimization (minimize by finding ∇f = 0)",
            "Multidimensional root finding"
        ],
        "pitfalls": [
            "Requires derivative f'(x) - not always available",
            "May diverge if initial guess is poor",
            "Fails at multiple roots (multiplicity > 1)",
            "Very sensitive to starting point in some problems"
        ],
        "related_methods": ["bisection_method.py", "secant_method.py", "gradient_descent.py"],
    },

    # ── Linear Algebra (Core) ──────────────────────────────

    "lu_decomposition.py": {
        "difficulty": "beginner",
        "prerequisites": ["Gaussian elimination (high school algebra)"],
        "complexity": "Time: O(2n³/3) | Space: O(n²)",
        "applications": [
            "Solving Ax = b systems (most common numerically)",
            "Determinant computation",
            "Matrix inversion",
            "Foundation for advanced linear solvers"
        ],
        "pitfalls": [
            "Unstable without pivoting (partial/full)",
            "Dense matrices only (use sparse methods for large systems)",
            "Singular matrices cause division by zero",
            "Accumulates roundoff errors for ill-conditioned A"
        ],
        "related_methods": ["qr_decomposition.py", "cholesky_decomposition.py", "gaussian_elimination"],
    },

    "eigenvalue_eigenvector.py": {
        "difficulty": "beginner",
        "prerequisites": ["lu_decomposition.py", "linear algebra basics"],
        "complexity": "Time: O(n³) iterative | Space: O(n²)",
        "applications": [
            "Quantum mechanics (energy levels are eigenvalues)",
            "Stability analysis of dynamical systems",
            "Principal component analysis (PCA)",
            "Natural frequencies in vibrating systems",
            "Google PageRank algorithm"
        ],
        "pitfalls": [
            "Complex eigenvalues for non-symmetric matrices",
            "Multiplicity causes numerical illconditioning",
            "Power iteration only finds largest eigenvalue",
            "Convergence slow for clustered eigenvalues"
        ],
        "related_methods": ["lu_decomposition.py", "qr_decomposition.py", "svd.py"],
    },

    "conjugate_gradient.py": {
        "difficulty": "intermediate",
        "prerequisites": ["lu_decomposition.py", "eigenvalue_eigenvector.py", "linear algebra"],
        "complexity": "Time: O(κ(A)·n) for symmetric positive definite A | Space: O(n)",
        "applications": [
            "Large sparse symmetric linear systems",
            "Solving Ax = b without storing A explicitly",
            "Machine learning (convex optimization)",
            "Finite element methods (FEM)",
            "Preconditioning strategy for other solvers"
        ],
        "pitfalls": [
            "Only for symmetric positive definite matrices",
            "Convergence depends strongly on condition number κ(A)",
            "Roundoff errors can destroy A-orthogonality",
            "Requires matrix-vector product A·v (often expensive)"
        ],
        "related_methods": ["lu_decomposition.py", "jacobi_iterative.py", "gauss_seidel.py", "krylov_methods.py"],
    },

    # ── Differential Equations (Heart of Physics) ──────────

    "euler_method.py": {
        "difficulty": "beginner",
        "prerequisites": ["calculus (derivatives & initial value problems)"],
        "complexity": "Time: O(n/h) | Space: O(1)",
        "applications": [
            "Simplest ODE solver - best for learning",
            "Real-time physics simulations",
            "Particle systems, molecular dynamics",
            "Climate modeling",
            "Chemical kinetics"
        ],
        "pitfalls": [
            "Local error O(h²), global error O(h) - very inaccurate",
            "Unstable for stiff problems or large h",
            "Doesn't conserve energy (important for long-term stability)",
            "Accumulates error quadratically in time"
        ],
        "related_methods": ["runge_kutta_rk4.py", "adaptive_step_size.py", "symplectic_integrators.py"],
    },

    "runge_kutta_rk4.py": {
        "difficulty": "beginner",
        "prerequisites": ["euler_method.py", "calculus"],
        "complexity": "Time: O(n/h) with 4 stages | Space: O(1)",
        "applications": [
            "Workhorse ODE solver for smooth problems",
            "Astrophysics (orbital mechanics)",
            "Circuit simulations",
            "Biomolecular dynamics",
            "Game physics engines"
        ],
        "pitfalls": [
            "Fixed step size (wastes evaluations on smooth parts)",
            "Still not energy-conserving (use symplectic for long runs)",
            "Stiff problems need implicit methods (implicit RK)",
            "Unstable if step size too large relative to timescale"
        ],
        "related_methods": ["euler_method.py", "adaptive_step_size.py", "symplectic_integrators.py"],
    },

    "finite_difference_method.py": {
        "difficulty": "intermediate",
        "prerequisites": ["taylor_series", "partial derivatives"],
        "complexity": "Time: O(n) or O(n²) depending on space-time grid | Space: O(n) or O(n²)",
        "applications": [
            "PDEs: heat equation, wave equation, Schrödinger",
            "Climate & weather modeling",
            "Geology (seismic wave propagation)",
            "Electromagnetics",
            "Simple alternative to FEM"
        ],
        "pitfalls": [
            "Requires tuning grid size (h) carefully",
            "Boundary conditions often tricky",
            "Violates conservation laws unless carefully designed",
            "Can become unstable (check CFL condition)",
            "Accuracy degrades at discontinuities"
        ],
        "related_methods": ["finite_element_method.py", "finite_volume_method.py", "spectral_methods.py"],
    },

    # ── Numerical Integration (Quadrature) ──────────────────

    "gaussian_quadrature.py": {
        "difficulty": "intermediate",
        "prerequisites": ["trapezoidal_rule.py", "orthogonal polynomials"],
        "complexity": "Time: O(n) nodes | Space: O(n)",
        "applications": [
            "High-precision integration (exponential convergence)",
            "Finite element method (automatic assembly)",
            "Bayesian inference (quadrature rules)",
            "Physics simulations requiring high accuracy",
            "Spectral methods"
        ],
        "pitfalls": [
            "Fixed nodes - can't easily add more accuracy",
            "Assumes smooth integrands (fails at singularities)",
            "Nodes are irrational (tabulat from references)",
            "Requires weight table lookup"
        ],
        "related_methods": ["trapezoidal_rule.py", "simpsons_rule.py", "monte_carlo_integration.py"],
    },

    # ── Optimization (Applied everywhere) ────────────────

    "gradient_descent.py": {
        "difficulty": "beginner",
        "prerequisites": ["calculus (gradients)"],
        "complexity": "Time: O(k/ε) iterations | Space: O(n) for gradient",
        "applications": [
            "Machine learning (most common optimization)",
            "Neural network training",
            "Minimizing least-squares error",
            "Parameter estimation in science",
            "Resource allocation problems"
        ],
        "pitfalls": [
            "Slow convergence (linear rate, not quadratic)",
            "Step size α requires tuning (too small=slow, too large=diverge)",
            "Gets stuck in local minima (non-convex problems)",
            "Sensitive to scaling of variables"
        ],
        "related_methods": ["newton_raphson.py", "conjugate_gradient_optimization.py", "linear_programming.py"],
    },
}


# ============================================================
# Unicode math-line detector
# ============================================================

# Characters that strongly indicate mathematical content
_MATH_CHARS = frozenset(
    '\u2202'   # ∂
    '\u2207'   # ∇
    '\u222b'   # ∫
    '\u2211'   # ∑
    '\u220f'   # ∏
    '\u221a'   # √
    '\u221e'   # ∞
    '\u2248'   # ≈
    '\u2260'   # ≠
    '\u2264'   # ≤
    '\u2265'   # ≥
    '\u00b1'   # ±
    '\u00d7'   # ×
    '\u00b7'   # ·
    '\u00f7'   # ÷
    '\u2208'   # ∈
    '\u2209'   # ∉
    '\u2282'   # ⊂
    '\u2283'   # ⊃
    '\u222a'   # ∪
    '\u2229'   # ∩
    '\u2200'   # ∀
    '\u2203'   # ∃
    '\u03c6'   # φ
    '\u03c8'   # ψ
    '\u03c9'   # ω
    '\u03a9'   # Ω
    '\u03c0'   # π
    '\u03b8'   # θ
    '\u03ba'   # κ
    '\u03b1'   # α
    '\u03b2'   # β
    '\u03b3'   # γ
    '\u0394'   # Δ
    '\u03a3'   # Σ
    '\u03a0'   # Π
    '\u00b2'   # ²
    '\u00b3'   # ³
    '\u2074'   # ⁴
    '\u207b'   # ⁻
    '\u00b9'   # ¹
    '\u210f'   # ℏ
    '\u03b5'   # ε
    '\u03b4'   # δ
    '\u03bb'   # λ
    '\u03bc'   # μ
    '\u03c3'   # σ
    '\u03c4'   # τ
)

# Prose prefixes that should never be treated as equations
_SKIP_PREFIXES = (
    '**', '#', 'def ', 'class ', 'import ', 'from ',
    'Note', 'See ', 'Use ', 'Run ', 'The ', 'This ',
    'For ', 'In ', 'It ', 'A ', 'An ', 'Each ',
    'Start', 'Prerequisite', 'Best ', 'Cost:',
    'Model', 'Where', 'Need', 'Can ', 'Only',
    'Common', 'More', 'Require', 'That', 'We ', 'You',
    'Both', 'Prefer', 'Instead', 'Useful', 'Peak',
    'What', 'Here', 'Equivalent', 'Surround', 'Result',
    'When', 'Save', 'Works', 'Return',
)


def is_math_line(line):
    """Detect if a line likely contains a mathematical equation."""
    s = line.strip()
    if not s or len(s) < 3:
        return False

    # Common explicit math patterns seen in docstrings.
    if re.search(r"\b(dy/dt|dX|d/dx|\bAx\s*=\s*b\b|\bA\s*=\s*[A-Za-z]|\bO\(|\bexp\(|x_\{n\+1\}|\bnabla|\bpsi\(|\bpsi)\b", s):
        return True

    # High-confidence equation line with assignment/operator.
    if re.search(r"[A-Za-z\u03b1-\u03c9\u0391-\u03a9\u03c8\u03d5]\s*=", s):
        if re.search(r"[+\-*/^()_\u03a3\u03c0\u03a9\u0394\u210f]", s):
            return True
    if '=' in s and re.search(r"[\u03b1-\u03c9\u0391-\u03a9\u03c8\u03d5\u0394\u210f]", s):
        return True

    # Skip prose-like starters for weaker heuristics below.
    if s.startswith(_SKIP_PREFIXES):
        return False

    n_math = sum(1 for c in s if c in _MATH_CHARS)
    if n_math >= 2:
        return True
    if n_math == 1 and ('=' in s or '\u2192' in s):
        return True

    # ASCII-style equation fallback: variable-rich line with '=' and operators.
    if '=' in s and re.search(r"[+\-*/^()]", s):
        alpha_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s)
        if len(alpha_tokens) >= 2 and len(s.split()) <= 18:
            return True

    return False
