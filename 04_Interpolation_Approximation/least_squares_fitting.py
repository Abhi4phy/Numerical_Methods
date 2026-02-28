"""
Least-Squares Fitting
======================
Find the best-fit curve that minimizes the sum of squared residuals:
    min_c Σ [yᵢ - model(xᵢ; c)]²

Types:
1. Linear least squares: model is linear in parameters (polynomial, etc.)
   → Solve the normal equations: (AᵀA) c = Aᵀ y
2. Nonlinear least squares: model is nonlinear in parameters
   → Use iterative methods (Gauss-Newton, Levenberg-Marquardt)

Properties:
- Maximum likelihood estimator for Gaussian noise.
- Sensitive to outliers (consider robust fitting for noisy data).
- Condition number of AᵀA can be problematic → use QR or SVD instead.

Physics applications:
- Fitting experimental data, calibration curves.
- Parameter estimation in models (decay rates, resonance frequencies).
"""

import numpy as np


def polynomial_fit(x, y, degree):
    """
    Fit a polynomial of given degree using least squares.
    
    Constructs the Vandermonde matrix and solves the normal equations.
    
    Parameters
    ----------
    x, y : ndarray – data points
    degree : int – polynomial degree
    
    Returns
    -------
    coeffs : ndarray – polynomial coefficients [a₀, a₁, ..., aₙ]
             where p(x) = a₀ + a₁x + a₂x² + ...
    """
    # Vandermonde matrix: A[i,j] = x[i]^j
    A = np.vander(x, degree + 1, increasing=True)
    
    # Normal equations: (AᵀA)c = Aᵀy
    # Using QR for better numerical stability
    Q, R = np.linalg.qr(A)
    coeffs = np.linalg.solve(R, Q.T @ y)
    
    return coeffs


def evaluate_polynomial(coeffs, x):
    """Evaluate polynomial with coefficients [a₀, a₁, ..., aₙ]."""
    result = np.zeros_like(np.atleast_1d(x), dtype=float)
    for i, c in enumerate(coeffs):
        result += c * np.atleast_1d(x)**i
    return result.squeeze()


def general_linear_fit(basis_funcs, x, y):
    """
    General linear least-squares fit with arbitrary basis functions.
    
    model(x) = c₁ φ₁(x) + c₂ φ₂(x) + ... + cₘ φₘ(x)
    
    Parameters
    ----------
    basis_funcs : list of callables – basis functions φᵢ(x)
    x, y : ndarray – data points
    
    Returns
    -------
    coeffs : ndarray – coefficients [c₁, c₂, ..., cₘ]
    """
    m = len(basis_funcs)
    n = len(x)
    A = np.zeros((n, m))
    
    for j, phi in enumerate(basis_funcs):
        A[:, j] = phi(x)
    
    # Solve using SVD (most robust)
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
    return coeffs


def gauss_newton(model, jacobian, x, y, p0, tol=1e-10, max_iter=100):
    """
    Gauss-Newton method for nonlinear least squares.
    
    Iteratively linearizes the model and solves:
        (JᵀJ) Δp = Jᵀ r
    where J is the Jacobian and r = y - model(x, p) is the residual.
    
    Parameters
    ----------
    model : callable – model(x, p) returning predicted y values
    jacobian : callable – jacobian(x, p) returning J matrix (n × m)
    x, y : ndarray – data
    p0 : ndarray – initial parameter guess
    
    Returns
    -------
    p : ndarray – optimized parameters
    history : list of residual norms
    """
    p = p0.astype(float).copy()
    history = []
    
    for k in range(max_iter):
        r = y - model(x, p)              # Residual
        J = jacobian(x, p)               # Jacobian
        
        res_norm = np.linalg.norm(r)
        history.append(res_norm)
        
        # Solve normal equations for step
        JtJ = J.T @ J
        Jtr = J.T @ r
        dp = np.linalg.solve(JtJ, Jtr)
        
        p += dp
        
        if np.linalg.norm(dp) < tol:
            print(f"Gauss-Newton converged in {k+1} iterations")
            break
    
    return p, history


def compute_fit_statistics(x, y, y_fit, n_params):
    """
    Compute goodness-of-fit statistics.
    
    Returns
    -------
    dict with R², adjusted R², RMSE, chi²
    """
    n = len(y)
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    
    R2 = 1 - ss_res / ss_tot
    R2_adj = 1 - (1 - R2) * (n - 1) / (n - n_params - 1)
    RMSE = np.sqrt(ss_res / n)
    chi2 = ss_res / (n - n_params)  # Reduced chi-squared
    
    return {'R2': R2, 'R2_adj': R2_adj, 'RMSE': RMSE, 'chi2_reduced': chi2}


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 60)
    print("LEAST-SQUARES FITTING DEMO")
    print("=" * 60)

    # --- Polynomial fitting ---
    print("\n--- Polynomial Fit ---")
    x = np.linspace(0, 5, 30)
    y_true = 1.0 + 2.0*x - 0.5*x**2
    y_noisy = y_true + np.random.randn(30) * 0.5

    for deg in [1, 2, 3]:
        coeffs = polynomial_fit(x, y_noisy, deg)
        y_fit = evaluate_polynomial(coeffs, x)
        stats = compute_fit_statistics(x, y_noisy, y_fit, deg + 1)
        print(f"  Degree {deg}: coeffs = {np.round(coeffs, 4)}, R² = {stats['R2']:.6f}")

    # --- Custom basis functions ---
    print("\n--- Custom Basis: f(x) = a·sin(x) + b·cos(x) + c ---")
    x2 = np.linspace(0, 2*np.pi, 50)
    y2 = 3*np.sin(x2) + 2*np.cos(x2) + 1 + np.random.randn(50)*0.3

    basis = [lambda x: np.sin(x), lambda x: np.cos(x), lambda x: np.ones_like(x)]
    coeffs2 = general_linear_fit(basis, x2, y2)
    print(f"  Fitted: {coeffs2[0]:.3f}·sin(x) + {coeffs2[1]:.3f}·cos(x) + {coeffs2[2]:.3f}")
    print(f"  True:   3.000·sin(x) + 2.000·cos(x) + 1.000")

    # --- Nonlinear fit: exponential decay ---
    print("\n--- Nonlinear Fit: y = a·exp(-b·x) + c ---")
    x3 = np.linspace(0, 5, 40)
    y3 = 5.0 * np.exp(-0.8 * x3) + 1.0 + np.random.randn(40) * 0.2

    def exp_model(x, p):
        return p[0] * np.exp(-p[1] * x) + p[2]

    def exp_jacobian(x, p):
        J = np.zeros((len(x), 3))
        J[:, 0] = np.exp(-p[1] * x)           # ∂f/∂a
        J[:, 1] = -p[0] * x * np.exp(-p[1]*x) # ∂f/∂b
        J[:, 2] = 1.0                          # ∂f/∂c
        return J

    p0 = np.array([4.0, 1.0, 0.5])  # Initial guess
    p_fit, hist = gauss_newton(exp_model, exp_jacobian, x3, y3, p0)
    print(f"  Fitted: a={p_fit[0]:.3f}, b={p_fit[1]:.3f}, c={p_fit[2]:.3f}")
    print(f"  True:   a=5.000, b=0.800, c=1.000")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Polynomial fit
        coeffs = polynomial_fit(x, y_noisy, 2)
        y_fit = evaluate_polynomial(coeffs, x)
        axes[0].scatter(x, y_noisy, c='gray', s=20, label='Data')
        x_fine = np.linspace(0, 5, 200)
        axes[0].plot(x_fine, evaluate_polynomial(coeffs, x_fine), 'r-', lw=2, label='Quadratic fit')
        axes[0].plot(x_fine, 1+2*x_fine-0.5*x_fine**2, 'k--', label='True')
        axes[0].set_title('Polynomial Least Squares')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Custom basis
        y2_fit = coeffs2[0]*np.sin(x2) + coeffs2[1]*np.cos(x2) + coeffs2[2]
        axes[1].scatter(x2, y2, c='gray', s=20, label='Data')
        axes[1].plot(x2, y2_fit, 'r-', lw=2, label='Fit')
        axes[1].set_title('Custom Basis Functions')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Nonlinear fit
        axes[2].scatter(x3, y3, c='gray', s=20, label='Data')
        x3f = np.linspace(0, 5, 200)
        axes[2].plot(x3f, exp_model(x3f, p_fit), 'r-', lw=2, label='Gauss-Newton fit')
        axes[2].plot(x3f, 5*np.exp(-0.8*x3f)+1, 'k--', label='True')
        axes[2].set_title('Nonlinear Least Squares')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("least_squares_fitting.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
