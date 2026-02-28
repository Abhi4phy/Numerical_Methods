"""
Gauss-Seidel Iterative Solver
===============================
Improvement over Jacobi: uses the LATEST values of x as soon as they
are computed within the same iteration.

Update rule:  x^{k+1}_i = (b_i - Σ_{j<i} A_{ij} x^{k+1}_j
                                 - Σ_{j>i} A_{ij} x^{k}_j) / A_{ii}

Key properties:
- Uses new values immediately → faster convergence than Jacobi.
- Converges for SPD matrices (always) and diagonally dominant matrices.
- NOT parallelizable (unlike Jacobi) due to sequential dependencies.
- SOR (Successive Over-Relaxation) is a generalization with parameter ω.

Physics applications:
- Poisson solvers, iterative PDE solutions, steady-state heat conduction.
"""

import numpy as np


def gauss_seidel(A, b, x0=None, tol=1e-8, max_iter=10000):
    """
    Gauss-Seidel iterative method.
    
    Parameters
    ----------
    A : ndarray, shape (n, n)
    b : ndarray, shape (n,)
    x0 : initial guess
    tol : convergence tolerance (relative residual)
    max_iter : maximum iterations
    
    Returns
    -------
    x : solution vector
    history : list of residual norms
    """
    n = len(b)
    x = x0.copy() if x0 is not None else np.zeros(n)
    history = []

    for k in range(max_iter):
        for i in range(n):
            # Use updated values for j < i, old values for j > i
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
            x[i] = (b[i] - sigma) / A[i, i]

        residual = np.linalg.norm(b - A @ x) / (np.linalg.norm(b) + 1e-16)
        history.append(residual)

        if residual < tol:
            print(f"Gauss-Seidel converged in {k+1} iterations (residual: {residual:.2e})")
            return x, history

    print(f"Gauss-Seidel did NOT converge after {max_iter} iterations")
    return x, history


def sor(A, b, omega=1.5, x0=None, tol=1e-8, max_iter=10000):
    """
    Successive Over-Relaxation (SOR).
    
    Generalization of Gauss-Seidel with relaxation parameter ω:
        x^{k+1}_i = (1-ω) x^k_i + ω * gauss_seidel_update
    
    - ω = 1: reduces to Gauss-Seidel
    - 1 < ω < 2: over-relaxation (faster convergence for many problems)
    - 0 < ω < 1: under-relaxation (can help convergence for some systems)
    
    Optimal ω depends on the spectral radius of the iteration matrix.
    """
    n = len(b)
    x = x0.copy() if x0 is not None else np.zeros(n)
    history = []

    for k in range(max_iter):
        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
            x_gs = (b[i] - sigma) / A[i, i]  # Gauss-Seidel update
            x[i] = (1 - omega) * x[i] + omega * x_gs  # SOR blend

        residual = np.linalg.norm(b - A @ x) / (np.linalg.norm(b) + 1e-16)
        history.append(residual)

        if residual < tol:
            print(f"SOR (ω={omega}) converged in {k+1} iterations")
            return x, history

    print(f"SOR did NOT converge after {max_iter} iterations")
    return x, history


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GAUSS-SEIDEL & SOR DEMO")
    print("=" * 60)

    # Diagonally dominant system
    A = np.array([
        [10, -1,  2,  0],
        [-1, 11, -1,  3],
        [ 2, -1, 10, -1],
        [ 0,  3, -1,  8]
    ], dtype=float)
    b = np.array([6, 25, -11, 15], dtype=float)

    print("\nMatrix A:")
    print(A)
    print("b:", b)

    # Gauss-Seidel
    x_gs, hist_gs = gauss_seidel(A, b)
    print(f"Solution: {np.round(x_gs, 6)}")
    print(f"Direct:   {np.round(np.linalg.solve(A, b), 6)}")

    # SOR with different ω
    print()
    for omega in [0.8, 1.0, 1.2, 1.5]:
        x_sor, hist_sor = sor(A, b, omega=omega)

    # Convergence comparison
    try:
        import matplotlib.pyplot as plt
        
        _, hist_gs = gauss_seidel(A, b, tol=1e-12)
        _, hist_sor08 = sor(A, b, omega=0.8, tol=1e-12)
        _, hist_sor12 = sor(A, b, omega=1.2, tol=1e-12)
        _, hist_sor15 = sor(A, b, omega=1.5, tol=1e-12)

        plt.figure(figsize=(8, 5))
        plt.semilogy(hist_gs, label="Gauss-Seidel (ω=1)")
        plt.semilogy(hist_sor08, label="SOR ω=0.8")
        plt.semilogy(hist_sor12, label="SOR ω=1.2")
        plt.semilogy(hist_sor15, label="SOR ω=1.5")
        plt.xlabel("Iteration")
        plt.ylabel("Relative Residual")
        plt.title("Gauss-Seidel vs SOR Convergence")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("gauss_seidel_sor_convergence.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
