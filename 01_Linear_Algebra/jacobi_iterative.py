"""
Jacobi Iterative Solver
========================
Solves Ax = b iteratively by splitting A = D + (L + U), where
D is the diagonal, L is strictly lower, U is strictly upper.

Update rule:  x^{k+1}_i = (b_i - Σ_{j≠i} A_{ij} x^{k}_j) / A_{ii}

Key properties:
- Converges if A is strictly diagonally dominant (|A_ii| > Σ_{j≠i} |A_ij|).
- Each iteration is O(n²) — great for sparse systems.
- Embarrassingly parallel: each x_i update is independent.

Physics applications:
- Iterative solution of discretized Laplace/Poisson equations.
- Heat diffusion, electrostatics on grids.
"""

import numpy as np


def jacobi_solver(A, b, x0=None, tol=1e-8, max_iter=10000):
    """
    Jacobi iterative method for solving Ax = b.
    
    Parameters
    ----------
    A : ndarray, shape (n, n) – coefficient matrix
    b : ndarray, shape (n,) – right-hand side
    x0 : ndarray or None – initial guess (zeros if None)
    tol : float – convergence tolerance (relative residual)
    max_iter : int – maximum iterations
    
    Returns
    -------
    x : ndarray – solution
    history : list – residual norm at each iteration
    """
    n = len(b)
    x = x0.copy() if x0 is not None else np.zeros(n)
    history = []

    D_inv = 1.0 / np.diag(A)  # Precompute 1/A_ii

    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            # Sum of A[i,j]*x[j] for j ≠ i
            sigma = A[i, :] @ x - A[i, i] * x[i]
            x_new[i] = (b[i] - sigma) * D_inv[i]

        # Compute residual
        residual = np.linalg.norm(b - A @ x_new) / (np.linalg.norm(b) + 1e-16)
        history.append(residual)

        if residual < tol:
            print(f"Jacobi converged in {k+1} iterations (residual: {residual:.2e})")
            return x_new, history

        x = x_new

    print(f"Jacobi did NOT converge after {max_iter} iterations (residual: {residual:.2e})")
    return x, history


def jacobi_vectorized(A, b, x0=None, tol=1e-8, max_iter=10000):
    """
    Vectorized Jacobi (faster, no Python loops over i).
    
    x^{k+1} = D^{-1} (b - (L+U) x^k)
    """
    n = len(b)
    x = x0.copy() if x0 is not None else np.zeros(n)
    D_inv = 1.0 / np.diag(A)
    R = A - np.diag(np.diag(A))  # R = L + U (off-diagonal part)
    history = []

    for k in range(max_iter):
        x_new = D_inv * (b - R @ x)

        residual = np.linalg.norm(b - A @ x_new) / (np.linalg.norm(b) + 1e-16)
        history.append(residual)

        if residual < tol:
            print(f"Jacobi (vectorized) converged in {k+1} iterations")
            return x_new, history

        x = x_new

    print(f"Jacobi did NOT converge after {max_iter} iterations")
    return x, history


def is_diagonally_dominant(A):
    """Check if A is strictly diagonally dominant (sufficient for convergence)."""
    n = A.shape[0]
    for i in range(n):
        if abs(A[i, i]) <= np.sum(np.abs(A[i, :])) - abs(A[i, i]):
            return False
    return True


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("JACOBI ITERATIVE SOLVER DEMO")
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
    print("Diagonally dominant?", is_diagonally_dominant(A))

    # Solve
    x, history = jacobi_solver(A, b)
    print(f"\nSolution: {np.round(x, 6)}")
    print(f"Direct solution: {np.round(np.linalg.solve(A, b), 6)}")
    print(f"Residual: {np.linalg.norm(A @ x - b):.2e}")

    # Convergence plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.semilogy(history, 'b-', linewidth=1.5)
        plt.xlabel("Iteration")
        plt.ylabel("Relative Residual")
        plt.title("Jacobi Method Convergence")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("jacobi_convergence.png", dpi=150)
        plt.show()
        print("Convergence plot saved.")
    except ImportError:
        print("(matplotlib not available for plotting)")
