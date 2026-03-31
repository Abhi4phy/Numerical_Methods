"""
Conjugate Gradient Method
==========================
Solves Ax = b for symmetric positive-definite (SPD) matrices.

The most important iterative solver in computational science.
Converges in at most n iterations (exact arithmetic), but typically
much fewer for well-conditioned systems.

Key properties:
- Each iteration requires ONE matrix-vector product (O(n²), or O(nnz) for sparse).
- Convergence rate depends on condition number κ(A) = λ_max/λ_min.
- With preconditioning, convergence can be dramatically accelerated.
- Krylov subspace method: searches over K_k = span{b, Ab, A²b, ...}.

Physics applications:
- Large-scale FEM systems, PDE solvers, optimization.
- Any problem leading to SPD linear systems.
"""

import numpy as np


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
    """
    Conjugate Gradient method for Ax = b (A must be SPD).
    
    Algorithm:
        r₀ = b - A x₀       (initial residual)
        p₀ = r₀              (initial search direction)
        
        For k = 0, 1, 2, ...:
            α_k = r_k^T r_k / (p_k^T A p_k)     (step size)
            x_{k+1} = x_k + α_k p_k              (update solution)
            r_{k+1} = r_k - α_k A p_k            (update residual)
            β_k = r_{k+1}^T r_{k+1} / (r_k^T r_k)  (conjugacy parameter)
            p_{k+1} = r_{k+1} + β_k p_k          (new search direction)
    
    Returns
    -------
    x : solution
    history : list of residual norms
    """
    n = len(b)
    if max_iter is None:
        max_iter = 2 * n

    x = x0.copy() if x0 is not None else np.zeros(n)
    r = b - A @ x          # Initial residual
    p = r.copy()            # Initial search direction
    rs_old = r @ r          # r^T r
    history = [np.sqrt(rs_old)]

    for k in range(max_iter):
        Ap = A @ p                  # Matrix-vector product
        alpha = rs_old / (p @ Ap)   # Step size
        x = x + alpha * p          # Update solution
        r = r - alpha * Ap         # Update residual

        rs_new = r @ r
        history.append(np.sqrt(rs_new))

        if np.sqrt(rs_new) < tol:
            print(f"CG converged in {k+1} iterations (residual: {np.sqrt(rs_new):.2e})")
            return x, history

        beta = rs_new / rs_old     # Conjugacy coefficient
        p = r + beta * p           # New search direction
        rs_old = rs_new

    print(f"CG did NOT converge after {max_iter} iterations")
    return x, history


def preconditioned_cg(A, b, M_inv, x0=None, tol=1e-10, max_iter=None):
    """
    Preconditioned Conjugate Gradient (PCG).
    
    Uses preconditioner M ≈ A such that M^{-1}A has a smaller 
    condition number.  Common choices: Jacobi (M = diag(A)),
    incomplete Cholesky, multigrid.
    
    Parameters
    ----------
    M_inv : callable or ndarray
        If callable: M_inv(r) returns M^{-1} r.
        If ndarray: treated as the inverse preconditioner matrix.
    """
    n = len(b)
    if max_iter is None:
        max_iter = 2 * n

    # Handle M_inv as matrix or function
    if callable(M_inv):
        apply_precond = M_inv
    else:
        apply_precond = lambda r: M_inv @ r

    x = x0.copy() if x0 is not None else np.zeros(n)
    r = b - A @ x
    z = apply_precond(r)
    p = z.copy()
    rz_old = r @ z
    history = [np.linalg.norm(r)]

    for k in range(max_iter):
        Ap = A @ p
        alpha = rz_old / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap

        res_norm = np.linalg.norm(r)
        history.append(res_norm)

        if res_norm < tol:
            print(f"PCG converged in {k+1} iterations")
            return x, history

        z = apply_precond(r)
        rz_new = r @ z
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    print(f"PCG did NOT converge after {max_iter} iterations")
    return x, history


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CONJUGATE GRADIENT DEMO")
    print("=" * 60)

    # Create a large-ish SPD system
    np.random.seed(42)
    n = 100
    B = np.random.randn(n, n)
    A = B.T @ B + 10 * np.eye(n)  # Well-conditioned SPD
    b = np.random.randn(n)

    print(f"\nSystem size: {n}x{n}")
    evals = np.linalg.eigvalsh(A)
    kappa = evals[-1] / evals[0]
    print(f"Condition number: {kappa:.1f}")

    # --- Example 1: Standard Conjugate Gradient ---
    # Basic CG implementation for SPD systems
    # TO TEST: Change system size n (50 to 200, 500), modify condition number
    # by changing B or diagonal shift (10 to 5, 100), try different tolerance values,
    # or use different RHS vectors (random, ones, specific patterns)
    # Observe convergence iterations vs condition number
    print("\n--- Conjugate Gradient ---")
    x_cg, hist_cg = conjugate_gradient(A, b)
    print(f"Residual: {np.linalg.norm(A @ x_cg - b):.2e}")

    # --- Example 2: Preconditioned CG with Jacobi ---
    # Shows how preconditioning reduces iteration count
    # TO TEST: Try other preconditioners (SSOR, ILU), change the diagonal shift,
    # compare with other preconditioner strategies, modify system conditioning,
    # or test with highly non-uniform diagonal (some diagonal elements very small)
    print("\n--- Preconditioned CG (Jacobi) ---")
    M_inv = np.diag(1.0 / np.diag(A))  # Jacobi preconditioner
    x_pcg, hist_pcg = preconditioned_cg(A, b, M_inv)
    print(f"Residual: {np.linalg.norm(A @ x_pcg - b):.2e}")

    # --- Example 3: Ill-conditioned system comparison ---
    # Demonstrates the critical importance of preconditioning for poor systems
    # TO TEST: Change the diagonal scaling (0.01 to 0.001, 0.1),
    # increase problem size n, try different initial guesses,
    # compare with other preconditioners, or modify the condition number manually
    # Observe how many iterations CG needs vs PCG on ill-conditioned systems
    print("\n--- Ill-conditioned system ---")
    A_ill = B.T @ B + 0.01 * np.eye(n)
    kappa_ill = np.linalg.eigvalsh(A_ill)[-1] / np.linalg.eigvalsh(A_ill)[0]
    print(f"Condition number: {kappa_ill:.1f}")
    x_ill, hist_ill = conjugate_gradient(A_ill, b, tol=1e-8)

    # Plot convergence
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.semilogy(hist_cg, 'b-', label=f"CG (κ={kappa:.0f})", linewidth=1.5)
        plt.semilogy(hist_pcg, 'r--', label=f"PCG Jacobi (κ={kappa:.0f})", linewidth=1.5)
        plt.semilogy(hist_ill, 'g:', label=f"CG (κ={kappa_ill:.0f})", linewidth=1.5)
        plt.xlabel("Iteration")
        plt.ylabel("||r||")
        plt.title("Conjugate Gradient Convergence")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("conjugate_gradient_convergence.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
