"""
Krylov Subspace Methods — GMRES & BiCGSTAB
=============================================
Solve large sparse linear systems Ax = b without forming A explicitly.

The Krylov subspace of order k is:
    K_k(A, b) = span{b, Ab, A²b, ..., A^{k-1}b}

**Conjugate Gradient** (in 01_Linear_Algebra/) works only for SPD matrices.
For general (non-symmetric, indefinite) systems, we need:

1. **GMRES (Generalized Minimum Residual)**
   - Minimizes ||b - Ax||₂ over K_k
   - Uses Arnoldi iteration to build orthonormal Krylov basis
   - Converges for any nonsingular A
   - Memory grows with iteration count → use restarting (GMRES(m))
   
2. **BiCGSTAB (Biconjugate Gradient Stabilized)**
   - For non-symmetric systems
   - Fixed memory per iteration (unlike GMRES)
   - Can have irregular convergence
   - Based on BiCG with stabilization

**Preconditioning:**
    Solve M⁻¹Ax = M⁻¹b instead of Ax = b.
    Good M ≈ A but easy to invert → fewer iterations.
    Common: ILU, Jacobi, SSOR, algebraic multigrid.

Physics: discretized PDEs (non-symmetric from advection, non-self-adjoint
operators), electromagnetics, fluid dynamics, linearized equations.

Where to start:
━━━━━━━━━━━━━━
Understand CG first (01_Linear_Algebra/conjugate_gradient.py) →
then GMRES for non-symmetric problems → add preconditioning.
"""

import numpy as np


def arnoldi_iteration(A_func, b, k):
    """
    Arnoldi iteration: build orthonormal basis for K_k(A, b).
    
    Produces:
    - Q : (n, k+1) orthonormal columns
    - H : (k+1, k) upper Hessenberg matrix
    
    Such that A @ Q[:,:k] = Q[:,:k+1] @ H
    
    A_func: callable that computes A @ v (matrix-free!)
    """
    n = len(b)
    Q = np.zeros((n, k + 1))
    H = np.zeros((k + 1, k))
    
    Q[:, 0] = b / np.linalg.norm(b)
    
    for j in range(k):
        v = A_func(Q[:, j])
        
        # Orthogonalize against previous vectors (modified Gram-Schmidt)
        for i in range(j + 1):
            H[i, j] = np.dot(Q[:, i], v)
            v = v - H[i, j] * Q[:, i]
        
        H[j + 1, j] = np.linalg.norm(v)
        
        if H[j + 1, j] < 1e-14:
            # Lucky breakdown — exact solution found
            return Q[:, :j+2], H[:j+2, :j+1]
        
        Q[:, j + 1] = v / H[j + 1, j]
    
    return Q, H


def gmres(A, b, x0=None, tol=1e-10, max_iter=None, restart=None):
    """
    GMRES — Generalized Minimum Residual method.
    
    Solves Ax = b for general nonsingular A.
    
    Parameters
    ----------
    A : ndarray or callable — matrix or matvec function
    b : ndarray
    x0 : initial guess (default: zeros)
    tol : convergence tolerance on relative residual
    max_iter : maximum iterations
    restart : restart after this many iterations (GMRES(m))
    
    Returns
    -------
    x : solution
    residuals : list of residual norms
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    if max_iter is None:
        max_iter = n
    
    # Matrix-vector product function
    if callable(A):
        A_func = A
    else:
        A_func = lambda v: A @ v
    
    if restart is None:
        restart = min(max_iter, n)
    
    x = x0.copy()
    residuals = []
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        return np.zeros(n), [0.0]
    
    total_iters = 0
    
    while total_iters < max_iter:
        r = b - A_func(x)
        r_norm = np.linalg.norm(r)
        residuals.append(r_norm)
        
        if r_norm / b_norm < tol:
            break
        
        # Arnoldi iteration
        m = min(restart, max_iter - total_iters)
        Q, H = arnoldi_iteration(A_func, r, m)
        
        # Solve least-squares: min ||beta*e1 - H*y||
        beta = r_norm
        e1 = np.zeros(H.shape[0])
        e1[0] = beta
        
        # Solve via QR of H
        y, _, _, _ = np.linalg.lstsq(H, e1, rcond=None)
        
        # Update solution
        x = x + Q[:, :len(y)] @ y
        total_iters += m
        
        # Check convergence
        r_new = b - A_func(x)
        r_new_norm = np.linalg.norm(r_new)
        residuals.append(r_new_norm)
        
        if r_new_norm / b_norm < tol:
            break
    
    return x, residuals


def bicgstab(A, b, x0=None, tol=1e-10, max_iter=None):
    """
    BiCGSTAB — Biconjugate Gradient Stabilized.
    
    For non-symmetric systems. Fixed memory per iteration.
    
    Parameters
    ----------
    A : ndarray or callable
    b : ndarray
    x0 : initial guess
    
    Returns
    -------
    x, residuals
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    if max_iter is None:
        max_iter = 2 * n
    
    if callable(A):
        A_func = A
    else:
        A_func = lambda v: A @ v
    
    x = x0.copy()
    r = b - A_func(x)
    r_hat = r.copy()  # Shadow residual (fixed)
    
    rho_old = 1.0
    alpha = 1.0
    omega = 1.0
    
    v = np.zeros(n)
    p = np.zeros(n)
    
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        return np.zeros(n), [0.0]
    
    residuals = [np.linalg.norm(r)]
    
    for k in range(max_iter):
        rho = np.dot(r_hat, r)
        
        if abs(rho) < 1e-30:
            print(f"BiCGSTAB: breakdown at iteration {k} (rho ≈ 0)")
            break
        
        beta = (rho / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        
        v = A_func(p)
        alpha = rho / np.dot(r_hat, v)
        
        s = r - alpha * v
        
        # Check if s is small enough
        s_norm = np.linalg.norm(s)
        if s_norm / b_norm < tol:
            x = x + alpha * p
            residuals.append(s_norm)
            break
        
        t = A_func(s)
        omega = np.dot(t, s) / np.dot(t, t)
        
        x = x + alpha * p + omega * s
        r = s - omega * t
        
        r_norm = np.linalg.norm(r)
        residuals.append(r_norm)
        
        if r_norm / b_norm < tol:
            break
        
        if abs(omega) < 1e-30:
            print(f"BiCGSTAB: breakdown at iteration {k} (omega ≈ 0)")
            break
        
        rho_old = rho
    
    return x, residuals


def preconditioned_gmres(A, b, M_inv, x0=None, tol=1e-10, max_iter=None, restart=30):
    """
    Left-preconditioned GMRES: solve M⁻¹Ax = M⁻¹b.
    
    M_inv : callable or matrix — applies M⁻¹ to a vector.
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    
    if callable(A):
        A_func = A
    else:
        A_func = lambda v: A @ v
    
    if callable(M_inv):
        M_inv_func = M_inv
    else:
        M_inv_func = lambda v: M_inv @ v
    
    # Preconditioned matvec
    PA_func = lambda v: M_inv_func(A_func(v))
    Pb = M_inv_func(b)
    
    return gmres(PA_func, Pb, x0, tol, max_iter, restart)


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("KRYLOV SUBSPACE METHODS: GMRES & BiCGSTAB")
    print("=" * 60)

    np.random.seed(42)

    # --- Non-symmetric system ---
    print("\n--- Non-symmetric system (n=100) ---")
    n = 100
    A = np.random.randn(n, n) + 5 * np.eye(n)  # Shifted for stability
    x_true = np.random.randn(n)
    b = A @ x_true
    
    x_gmres, res_gmres = gmres(A, b, tol=1e-10)
    x_bicgs, res_bicgs = bicgstab(A, b, tol=1e-10)
    
    print(f"GMRES:    ||x - x_true|| = {np.linalg.norm(x_gmres - x_true):.2e}, "
          f"iters = {len(res_gmres)}")
    print(f"BiCGSTAB: ||x - x_true|| = {np.linalg.norm(x_bicgs - x_true):.2e}, "
          f"iters = {len(res_bicgs)}")

    # --- Matrix-free operation ---
    print("\n--- Matrix-free: convection-diffusion operator ---")
    n = 200
    h = 1.0 / (n + 1)
    
    def conv_diff_matvec(v):
        """(-D∂²/∂x² + a∂/∂x) discretized with FD."""
        D = 0.01  # Diffusion
        a = 1.0   # Advection speed
        result = np.zeros_like(v)
        for i in range(len(v)):
            # Diffusion: -D(v[i+1] - 2v[i] + v[i-1])/h²
            vi = v[i]
            vim = v[i-1] if i > 0 else 0
            vip = v[i+1] if i < len(v)-1 else 0
            result[i] = -D * (vip - 2*vi + vim) / h**2
            # Advection: a(v[i] - v[i-1])/h (upwind)
            result[i] += a * (vi - vim) / h
        return result
    
    # RHS
    x_grid = np.linspace(h, 1-h, n)
    b_cd = np.sin(np.pi * x_grid)
    
    x_sol, res_cd = gmres(conv_diff_matvec, b_cd, tol=1e-8, restart=50)
    print(f"Conv-Diff: converged in {len(res_cd)} iterations")
    print(f"Final residual: {res_cd[-1]:.2e}")

    # --- Preconditioning effect ---
    print("\n--- Preconditioning comparison ---")
    n = 50
    # Ill-conditioned system
    A_ill = np.random.randn(n, n)
    A_ill = A_ill.T @ A_ill + 0.01 * np.eye(n)  # SPD but ill-conditioned
    x_true = np.ones(n)
    b_ill = A_ill @ x_true
    
    # No preconditioner
    _, res_no_pc = gmres(A_ill, b_ill, tol=1e-10, restart=n)
    
    # Jacobi preconditioner
    D_inv = np.diag(1.0 / np.diag(A_ill))
    _, res_jac = preconditioned_gmres(A_ill, b_ill, D_inv, tol=1e-10, restart=n)
    
    print(f"No preconditioner:  {len(res_no_pc)} iters, final res = {res_no_pc[-1]:.2e}")
    print(f"Jacobi precond:     {len(res_jac)} iters, final res = {res_jac[-1]:.2e}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].semilogy(res_gmres, 'b-', label='GMRES')
        axes[0].semilogy(res_bicgs, 'r-', label='BiCGSTAB')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Residual norm')
        axes[0].set_title('GMRES vs BiCGSTAB')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(x_grid, x_sol, 'b-', linewidth=1.5)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('u(x)')
        axes[1].set_title('Convection-Diffusion Solution (matrix-free)')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].semilogy(res_no_pc, 'b-', label='No preconditioner')
        axes[2].semilogy(res_jac, 'r-', label='Jacobi preconditioner')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Residual norm')
        axes[2].set_title('Effect of Preconditioning')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("krylov_methods.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
