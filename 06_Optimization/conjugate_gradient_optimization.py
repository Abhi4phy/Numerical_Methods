"""
Conjugate Gradient Optimization & Quasi-Newton (BFGS)
=====================================================

Two powerful optimization methods that go beyond steepest descent.

1. **Conjugate Gradient (for optimization)**
   Unlike the CG linear solver (Ax=b), this CG minimizes general
   nonlinear functions by choosing conjugate search directions:

       d_0 = -∇f(x_0)
       d_{k+1} = -∇f(x_{k+1}) + β_k d_k

   β choices: Fletcher–Reeves, Polak–Ribière, Hestenes–Stiefel.
   The method achieves superlinear convergence on well-conditioned problems.

2. **BFGS (Broyden–Fletcher–Goldfarb–Shanno)**
   A quasi-Newton method that builds an approximation to the
   inverse Hessian H⁻¹ using rank-2 updates:

       B_{k+1} = B_k + (Δx Δx^T)/(Δx^T Δg)
                 - (B_k Δg (B_k Δg)^T)/(Δg^T B_k Δg)

   where Δx = x_{k+1} - x_k, Δg = ∇f_{k+1} - ∇f_k.

   BFGS converges superlinearly and doesn't need exact Hessians.

Physics: widely used for structure optimization (DFT geometry opt),
         potential energy minimization, inverse problems.
"""

import numpy as np


# ============================================================
# Nonlinear Conjugate Gradient
# ============================================================
def conjugate_gradient_optimize(f, grad_f, x0, method='PR',
                                 tol=1e-8, max_iter=10000):
    """
    Nonlinear Conjugate Gradient optimization.
    
    Parameters
    ----------
    f : callable — objective function
    grad_f : callable — gradient
    x0 : ndarray — starting point
    method : str — 'FR' (Fletcher-Reeves), 'PR' (Polak-Ribière),
                   'HS' (Hestenes-Stiefel)
    tol : float
    max_iter : int
    
    Returns
    -------
    x, history
    """
    x = np.array(x0, dtype=float)
    g = grad_f(x)
    d = -g.copy()
    history = [(x.copy(), np.linalg.norm(g), f(x))]
    
    for k in range(max_iter):
        grad_norm = np.linalg.norm(g)
        if grad_norm < tol:
            print(f"CG-opt ({method}) converged in {k} iterations")
            return x, history
        
        # Line search (backtracking)
        alpha = _backtracking_line_search(f, grad_f, x, d)
        
        x_new = x + alpha * d
        g_new = grad_f(x_new)
        
        # Compute β
        if method == 'FR':  # Fletcher-Reeves
            beta = np.dot(g_new, g_new) / np.dot(g, g)
        elif method == 'PR':  # Polak-Ribière
            beta = np.dot(g_new, g_new - g) / np.dot(g, g)
            beta = max(beta, 0)  # Restart if negative
        elif method == 'HS':  # Hestenes-Stiefel
            dg = g_new - g
            beta = np.dot(g_new, dg) / np.dot(d, dg) if np.dot(d, dg) != 0 else 0
        else:
            raise ValueError(f"Unknown method: {method}")
        
        d = -g_new + beta * d
        
        # Restart if direction is not descent
        if np.dot(d, g_new) > 0:
            d = -g_new
        
        x = x_new
        g = g_new
        history.append((x.copy(), np.linalg.norm(g), f(x)))
    
    print(f"CG-opt ({method}) did NOT converge after {max_iter} iterations")
    return x, history


# ============================================================
# BFGS Quasi-Newton
# ============================================================
def bfgs(f, grad_f, x0, tol=1e-8, max_iter=10000):
    """
    BFGS quasi-Newton optimization.
    
    Builds an approximation to the inverse Hessian from gradient
    differences. Achieves superlinear convergence.
    
    Returns
    -------
    x, history
    """
    n = len(x0)
    x = np.array(x0, dtype=float)
    g = grad_f(x)
    B = np.eye(n)  # Approximate inverse Hessian
    history = [(x.copy(), np.linalg.norm(g), f(x))]
    
    for k in range(max_iter):
        grad_norm = np.linalg.norm(g)
        if grad_norm < tol:
            print(f"BFGS converged in {k} iterations")
            return x, history
        
        # Search direction
        d = -B @ g
        
        # Backtracking line search
        alpha = _backtracking_line_search(f, grad_f, x, d)
        
        s = alpha * d  # Step
        x_new = x + s
        g_new = grad_f(x_new)
        y = g_new - g  # Gradient difference
        
        # BFGS inverse Hessian update
        sy = np.dot(s, y)
        if sy > 1e-16:  # Skip if curvature condition fails
            rho = 1.0 / sy
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            B = V @ B @ V.T + rho * np.outer(s, s)
        
        x = x_new
        g = g_new
        history.append((x.copy(), np.linalg.norm(g), f(x)))
    
    print(f"BFGS did NOT converge after {max_iter} iterations")
    return x, history


def l_bfgs(f, grad_f, x0, m=10, tol=1e-8, max_iter=10000):
    """
    Limited-memory BFGS (L-BFGS).
    
    Instead of storing the full n×n inverse Hessian, stores the
    last m (s, y) pairs and computes the search direction via
    the two-loop recursion. Memory: O(mn) instead of O(n²).
    
    Ideal for very large-scale optimization.
    """
    x = np.array(x0, dtype=float)
    g = grad_f(x)
    history = [(x.copy(), np.linalg.norm(g), f(x))]
    
    s_list = []  # Step vectors
    y_list = []  # Gradient differences
    rho_list = []
    
    for k in range(max_iter):
        grad_norm = np.linalg.norm(g)
        if grad_norm < tol:
            print(f"L-BFGS converged in {k} iterations")
            return x, history
        
        # Two-loop recursion to compute search direction
        q = g.copy()
        alphas = []
        
        for i in range(len(s_list) - 1, -1, -1):
            a = rho_list[i] * np.dot(s_list[i], q)
            alphas.insert(0, a)
            q = q - a * y_list[i]
        
        # Initial Hessian approximation
        if len(s_list) > 0:
            gamma = np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1])
            r = gamma * q
        else:
            r = q.copy()
        
        for i in range(len(s_list)):
            b = rho_list[i] * np.dot(y_list[i], r)
            r = r + (alphas[i] - b) * s_list[i]
        
        d = -r
        
        # Line search
        alpha = _backtracking_line_search(f, grad_f, x, d)
        
        s = alpha * d
        x_new = x + s
        g_new = grad_f(x_new)
        y = g_new - g
        
        sy = np.dot(s, y)
        if sy > 1e-16:
            if len(s_list) >= m:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)
            s_list.append(s)
            y_list.append(y)
            rho_list.append(1.0 / sy)
        
        x = x_new
        g = g_new
        history.append((x.copy(), np.linalg.norm(g), f(x)))
    
    return x, history


# ============================================================
# Helper: Backtracking line search
# ============================================================
def _backtracking_line_search(f, grad_f, x, d, alpha0=1.0, c=1e-4, rho=0.5):
    """Armijo backtracking line search."""
    alpha = alpha0
    fx = f(x)
    g = grad_f(x)
    slope = np.dot(g, d)
    
    for _ in range(50):
        if f(x + alpha * d) <= fx + c * alpha * slope:
            return alpha
        alpha *= rho
    
    return alpha


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CONJUGATE GRADIENT & BFGS OPTIMIZATION DEMO")
    print("=" * 60)

    # --- Rosenbrock ---
    # TO TEST: Change x0 and compare method='FR'/'PR'/'HS' plus BFGS and L-BFGS iterations; observe convergence speed and final f value.
    f_rosen = lambda x: (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    grad_rosen = lambda x: np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ])
    
    x0 = np.array([-1.0, 1.0])
    print(f"\nRosenbrock: f(x,y) = (1-x)² + 100(y-x²)²")
    print(f"Start: {x0}, True minimum: [1,1]\n")
    
    # CG variants
    for method in ['FR', 'PR', 'HS']:
        x_cg, h_cg = conjugate_gradient_optimize(f_rosen, grad_rosen, x0, method=method)
        print(f"  CG-{method}: x = [{x_cg[0]:.8f}, {x_cg[1]:.8f}], "
              f"f = {f_rosen(x_cg):.2e}, iters = {len(h_cg)}")
    
    # BFGS
    x_bfgs, h_bfgs = bfgs(f_rosen, grad_rosen, x0)
    print(f"  BFGS:  x = [{x_bfgs[0]:.8f}, {x_bfgs[1]:.8f}], "
          f"f = {f_rosen(x_bfgs):.2e}, iters = {len(h_bfgs)}")
    
    # L-BFGS
    x_lb, h_lb = l_bfgs(f_rosen, grad_rosen, x0)
    print(f"  L-BFGS: x = [{x_lb[0]:.8f}, {x_lb[1]:.8f}], "
          f"f = {f_rosen(x_lb):.2e}, iters = {len(h_lb)}")

    # --- Higher-dimensional test ---
    # TO TEST: Vary dimension n and condition number range (eigvals upper bound), then observe ||x-x*|| and iteration counts across CG/BFGS/L-BFGS.
    print("\n--- High-dimensional quadratic (n=50, κ=1000) ---")
    n = 50
    np.random.seed(42)
    eigvals = np.linspace(1, 1000, n)  # Condition number κ = 1000
    Q = np.random.randn(n, n)
    Q, _ = np.linalg.qr(Q)
    A = Q @ np.diag(eigvals) @ Q.T
    b = np.random.randn(n)
    
    f_quad = lambda x: 0.5 * x @ A @ x - b @ x
    grad_quad = lambda x: A @ x - b
    x_true = np.linalg.solve(A, b)
    
    x0_q = np.zeros(n)
    
    x_cg, h_cg = conjugate_gradient_optimize(f_quad, grad_quad, x0_q, method='PR')
    print(f"  CG-PR:  ||x - x*|| = {np.linalg.norm(x_cg - x_true):.2e}, iters = {len(h_cg)}")
    
    x_bfgs, h_bfgs = bfgs(f_quad, grad_quad, x0_q)
    print(f"  BFGS:   ||x - x*|| = {np.linalg.norm(x_bfgs - x_true):.2e}, iters = {len(h_bfgs)}")
    
    x_lb, h_lb = l_bfgs(f_quad, grad_quad, x0_q, m=20)
    print(f"  L-BFGS: ||x - x*|| = {np.linalg.norm(x_lb - x_true):.2e}, iters = {len(h_lb)}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        # Re-run on Rosenbrock for plotting
        x0 = np.array([-1.0, 1.0])
        _, h_cg_pr = conjugate_gradient_optimize(f_rosen, grad_rosen, x0, method='PR')
        _, h_bfgs = bfgs(f_rosen, grad_rosen, x0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Function value convergence
        fvals_cg = [h[2] for h in h_cg_pr]
        fvals_bfgs = [h[2] for h in h_bfgs]
        
        axes[0].semilogy(fvals_cg, 'b-', label='CG (Polak-Ribière)')
        axes[0].semilogy(fvals_bfgs, 'r-', label='BFGS')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('f(x)')
        axes[0].set_title('Convergence: Rosenbrock')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Paths on Rosenbrock contour
        x_range = np.linspace(-1.5, 1.5, 200)
        y_range = np.linspace(-0.5, 2.0, 200)
        X, Y = np.meshgrid(x_range, y_range)
        Z = (1-X)**2 + 100*(Y-X**2)**2

        axes[1].contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.5)
        
        path_cg = np.array([h[0] for h in h_cg_pr])
        path_bfgs = np.array([h[0] for h in h_bfgs])
        
        axes[1].plot(path_cg[:, 0], path_cg[:, 1], 'b.-', markersize=3, linewidth=0.7, label='CG-PR')
        axes[1].plot(path_bfgs[:, 0], path_bfgs[:, 1], 'r.-', markersize=3, linewidth=0.7, label='BFGS')
        axes[1].plot(1, 1, 'k*', markersize=15, label='Minimum')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_title('Optimization Paths')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("cg_bfgs_optimization.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
