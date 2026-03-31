"""
Gradient Descent
=================
First-order iterative optimization: move in the direction of
steepest descent (-∇f) to find a local minimum.

    x_{k+1} = x_k - α ∇f(x_k)

where α is the learning rate (step size).

Variants:
1. Fixed step size — simple but requires tuning α.
2. Line search — find optimal α along the descent direction.
3. Momentum — accelerate convergence, dampen oscillations.
4. Adaptive (Adam, RMSprop) — per-parameter learning rates.

Properties:
- Converges for convex functions with Lipschitz-continuous gradient.
- Rate: linear convergence, O(1/k) in general, O(κ^k) for strongly convex.
- κ = L/μ (condition number) determines speed.
- Zigzagging in narrow valleys is a known weakness.

Physics: Minimizing energy functionals, finding equilibrium states.
"""

import numpy as np


def gradient_descent(grad_f, x0, alpha=0.01, tol=1e-8, max_iter=10000):
    """
    Basic gradient descent with fixed step size.
    
    Parameters
    ----------
    grad_f : callable – gradient ∇f(x)
    x0 : ndarray – initial point
    alpha : float – learning rate
    tol : float – convergence tolerance on gradient norm
    max_iter : int
    
    Returns
    -------
    x : ndarray – minimizer
    history : list of (x, ||∇f||) tuples
    """
    x = np.array(x0, dtype=float)
    history = []
    
    for k in range(max_iter):
        g = np.atleast_1d(grad_f(x))
        grad_norm = np.linalg.norm(g)
        history.append((x.copy(), grad_norm))
        
        if grad_norm < tol:
            print(f"GD converged in {k} iterations (||∇f|| = {grad_norm:.2e})")
            return x, history
        
        x = x - alpha * g
    
    print(f"GD did NOT converge after {max_iter} iterations")
    return x, history


def gradient_descent_line_search(f, grad_f, x0, tol=1e-8, max_iter=10000):
    """
    Gradient descent with backtracking line search (Armijo condition).
    
    At each step, find α such that:
        f(x - α∇f) ≤ f(x) - c₁ α ||∇f||²
    
    Start with α=1 and shrink by factor ρ until condition is met.
    """
    x = np.array(x0, dtype=float)
    history = []
    c1 = 1e-4  # Sufficient decrease parameter
    rho = 0.5  # Backtracking factor
    
    for k in range(max_iter):
        g = np.atleast_1d(grad_f(x))
        grad_norm = np.linalg.norm(g)
        history.append((x.copy(), grad_norm))
        
        if grad_norm < tol:
            print(f"GD+LS converged in {k} iterations")
            return x, history
        
        # Backtracking line search
        alpha = 1.0
        fx = f(x)
        while f(x - alpha * g) > fx - c1 * alpha * grad_norm**2:
            alpha *= rho
            if alpha < 1e-16:
                break
        
        x = x - alpha * g
    
    return x, history


def gradient_descent_momentum(grad_f, x0, alpha=0.01, beta=0.9,
                               tol=1e-8, max_iter=10000):
    """
    Gradient descent with momentum (heavy-ball method).
    
    v_{k+1} = β v_k - α ∇f(x_k)
    x_{k+1} = x_k + v_{k+1}
    
    Momentum helps:
    - Accelerate through flat regions
    - Dampen oscillations in narrow valleys
    """
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    history = []
    
    for k in range(max_iter):
        g = np.atleast_1d(grad_f(x))
        grad_norm = np.linalg.norm(g)
        history.append((x.copy(), grad_norm))
        
        if grad_norm < tol:
            print(f"GD+Momentum converged in {k} iterations")
            return x, history
        
        v = beta * v - alpha * g
        x = x + v
    
    return x, history


def nesterov_accelerated(grad_f, x0, alpha=0.01, beta=0.9,
                          tol=1e-8, max_iter=10000):
    """
    Nesterov Accelerated Gradient (NAG).
    
    Look ahead: compute gradient at the "momentum point" instead of current point.
    
    v_{k+1} = β v_k - α ∇f(x_k + β v_k)
    x_{k+1} = x_k + v_{k+1}
    
    This achieves optimal convergence rate O(1/k²) for convex functions.
    """
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    history = []
    
    for k in range(max_iter):
        g = np.atleast_1d(grad_f(x + beta * v))  # Lookahead gradient
        grad_norm = np.linalg.norm(g)
        history.append((x.copy(), grad_norm))
        
        if grad_norm < tol:
            print(f"NAG converged in {k} iterations")
            return x, history
        
        v = beta * v - alpha * g
        x = x + v
    
    return x, history


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GRADIENT DESCENT DEMO")
    print("=" * 60)

    # --- Rosenbrock function (classic optimization test) ---
    # TO TEST: Try different learning rates (alpha = 0.0001, 0.01) and momentum (beta = 0.8, 0.95)
    # Compare convergence speeds of Fixed α vs Line Search vs Momentum
    print("\n--- Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)² ---")
    
    f_rosen = lambda x: (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    grad_rosen = lambda x: np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ])
    
    x0 = np.array([-1.0, 1.0])
    print(f"Starting point: {x0}")
    print(f"True minimum: [1, 1], f = 0\n")

    # Fixed step
    x_gd, h_gd = gradient_descent(grad_rosen, x0, alpha=0.001, max_iter=50000)
    print(f"  Fixed α=0.001: x = [{x_gd[0]:.6f}, {x_gd[1]:.6f}], "
          f"f = {f_rosen(x_gd):.2e}")

    # Line search
    x_ls, h_ls = gradient_descent_line_search(f_rosen, grad_rosen, x0)
    print(f"  Line search:   x = [{x_ls[0]:.6f}, {x_ls[1]:.6f}], "
          f"f = {f_rosen(x_ls):.2e}")

    # Momentum
    x_mom, h_mom = gradient_descent_momentum(grad_rosen, x0, alpha=0.001, beta=0.9, max_iter=50000)
    print(f"  Momentum:      x = [{x_mom[0]:.6f}, {x_mom[1]:.6f}], "
          f"f = {f_rosen(x_mom):.2e}")

    # --- Simple quadratic (shows condition number effect) ---
    # TO TEST: Change condition number from 100 to 10, 1000, or 10000 to see convergence differences
    print("\n--- Quadratic: f(x) = x₁² + 100·x₂² (κ=100) ---")
    grad_quad = lambda x: np.array([2*x[0], 200*x[1]])
    
    x_q, h_q = gradient_descent(grad_quad, [10.0, 10.0], alpha=0.005, max_iter=5000)
    print(f"  Result: [{x_q[0]:.8f}, {x_q[1]:.8f}], {len(h_q)} iters")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Rosenbrock contours + paths
        x_range = np.linspace(-2, 2, 200)
        y_range = np.linspace(-1, 3, 200)
        X, Y = np.meshgrid(x_range, y_range)
        Z = (1-X)**2 + 100*(Y-X**2)**2

        axes[0].contour(X, Y, Z, levels=np.logspace(-1, 3.5, 20), cmap='viridis', alpha=0.6)
        
        # Plot paths
        path_gd = np.array([p[0] for p in h_gd[:500]])
        path_ls = np.array([p[0] for p in h_ls[:500]])
        
        axes[0].plot(path_gd[:, 0], path_gd[:, 1], 'r-', alpha=0.7, linewidth=0.5, label='Fixed α')
        axes[0].plot(path_ls[:, 0], path_ls[:, 1], 'b-', alpha=0.7, linewidth=0.5, label='Line search')
        axes[0].plot(1, 1, 'k*', markersize=15)
        axes[0].plot(x0[0], x0[1], 'go', markersize=8, label='Start')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_title('Gradient Descent on Rosenbrock')
        axes[0].legend(fontsize=8)

        # Convergence
        grads_gd = [g for _, g in h_gd]
        grads_ls = [g for _, g in h_ls]
        grads_mom = [g for _, g in h_mom]
        
        axes[1].semilogy(grads_gd[:1000], 'r-', alpha=0.7, label='Fixed α')
        axes[1].semilogy(grads_ls[:1000], 'b-', alpha=0.7, label='Line search')
        axes[1].semilogy(grads_mom[:1000], 'g-', alpha=0.7, label='Momentum')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('||∇f||')
        axes[1].set_title('Convergence Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("gradient_descent.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
