"""
Newton-Raphson Method
======================
Uses the derivative to construct a tangent line approximation,
finding where it crosses zero.

    x_{n+1} = x_n - f(x_n) / f'(x_n)

Properties:
- Quadratic convergence near the root: error_{n+1} ≈ C · error_n².
- Requires f'(x) — can be expensive or unavailable.
- Can fail: diverge, cycle, or converge to wrong root.
- Generalizes to systems: x_{n+1} = x_n - J⁻¹(x_n) f(x_n).

Physics applications:
- Finding equilibrium configurations, eigenfrequencies.
- Implicit time-stepping in ODE/PDE solvers.
"""

import numpy as np


def newton_raphson(f, fp, x0, tol=1e-12, max_iter=100):
    """
    Newton-Raphson method for f(x) = 0.
    
    Parameters
    ----------
    f : callable – function
    fp : callable – derivative f'(x)
    x0 : float – initial guess
    tol : float – convergence tolerance
    max_iter : int
    
    Returns
    -------
    root : float
    info : dict with convergence history
    """
    x = x0
    history = [x]
    
    for k in range(max_iter):
        fx = f(x)
        fpx = fp(x)
        
        if abs(fpx) < 1e-16:
            print(f"  Warning: derivative near zero at x = {x:.6f}")
            break
        
        x_new = x - fx / fpx
        history.append(x_new)
        
        if abs(x_new - x) < tol:
            return x_new, {'iterations': k + 1, 'history': history, 'converged': True}
        
        x = x_new
    
    return x, {'iterations': max_iter, 'history': history, 'converged': False}


def newton_raphson_auto(f, x0, tol=1e-12, max_iter=100, h=1e-8):
    """
    Newton's method with automatic numerical differentiation.
    
    f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    """
    fp = lambda x: (f(x + h) - f(x - h)) / (2 * h)
    return newton_raphson(f, fp, x0, tol, max_iter)


def newton_system(F, J, x0, tol=1e-12, max_iter=100):
    """
    Newton's method for systems of nonlinear equations.
    
    Solve F(x) = 0 where F: R^n → R^n.
    
    x_{k+1} = x_k - J(x_k)^{-1} F(x_k)
    (Solve J·Δx = -F instead of inverting J)
    
    Parameters
    ----------
    F : callable – F(x) returning ndarray of length n
    J : callable – J(x) returning Jacobian matrix (n × n)
    x0 : ndarray – initial guess
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    for k in range(max_iter):
        Fx = np.array(F(x))
        Jx = np.array(J(x))
        
        # Solve J · Δx = -F
        dx = np.linalg.solve(Jx, -Fx)
        x = x + dx
        history.append(x.copy())
        
        if np.linalg.norm(dx) < tol:
            return x, {'iterations': k + 1, 'history': history, 'converged': True}
    
    return x, {'iterations': max_iter, 'history': history, 'converged': False}


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("NEWTON-RAPHSON METHOD DEMO")
    print("=" * 60)

    # --- Example 1: Scalar ---
    print("\n--- f(x) = x³ - x - 2 ---")
    f = lambda x: x**3 - x - 2
    fp = lambda x: 3*x**2 - 1
    
    root, info = newton_raphson(f, fp, 1.5)
    print(f"Root: {root:.15f}")
    print(f"f(root): {f(root):.2e}")
    print(f"Iterations: {info['iterations']}")
    
    # Show quadratic convergence
    print("\nQuadratic convergence:")
    errors = [abs(x - root) for x in info['history']]
    for k in range(len(errors) - 1):
        if errors[k] > 1e-15:
            ratio = errors[k+1] / errors[k]**2 if errors[k] > 0 else 0
            print(f"  k={k}: error = {errors[k]:.4e}, "
                  f"error_{k+1}/error_k² = {ratio:.4f}")

    # --- Example 2: Square root computation ---
    print("\n--- Computing √2 via x² - 2 = 0 ---")
    f_sqrt = lambda x: x**2 - 2
    fp_sqrt = lambda x: 2*x
    root_sqrt, _ = newton_raphson(f_sqrt, fp_sqrt, 1.0)
    print(f"  Newton: {root_sqrt:.15f}")
    print(f"  Exact:  {np.sqrt(2):.15f}")

    # --- Example 3: System of equations ---
    print("\n--- System: x² + y² = 1, x - y = 0 (circle ∩ line) ---")
    F = lambda x: [x[0]**2 + x[1]**2 - 1, x[0] - x[1]]
    J = lambda x: [[2*x[0], 2*x[1]], [1, -1]]
    
    root_sys, info_sys = newton_system(F, J, [0.5, 0.3])
    print(f"  Root: ({root_sys[0]:.12f}, {root_sys[1]:.12f})")
    print(f"  Expected: ({1/np.sqrt(2):.12f}, {1/np.sqrt(2):.12f})")
    print(f"  Iterations: {info_sys['iterations']}")

    # --- Example 4: Failure case ---
    print("\n--- Failure: f(x) = x^{1/3} near x=0 ---")
    f_fail = lambda x: np.sign(x) * np.abs(x)**(1/3)
    fp_fail = lambda x: (1/3) * np.abs(x)**(-2/3) if abs(x) > 1e-15 else 1e15
    root_fail, info_fail = newton_raphson(f_fail, fp_fail, 0.1, max_iter=20)
    print(f"  After 20 iterations: x = {root_fail:.6f} (diverges!)")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Newton iterations
        x_plot = np.linspace(0.5, 2.5, 300)
        axes[0].plot(x_plot, f(x_plot), 'b-', lw=2)
        axes[0].axhline(y=0, color='k', linewidth=0.5)
        
        for k in range(min(4, len(info['history']) - 1)):
            xk = info['history'][k]
            axes[0].plot(xk, f(xk), 'ro', markersize=6)
            # Tangent line
            xt = np.linspace(xk - 0.5, xk + 0.5, 50)
            axes[0].plot(xt, f(xk) + fp(xk)*(xt - xk), 'r--', alpha=0.5)
        
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('f(x)')
        axes[0].set_title('Newton-Raphson Iterations')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-5, 10)

        # Convergence comparison with bisection
        from bisection_method import bisection
        _, info_bis = bisection(f, 1, 2, tol=1e-15)
        
        errors_newton = [abs(x - root) for x in info['history'] if abs(x - root) > 1e-16]
        errors_bis = [abs(c - root) for c, _ in info_bis['history'] if abs(c - root) > 1e-16]
        
        axes[1].semilogy(range(len(errors_newton)), errors_newton, 'ro-', label='Newton (quadratic)')
        axes[1].semilogy(range(len(errors_bis)), errors_bis, 'bs-', label='Bisection (linear)')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('|Error|')
        axes[1].set_title('Convergence: Newton vs Bisection')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("newton_raphson.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
