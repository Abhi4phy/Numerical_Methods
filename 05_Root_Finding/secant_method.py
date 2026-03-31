"""
Secant Method
==============
A derivative-free variant of Newton's method that approximates f'
using the secant line through the two most recent iterates.

    x_{n+1} = x_n - f(x_n) · (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

Properties:
- Superlinear convergence: order ≈ 1.618 (the golden ratio φ).
- Does NOT require derivative computation.
- Needs TWO initial guesses (but no bracket required).
- More efficient than bisection, nearly as fast as Newton per step.
- Can fail if f(x_n) ≈ f(x_{n-1}).

Related: Regula Falsi (False Position) maintains a bracket like bisection
but uses secant-line interpolation.
"""

import numpy as np


def secant_method(f, x0, x1, tol=1e-12, max_iter=100):
    """
    Secant method for f(x) = 0.
    
    Parameters
    ----------
    f : callable
    x0, x1 : float – two initial guesses
    tol : float – convergence tolerance
    max_iter : int
    
    Returns
    -------
    root : float
    info : dict
    """
    f0 = f(x0)
    f1 = f(x1)
    history = [x0, x1]
    
    for k in range(max_iter):
        if abs(f1 - f0) < 1e-16:
            print("  Warning: f values too close, division by near-zero.")
            break
        
        # Secant formula
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        history.append(x2)
        
        if abs(x2 - x1) < tol:
            return x2, {'iterations': k + 1, 'history': history, 'converged': True}
        
        # Shift
        x0, f0 = x1, f1
        x1, f1 = x2, f(x2)
    
    return x1, {'iterations': max_iter, 'history': history, 'converged': False}


def regula_falsi(f, a, b, tol=1e-12, max_iter=100):
    """
    Regula Falsi (False Position) method.
    
    Like bisection, but uses the secant line through (a, f(a)) and (b, f(b))
    to find the next point, instead of the midpoint.
    
    Maintains a bracket [a, b] → always converges, but can be slow
    if one endpoint gets "stuck" (Illinois modification fixes this).
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    
    history = []
    
    for k in range(max_iter):
        # Secant interpolation within bracket
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        history.append((c, abs(b - a)))
        
        if abs(fc) < tol or abs(b - a) < tol:
            return c, {'iterations': k + 1, 'history': history}
        
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    
    return c, {'iterations': max_iter, 'history': history}


def illinois_method(f, a, b, tol=1e-12, max_iter=100):
    """
    Illinois method: Modified Regula Falsi that prevents stagnation.
    
    When one endpoint is retained, its f value is halved, preventing
    the other endpoint from getting stuck.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    
    history = []
    side = 0  # Track which side was last retained
    
    for k in range(max_iter):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        history.append((c, abs(b - a)))
        
        if abs(fc) < tol:
            return c, {'iterations': k + 1, 'history': history}
        
        if fa * fc < 0:
            b, fb = c, fc
            if side == -1:
                fa /= 2  # Illinois modification
            side = -1
        else:
            a, fa = c, fc
            if side == 1:
                fb /= 2  # Illinois modification
            side = 1
    
    return c, {'iterations': max_iter, 'history': history}


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SECANT METHOD DEMO")
    print("=" * 60)

    # --- Example 1 ---
    # Secant method needs TWO initial guesses (no bracket required like bisection)
    # TO TEST: Try different starting pairs: [0.5, 2.5], [1.0, 3.0], [0.0, 2.0]
    print("\n--- f(x) = x³ - x - 2 ---")
    f = lambda x: x**3 - x - 2
    
    root, info = secant_method(f, 1.0, 2.0)  # Change starting points to other pairs
    print(f"Root: {root:.15f}")
    print(f"f(root): {f(root):.2e}")
    print(f"Iterations: {info['iterations']}")

    # Show convergence order ≈ φ = 1.618
    errors = [abs(x - root) for x in info['history'] if abs(x - root) > 1e-16]
    print("\nConvergence order estimation:")
    for k in range(2, min(len(errors)-1, 8)):
        if errors[k-1] > 1e-15 and errors[k-2] > 1e-15:
            order = np.log(errors[k]/errors[k-1]) / np.log(errors[k-1]/errors[k-2])
            print(f"  k={k}: approx order = {order:.3f} (φ ≈ 1.618)")

    # --- Regula Falsi vs Illinois ---
    # Both methods maintain bracketing like bisection but use interpolation
    # TO TEST: Try other brackets or different functions
    # Example: regula_falsi(f2, 1, 4) or different f2 = lambda x: ...
    print("\n--- Comparing Regula Falsi vs Illinois ---")
    f2 = lambda x: x**3 - 2*x - 5

    root_rf, info_rf = regula_falsi(f2, 2, 3)  # Change [2, 3] to other brackets
    root_il, info_il = illinois_method(f2, 2, 3)  # Same bracket for comparison
    print(f"Regula Falsi: root = {root_rf:.12f}, iterations = {info_rf['iterations']}")
    print(f"Illinois:     root = {root_il:.12f}, iterations = {info_il['iterations']}")

    # --- Comparison table ---
    print("\n--- Method comparison: f(x) = cos(x) - x ---")
    f3 = lambda x: np.cos(x) - x
    fp3 = lambda x: -np.sin(x) - 1

    from bisection_method import bisection
    from newton_raphson import newton_raphson

    r_bis, i_bis = bisection(f3, 0, 1)
    r_new, i_new = newton_raphson(f3, fp3, 0.5)
    r_sec, i_sec = secant_method(f3, 0.0, 1.0)
    
    print(f"  Bisection:      root={r_bis:.12f}  iters={i_bis['iterations']}")
    print(f"  Newton-Raphson: root={r_new:.12f}  iters={i_new['iterations']}")
    print(f"  Secant:         root={r_sec:.12f}  iters={i_sec['iterations']}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Secant iterations
        x_plot = np.linspace(0.5, 2.5, 300)
        axes[0].plot(x_plot, f(x_plot), 'b-', lw=2)
        axes[0].axhline(y=0, color='k', linewidth=0.5)
        
        hist = info['history']
        for k in range(min(4, len(hist) - 1)):
            axes[0].plot(hist[k], f(hist[k]), 'ro', markersize=6)
            # Secant line
            if k > 0:
                x_sec = np.linspace(hist[k-1]-0.3, hist[k]+0.3, 50)
                slope = (f(hist[k]) - f(hist[k-1])) / (hist[k] - hist[k-1])
                axes[0].plot(x_sec, f(hist[k]) + slope*(x_sec - hist[k]), 
                            'r--', alpha=0.4)
        
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('f(x)')
        axes[0].set_title('Secant Method Iterations')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-5, 10)

        # Convergence comparison
        errors_sec = [abs(x - root) for x in info['history'] if abs(x - root) > 1e-16]
        errors_bis = [abs(c - root) for c, _ in i_bis['history'] if abs(c - root) > 1e-16]
        errors_new = [abs(x - root) for x in i_new['history'] if abs(x - root) > 1e-16]
        
        axes[1].semilogy(range(len(errors_bis)), errors_bis, 'bs-', label='Bisection')
        axes[1].semilogy(range(len(errors_sec)), errors_sec, 'go-', label='Secant')
        axes[1].semilogy(range(len(errors_new)), errors_new, 'r^-', label='Newton')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('|Error|')
        axes[1].set_title('Convergence Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("secant_method.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
