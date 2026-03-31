"""
Cubic Spline Interpolation
============================
Piecewise cubic polynomials that are C² continuous (continuous function,
first derivative, and second derivative at the knots).

On each interval [xᵢ, xᵢ₊₁], the spline is:
    Sᵢ(x) = aᵢ + bᵢ(x-xᵢ) + cᵢ(x-xᵢ)² + dᵢ(x-xᵢ)³

Conditions:
1. Interpolation: S(xᵢ) = yᵢ                          (n+1 equations)
2. Continuity of S': Sᵢ'(xᵢ₊₁) = Sᵢ₊₁'(xᵢ₊₁)       (n-1 equations)
3. Continuity of S'': Sᵢ''(xᵢ₊₁) = Sᵢ₊₁''(xᵢ₊₁)    (n-1 equations)
4. Two extra conditions (boundary):
   - Natural: S''(x₀) = S''(xₙ) = 0
   - Clamped: S'(x₀) = f'(x₀), S'(xₙ) = f'(xₙ)

Advantages over polynomial interpolation:
- No Runge's phenomenon.
- Local control: changing one point only affects nearby panels.
- C² smoothness — great for physical simulations.
"""

import numpy as np


def cubic_spline_natural(x_nodes, y_nodes):
    """
    Natural cubic spline interpolation (S''(x₀) = S''(xₙ) = 0).
    
    Returns coefficients (a, b, c, d) for each interval.
    
    Sᵢ(x) = aᵢ + bᵢ(x-xᵢ) + cᵢ(x-xᵢ)² + dᵢ(x-xᵢ)³
    """
    n = len(x_nodes) - 1  # number of intervals
    h = np.diff(x_nodes)
    
    # a coefficients are just the y values
    a = y_nodes.copy()
    
    # Set up tridiagonal system for c coefficients
    # From continuity of S'' at interior nodes:
    # hᵢ₋₁ cᵢ₋₁ + 2(hᵢ₋₁+hᵢ) cᵢ + hᵢ cᵢ₊₁ = 3(δᵢ - δᵢ₋₁)
    # where δᵢ = (yᵢ₊₁ - yᵢ) / hᵢ
    
    delta = np.diff(y_nodes) / h
    
    # Tridiagonal system size (n-1) for interior c values
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
    
    # Natural BC: c₀ = 0, cₙ = 0
    A[0, 0] = 1
    A[n, n] = 1
    
    # Interior equations
    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        rhs[i] = 3 * (delta[i] - delta[i-1])
    
    # Solve for c
    c = np.linalg.solve(A, rhs)
    
    # Compute b and d from c
    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        b[i] = delta[i] - h[i] * (2*c[i] + c[i+1]) / 3
        d[i] = (c[i+1] - c[i]) / (3 * h[i])
    
    return a[:n], b, c[:n], d


def evaluate_spline(x_nodes, a, b, c, d, x):
    """
    Evaluate the cubic spline at point(s) x.
    """
    x = np.atleast_1d(np.array(x, dtype=float))
    result = np.zeros_like(x)
    
    n = len(a)
    for idx, xi in enumerate(x):
        # Find the right interval
        i = np.searchsorted(x_nodes[1:], xi, side='right')
        i = min(i, n - 1)
        
        dx = xi - x_nodes[i]
        result[idx] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
    
    return result.squeeze()


def cubic_spline_clamped(x_nodes, y_nodes, fp_left, fp_right):
    """
    Clamped cubic spline: S'(x₀) = fp_left, S'(xₙ) = fp_right.
    
    More accurate than natural spline when derivatives are known.
    """
    n = len(x_nodes) - 1
    h = np.diff(x_nodes)
    delta = np.diff(y_nodes) / h
    
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
    
    # Clamped BC at left: 2h₀c₀ + h₀c₁ = 3(δ₀ - f'(a))
    A[0, 0] = 2 * h[0]
    A[0, 1] = h[0]
    rhs[0] = 3 * (delta[0] - fp_left)
    
    # Clamped BC at right: hₙ₋₁cₙ₋₁ + 2hₙ₋₁cₙ = 3(f'(b) - δₙ₋₁)
    A[n, n-1] = h[n-1]
    A[n, n] = 2 * h[n-1]
    rhs[n] = 3 * (fp_right - delta[n-1])
    
    # Interior equations (same as natural)
    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        rhs[i] = 3 * (delta[i] - delta[i-1])
    
    c = np.linalg.solve(A, rhs)
    
    a = y_nodes[:n].copy()
    b_coeff = np.zeros(n)
    d_coeff = np.zeros(n)
    for i in range(n):
        b_coeff[i] = delta[i] - h[i] * (2*c[i] + c[i+1]) / 3
        d_coeff[i] = (c[i+1] - c[i]) / (3 * h[i])
    
    return a, b_coeff, c[:n], d_coeff


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CUBIC SPLINE INTERPOLATION DEMO")
    print("=" * 60)

    # --- Runge function: shows splines avoid Runge's phenomenon ---
    # TO TEST: Try different numbers of nodes (n = 5, 15, 25) and compare with Lagrange
    # Also test with uniform vs non-uniform node spacing
    print("\n--- Runge function: f(x) = 1/(1+25x²) ---")
    runge = lambda x: 1.0 / (1 + 25 * x**2)
    
    n = 11  # Try different values: 5, 15, 25
    x_nodes = np.linspace(-1, 1, n)
    y_nodes = runge(x_nodes)
    
    a, b, c, d = cubic_spline_natural(x_nodes, y_nodes)
    
    x_test = np.linspace(-1, 1, 200)
    y_spline = evaluate_spline(x_nodes, a, b, c, d, x_test)
    y_exact = runge(x_test)
    
    error = np.max(np.abs(y_spline - y_exact))
    print(f"  Natural spline error (n={n}): {error:.6e}")

    # --- Convergence ---
    print("\n--- Convergence study ---")
    for n in [5, 11, 21, 41, 81]:
        xn = np.linspace(-1, 1, n)
        yn = runge(xn)
        a, b, c, d = cubic_spline_natural(xn, yn)
        y_sp = evaluate_spline(xn, a, b, c, d, x_test)
        err = np.max(np.abs(y_sp - y_exact))
        print(f"  n={n:3d}  |  Max error = {err:.6e}")

    # --- Physical example: smooth trajectory ---
    print("\n--- Smooth trajectory through waypoints ---")
    t_nodes = np.array([0, 1, 2, 4, 5, 7, 8])
    x_data = np.array([0, 0.5, 1.2, 2.0, 2.5, 2.8, 3.0])
    y_data = np.array([0, 0.8, 1.5, 1.0, 0.5, 0.2, 0.0])
    
    ax, bx, cx, dx = cubic_spline_natural(t_nodes, x_data)
    ay, by, cy, dy = cubic_spline_natural(t_nodes, y_data)
    
    t_fine = np.linspace(0, 8, 200)
    x_smooth = evaluate_spline(t_nodes, ax, bx, cx, dx, t_fine)
    y_smooth = evaluate_spline(t_nodes, ay, by, cy, dy, t_fine)
    
    print("  Smoothly interpolated 7 waypoints with C² continuity.")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Runge function
        xn = np.linspace(-1, 1, 11)
        yn = runge(xn)
        a, b, c, d = cubic_spline_natural(xn, yn)
        y_sp = evaluate_spline(xn, a, b, c, d, x_test)
        
        axes[0].plot(x_test, y_exact, 'k-', lw=2, label='Exact')
        axes[0].plot(x_test, y_sp, 'b--', label='Cubic spline (n=11)')
        axes[0].plot(xn, yn, 'ro', markersize=6, label='Nodes')
        axes[0].set_title('Cubic Spline vs Runge Function')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Trajectory
        axes[1].plot(x_smooth, y_smooth, 'b-', lw=2, label='Spline path')
        axes[1].plot(x_data, y_data, 'ro', markersize=8, label='Waypoints')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_title('Smooth Trajectory')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect('equal')

        # Derivatives continuity
        xn = np.linspace(0, 2*np.pi, 8)
        yn = np.sin(xn)
        a, b, c, d = cubic_spline_natural(xn, yn)
        x_fine = np.linspace(0, 2*np.pi, 500)
        y_sp = evaluate_spline(xn, a, b, c, d, x_fine)
        
        # Numerical derivative of spline
        dy = np.gradient(y_sp, x_fine)
        axes[2].plot(x_fine, dy, 'b-', label="S'(x) (numerical)")
        axes[2].plot(x_fine, np.cos(x_fine), 'r--', label="cos(x) exact")
        axes[2].set_title("Derivative Smoothness")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("cubic_spline.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
