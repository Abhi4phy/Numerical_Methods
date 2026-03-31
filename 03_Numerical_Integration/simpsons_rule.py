"""
Simpson's Rule
===============
Approximates ∫_a^b f(x) dx using quadratic (parabolic) interpolation
through groups of three points.

Formula (composite):
    ∫_a^b f(x) dx ≈ h/3 [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + f(x_n)]

Properties:
- Fourth-order accurate: error = O(h⁴).
- Exact for polynomials up to degree 3.
- Error term: -(b-a) h⁴ f⁴(ξ) / 180 for some ξ.
- Requires an EVEN number of subintervals (odd number of points).
- Simpson's 3/8 rule uses cubic interpolation.
"""

import numpy as np


def simpsons_rule(f, a, b, n):
    """
    Composite Simpson's 1/3 Rule.
    
    Parameters
    ----------
    f : callable – integrand
    a, b : float – limits
    n : int – number of subintervals (MUST be even)
    
    Returns
    -------
    result : float – approximate integral
    """
    if n % 2 != 0:
        raise ValueError("Simpson's rule requires an even number of subintervals.")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # Simpson's weights: 1, 4, 2, 4, 2, ..., 4, 1
    result = y[0] + y[-1]
    result += 4 * np.sum(y[1:-1:2])   # Odd-indexed points (weight 4)
    result += 2 * np.sum(y[2:-1:2])   # Even-indexed points (weight 2)
    result *= h / 3

    return result


def simpsons_38_rule(f, a, b, n):
    """
    Composite Simpson's 3/8 Rule (based on cubic interpolation).
    
    ∫ ≈ 3h/8 [f₀ + 3f₁ + 3f₂ + 2f₃ + 3f₄ + 3f₅ + 2f₆ + ... + f_n]
    
    Requires n to be a multiple of 3. Fourth-order accurate.
    """
    if n % 3 != 0:
        raise ValueError("Simpson's 3/8 rule requires n divisible by 3.")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    result = y[0] + y[-1]
    for i in range(1, n):
        if i % 3 == 0:
            result += 2 * y[i]
        else:
            result += 3 * y[i]
    result *= 3 * h / 8

    return result


def adaptive_simpsons(f, a, b, tol=1e-10, max_depth=50):
    """
    Adaptive Simpson's method.
    
    Recursively subdivides intervals where the error estimate is too large.
    Concentrates effort where the integrand is difficult.
    
    Error estimate: |S(a,b) - S(a,m) - S(m,b)| / 15  (from Richardson extrapolation)
    """
    def _simpson(f, a, b):
        """Simple Simpson's rule on [a, b]."""
        m = (a + b) / 2
        h = (b - a) / 6
        return h * (f(a) + 4 * f(m) + f(b))

    def _recursive(f, a, b, S, tol, depth):
        m = (a + b) / 2
        S_left = _simpson(f, a, m)
        S_right = _simpson(f, m, b)
        error = (S_left + S_right - S) / 15.0

        if abs(error) < tol or depth >= max_depth:
            return S_left + S_right + error  # Richardson correction
        else:
            return (_recursive(f, a, m, S_left, tol/2, depth+1) +
                    _recursive(f, m, b, S_right, tol/2, depth+1))

    S = _simpson(f, a, b)
    return _recursive(f, a, b, S, tol, 0)


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SIMPSON'S RULE DEMO")
    print("=" * 60)

    # --- Convergence comparison ---
    # TO TEST: Verify Simpson's O(h⁴) > Trapezoidal O(h²). Check error ratios: Simpson should have ~16x improvement.
    # Parameters: f=sin(x), n=[6,12,24,48,96] (must be even for Simpson 1/3), domain [0,π], exact=2.
    # Initial values: All three methods (Trapezoidal, Simpson 1/3, Simpson 3/8).
    # Observe: Error columns show Simpson 1/3 much smaller. Error ratio ≈ 16 for Simpson vs 4 for Trapez.
    # Try: Compare on f=x⁴ (Simpson should be exact!), or f=exp(x).
    print("\n--- ∫₀^π sin(x) dx = 2 ---")
    print(f"{'n':>6s} | {'Trapezoidal':>14s} | {'Simpson 1/3':>14s} | {'Simpson 3/8':>14s}")
    print("-" * 60)

    from trapezoidal_rule import trapezoidal

    for n in [6, 12, 24, 48, 96]:
        t = trapezoidal(np.sin, 0, np.pi, n)
        s13 = simpsons_rule(np.sin, 0, np.pi, n)
        s38 = simpsons_38_rule(np.sin, 0, np.pi, n)
        print(f"{n:6d} | {abs(t-2):14.4e} | {abs(s13-2):14.4e} | {abs(s38-2):14.4e}")

    # --- Order of accuracy ---
    # TO TEST: Verify O(h⁴) by computing convergence order = log₂(error_old/error_new). Should be ~4.
    # Parameters: f=exp(-x²), domain [0,1], exact=∫exp(-x²)dx, n=[4,8,16,32,64].
    # Initial values: Each n is doubled; with O(h⁴), error should drop by factor 16 each time.
    # Observe: Order column should show values ~4 as n gets large (asymptotic order).
    # Try: f=cos(x), f=sqrt(1+x²), or f=tan(x) to verify 4th-order holds for various smooth functions.
    print("\n--- Order of accuracy verification ---")
    f = lambda x: np.exp(-x**2)
    exact = 0.7468241328124271  # ∫₀^1 exp(-x²) dx
    
    prev_err = None
    for n in [4, 8, 16, 32, 64]:
        err = abs(simpsons_rule(f, 0, 1, n) - exact)
        order = np.log2(prev_err / err) if prev_err else float('nan')
        print(f"  n={n:3d}  error={err:.4e}  order≈{order:.2f}")
        prev_err = err

    # --- Adaptive Simpson's ---
    print("\n--- Adaptive Simpson's ---")
    # Function with a spike
    f_spike = lambda x: 1.0 / (1 + 100 * (x - 0.5)**2)  # Narrow peak at x=0.5
    exact_spike = np.arctan(10) / 5  # ≈ 0.29516...

    result = adaptive_simpsons(f_spike, 0, 1, tol=1e-12)
    print(f"  ∫ 1/(1+100(x-0.5)²) dx = {result:.12f}")
    print(f"  Exact:                    {exact_spike:.12f}")
    print(f"  Error:                    {abs(result - exact_spike):.2e}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Convergence
        ns = [2*k for k in range(1, 65)]
        err_trap = [abs(trapezoidal(np.sin, 0, np.pi, n) - 2) for n in ns]
        err_simp = [abs(simpsons_rule(np.sin, 0, np.pi, n) - 2) for n in ns]

        axes[0].loglog(ns, err_trap, 'b-', label='Trapezoidal O(h²)')
        axes[0].loglog(ns, err_simp, 'r-', label="Simpson's O(h⁴)")
        axes[0].set_xlabel('n (subintervals)')
        axes[0].set_ylabel('Absolute error')
        axes[0].set_title('Convergence Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # The spike function
        x = np.linspace(0, 1, 500)
        axes[1].plot(x, f_spike(x), 'b-', linewidth=1.5)
        axes[1].fill_between(x, 0, f_spike(x), alpha=0.2)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('f(x)')
        axes[1].set_title('Peaked Function (Adaptive Simpson works well)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("simpsons_rule.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
