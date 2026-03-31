"""
Trapezoidal Rule
=================
Approximates ∫_a^b f(x) dx by connecting points with straight lines
and summing the areas of the resulting trapezoids.

Formula (composite):
    ∫_a^b f(x) dx ≈ h/2 [f(x₀) + 2f(x₁) + 2f(x₂) + ... + 2f(x_{n-1}) + f(x_n)]

Properties:
- Second-order accurate: error = O(h²).
- Error term: -(b-a) h² f''(ξ) / 12 for some ξ ∈ [a,b].
- Surprisingly accurate for periodic functions on full periods (exponential!).
- Foundation for Romberg integration.
"""

import numpy as np


def trapezoidal(f, a, b, n):
    """
    Composite Trapezoidal Rule.
    
    Parameters
    ----------
    f : callable – integrand
    a, b : float – integration limits
    n : int – number of subintervals
    
    Returns
    -------
    result : float – approximate integral
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    result = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return result


def trapezoidal_unequal(x, y):
    """
    Trapezoidal rule for unevenly spaced data points.
    
    ∫ ≈ Σ (x_{i+1} - x_i) * (y_{i+1} + y_i) / 2
    
    Equivalent to np.trapz(y, x).
    """
    result = 0.0
    for i in range(len(x) - 1):
        result += 0.5 * (x[i+1] - x[i]) * (y[i+1] + y[i])
    return result


def romberg_integration(f, a, b, max_order=10, tol=1e-12):
    """
    Romberg Integration: Richardson extrapolation applied to trapezoidal rule.
    
    Creates a triangular table of increasingly accurate estimates:
    T(h), T(h/2), T(h/4), ...
    and then extrapolates to eliminate error terms systematically.
    
    This achieves high-order accuracy using only the trapezoidal rule!
    
    Returns
    -------
    result : float
    table : ndarray – Romberg table (for inspection)
    """
    R = np.zeros((max_order, max_order))
    
    # First column: trapezoidal rule with 1, 2, 4, 8, ... subintervals
    for k in range(max_order):
        n = 2**k
        R[k, 0] = trapezoidal(f, a, b, n)
        
        # Richardson extrapolation
        for j in range(1, k + 1):
            R[k, j] = (4**j * R[k, j-1] - R[k-1, j-1]) / (4**j - 1)
        
        # Check convergence
        if k > 0 and abs(R[k, k] - R[k-1, k-1]) < tol:
            print(f"Romberg converged at order {k} ({2**k} panels)")
            return R[k, k], R[:k+1, :k+1]
    
    return R[-1, -1], R


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TRAPEZOIDAL RULE DEMO")
    print("=" * 60)

    # --- Test function: ∫₀^π sin(x) dx = 2 ---
    # TO TEST: Verify convergence rate O(h²). Try n=[2,4,8,16,32,64,128] and watch error ratios.
    # Parameters: integrand f=sin(x), domain [0,π], exact value=2, n (number of intervals).
    # Initial values: n=[2,4,8,...,128] (each n=4*previous n).
    # Observe: Error ratio consecutive runs should be ~4 (quadratic convergence). Ratio = errors[i-1]/errors[i].
    # Try: f=exp(x), f=cos(x), or f=1/(1+x²) on different domains to verify O(h²) holds generally.
    print("\n--- ∫₀^π sin(x) dx = 2 ---")
    exact = 2.0
    print(f"{'n':>6s} | {'Trapezoidal':>14s} | {'Error':>12s} | {'Ratio':>8s}")
    print("-" * 50)

    prev_error = None
    for n in [2, 4, 8, 16, 32, 64, 128]:
        result = trapezoidal(np.sin, 0, np.pi, n)
        error = abs(result - exact)
        ratio = prev_error / error if prev_error else float('nan')
        print(f"{n:6d} | {result:14.10f} | {error:12.4e} | {ratio:8.2f}")
        prev_error = error

    # --- Romberg integration ---
    # TO TEST: Compare final result accuracy vs explicit computation. Try max_order=[4, 6, 8, 10].
    # Parameters: f=sin(x), domain [0,π], exact=2, max_order=default, tol=1e-12.
    # Initial values: Uses trapezoidal rule internally with Richardson extrapolation.
    # Observe: Diagonal of Romberg table shows rapid convergence (better than pure trapezoidal).
    # Insight: Romberg achieves high order (4,6,8th) combining multiple resolutions. Check the table!
    print("\n--- Romberg Integration ---")
    result, table = romberg_integration(np.sin, 0, np.pi)
    print(f"Result: {result:.15f}")
    print(f"Exact:  {exact:.15f}")
    print(f"Error:  {abs(result - exact):.2e}")

    print("\nRomberg Table (diagonal = best estimates):")
    for i in range(min(table.shape[0], 6)):
        row = " ".join(f"{table[i,j]:12.10f}" for j in range(i+1))
        print(f"  {row}")

    # --- Periodic function: exponential accuracy ---
    # TO TEST: Compare periodic vs non-periodic convergence. Try periodic functions vs polynomial.
    # Parameters: f=exp(sin(x)) (2π-periodic and smooth), domain [0,2π], n=[4,8,16,32].
    # Initial values: Periodic functions enjoy exponential convergence (superconvergence!).
    # Observe: Error diminishes much faster than polynomial functions. n=32 gives tiny error!
    # Try: f=1/(2-cos(x)) or f=exp(-sin²(x)), both periodic. Compare errors to nonperiodic f=1/(1+x²).
    print("\n--- Periodic: ∫₀^{2π} exp(sin(x)) dx ---")
    from scipy import integrate as sp_integrate
    exact_periodic = sp_integrate.quad(lambda x: np.exp(np.sin(x)), 0, 2*np.pi)[0]
    
    for n in [4, 8, 16, 32]:
        result = trapezoidal(lambda x: np.exp(np.sin(x)), 0, 2*np.pi, n)
        error = abs(result - exact_periodic)
        print(f"  n = {n:3d}  |  Error = {error:.4e}")

    # Plot convergence
    try:
        import matplotlib.pyplot as plt
        ns = [2**k for k in range(1, 12)]
        errors_sin = [abs(trapezoidal(np.sin, 0, np.pi, n) - 2.0) for n in ns]
        errors_periodic = [abs(trapezoidal(lambda x: np.exp(np.sin(x)), 
                               0, 2*np.pi, n) - exact_periodic) for n in ns]

        plt.figure(figsize=(8, 5))
        plt.loglog(ns, errors_sin, 'bo-', label='sin(x) on [0,π] (O(h²))')
        plt.loglog(ns, errors_periodic, 'rs-', label='exp(sin(x)) on [0,2π] (exponential)')
        plt.loglog(ns, [errors_sin[0]*(ns[0]/n)**2 for n in ns], 'b--', alpha=0.5, label='O(h²) reference')
        plt.xlabel('Number of subintervals')
        plt.ylabel('Absolute error')
        plt.title('Trapezoidal Rule Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("trapezoidal_rule.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
