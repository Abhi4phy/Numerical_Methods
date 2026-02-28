"""
Lagrange Interpolation
=======================
Given n+1 data points (x₀,y₀), ..., (xₙ,yₙ), construct the unique
polynomial P(x) of degree ≤ n passing through all points.

Formula:
    P(x) = Σᵢ yᵢ Lᵢ(x)
    
    where Lᵢ(x) = Π_{j≠i} (x - xⱼ)/(xᵢ - xⱼ)  (Lagrange basis polynomials)

Properties:
- Unique polynomial of degree ≤ n through n+1 points.
- Does NOT require equally spaced points.
- Suffers from Runge's phenomenon with equidistant points.
- Adding a new point requires recomputing everything (Newton form fixes this).
- O(n²) to evaluate at one point.
"""

import numpy as np


def lagrange_basis(x_nodes, i, x):
    """
    Evaluate the i-th Lagrange basis polynomial Lᵢ(x).
    
    Lᵢ(x) = Π_{j≠i} (x - xⱼ)/(xᵢ - xⱼ)
    
    Parameters
    ----------
    x_nodes : ndarray – interpolation nodes
    i : int – index of basis polynomial
    x : float or ndarray – evaluation point(s)
    """
    n = len(x_nodes)
    L = np.ones_like(np.atleast_1d(x), dtype=float)
    
    for j in range(n):
        if j != i:
            L *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
    
    return L


def lagrange_interpolation(x_nodes, y_nodes, x):
    """
    Evaluate the Lagrange interpolating polynomial at x.
    
    Parameters
    ----------
    x_nodes : ndarray – interpolation nodes (n+1 points)
    y_nodes : ndarray – function values at nodes
    x : float or ndarray – evaluation point(s)
    
    Returns
    -------
    P(x) : float or ndarray
    """
    n = len(x_nodes)
    x = np.atleast_1d(np.array(x, dtype=float))
    result = np.zeros_like(x)
    
    for i in range(n):
        result += y_nodes[i] * lagrange_basis(x_nodes, i, x)
    
    return result.squeeze()


def lagrange_error_bound(f_deriv_n1, x_nodes, x):
    """
    Error bound for Lagrange interpolation.
    
    |f(x) - P_n(x)| ≤ |ω(x)| / (n+1)! * max|f^{(n+1)}|
    
    where ω(x) = Π(x - xᵢ) is the nodal polynomial.
    """
    n = len(x_nodes)
    omega = np.prod([x - xi for xi in x_nodes])
    M = f_deriv_n1  # max of (n+1)-th derivative
    bound = abs(omega) / np.math.factorial(n) * M
    return bound


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LAGRANGE INTERPOLATION DEMO")
    print("=" * 60)

    # --- Example 1: Interpolate sin(x) ---
    print("\n--- Interpolating sin(x) with n points ---")
    for n in [3, 5, 7, 10]:
        x_nodes = np.linspace(0, np.pi, n)
        y_nodes = np.sin(x_nodes)
        
        x_test = np.linspace(0, np.pi, 100)
        y_interp = np.array([lagrange_interpolation(x_nodes, y_nodes, xi) for xi in x_test])
        y_exact = np.sin(x_test)
        
        error = np.max(np.abs(y_interp - y_exact))
        print(f"  n = {n:2d}  |  Max error = {error:.6e}")

    # --- Runge's phenomenon ---
    print("\n--- Runge's Phenomenon: f(x) = 1/(1+25x²) ---")
    runge = lambda x: 1.0 / (1 + 25 * x**2)
    
    x_test = np.linspace(-1, 1, 200)
    for n in [5, 9, 13, 17, 21]:
        # Equally spaced nodes
        x_eq = np.linspace(-1, 1, n)
        y_eq = runge(x_eq)
        y_interp_eq = np.array([lagrange_interpolation(x_eq, y_eq, xi) for xi in x_test])
        err_eq = np.max(np.abs(y_interp_eq - runge(x_test)))

        # Chebyshev nodes (cure for Runge's phenomenon)
        x_cheb = np.cos(np.pi * np.arange(n) / (n - 1))
        y_cheb = runge(x_cheb)
        y_interp_cheb = np.array([lagrange_interpolation(x_cheb, y_cheb, xi) for xi in x_test])
        err_cheb = np.max(np.abs(y_interp_cheb - runge(x_test)))

        print(f"  n={n:2d}: equidistant err = {err_eq:10.4e}  |  "
              f"Chebyshev err = {err_cheb:10.4e}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Sin interpolation
        for n in [3, 5, 9]:
            x_nodes = np.linspace(0, np.pi, n)
            y_nodes = np.sin(x_nodes)
            y_interp = np.array([lagrange_interpolation(x_nodes, y_nodes, xi) for xi in x_test])
            axes[0].plot(np.linspace(0, np.pi, 200), y_interp, label=f'n={n}')
        axes[0].plot(np.linspace(0, np.pi, 200), np.sin(np.linspace(0, np.pi, 200)),
                     'k--', lw=2, label='sin(x)')
        axes[0].set_title('Lagrange Interpolation of sin(x)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Runge's phenomenon
        x_fine = np.linspace(-1, 1, 500)
        axes[1].plot(x_fine, runge(x_fine), 'k-', lw=2, label='f(x)')
        
        n = 15
        x_eq = np.linspace(-1, 1, n)
        y_eq_i = np.array([lagrange_interpolation(x_eq, runge(x_eq), xi) for xi in x_fine])
        axes[1].plot(x_fine, y_eq_i, 'r-', label=f'Equidistant n={n}')
        
        x_cheb = np.cos(np.pi * np.arange(n) / (n - 1))
        y_ch_i = np.array([lagrange_interpolation(x_cheb, runge(x_cheb), xi) for xi in x_fine])
        axes[1].plot(x_fine, y_ch_i, 'b--', label=f'Chebyshev n={n}')
        
        axes[1].set_ylim(-0.5, 1.5)
        axes[1].set_title("Runge's Phenomenon")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("lagrange_interpolation.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
