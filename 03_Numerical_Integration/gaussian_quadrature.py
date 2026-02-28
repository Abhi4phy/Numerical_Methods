"""
Gaussian Quadrature
====================
Chooses BOTH the weights and the nodes optimally to maximize accuracy.

Key insight: With n points, Newton-Cotes is exact for polynomials of
degree n-1. But if we also optimize the node positions, we can achieve
exactness for polynomials of degree 2n-1. This is Gaussian quadrature.

∫₋₁¹ f(x) dx ≈ Σᵢ wᵢ f(xᵢ)

The nodes xᵢ are roots of Legendre polynomials.
The weights wᵢ are determined by the nodes.

Variants:
- Gauss-Legendre: standard, for smooth functions on [-1, 1]
- Gauss-Hermite: for ∫ f(x) e^{-x²} dx (quantum mechanics!)
- Gauss-Laguerre: for ∫₀^∞ f(x) e^{-x} dx
- Gauss-Chebyshev: for ∫ f(x)/√(1-x²) dx
"""

import numpy as np


def gauss_legendre_nodes_weights(n):
    """
    Compute Gauss-Legendre quadrature nodes and weights.
    
    The nodes are roots of the n-th Legendre polynomial P_n(x).
    We find them using the eigenvalue method:
    The nodes are eigenvalues of the companion (Jacobi) matrix.
    
    Parameters
    ----------
    n : int – number of quadrature points
    
    Returns
    -------
    x : ndarray – nodes in [-1, 1]
    w : ndarray – weights
    """
    # Construct the symmetric tridiagonal Jacobi matrix
    # β_i = i / √(4i² - 1)
    i = np.arange(1, n)
    beta = i / np.sqrt(4.0 * i**2 - 1.0)

    # Eigenvalues = nodes, eigenvectors determine weights
    J = np.diag(beta, -1) + np.diag(beta, 1)
    nodes, eigvecs = np.linalg.eigh(J)

    # Weights: w_i = 2 * (first component of eigenvector)²
    weights = 2.0 * eigvecs[0, :] ** 2

    return nodes, weights


def gauss_quadrature(f, a, b, n):
    """
    Gaussian quadrature on arbitrary interval [a, b].
    
    Transform from [-1, 1] to [a, b]:
        x = (b-a)/2 * t + (b+a)/2
        dx = (b-a)/2 * dt
    
    Parameters
    ----------
    f : callable
    a, b : float – integration limits
    n : int – number of quadrature points
    
    Returns
    -------
    result : float
    """
    nodes, weights = gauss_legendre_nodes_weights(n)

    # Map to [a, b]
    x = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    result = 0.5 * (b - a) * np.sum(weights * f(x))

    return result


def gauss_hermite_nodes_weights(n):
    """
    Gauss-Hermite quadrature for ∫_{-∞}^{∞} f(x) e^{-x²} dx.
    
    Crucial in quantum mechanics (harmonic oscillator wave functions),
    statistical mechanics (Gaussian integrals).
    
    Nodes are roots of Hermite polynomials H_n(x).
    """
    i = np.arange(1, n)
    beta = np.sqrt(i / 2.0)

    J = np.diag(beta, -1) + np.diag(beta, 1)
    nodes, eigvecs = np.linalg.eigh(J)
    weights = np.sqrt(np.pi) * eigvecs[0, :] ** 2

    return nodes, weights


def composite_gauss(f, a, b, n_elements, n_points):
    """
    Composite Gaussian quadrature: divide [a,b] into sub-intervals
    and apply Gauss quadrature on each.
    
    Combines the high accuracy of Gauss quadrature with the flexibility
    of adaptive methods.
    """
    x_edges = np.linspace(a, b, n_elements + 1)
    result = 0.0
    for i in range(n_elements):
        result += gauss_quadrature(f, x_edges[i], x_edges[i + 1], n_points)
    return result


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GAUSSIAN QUADRATURE DEMO")
    print("=" * 60)

    # --- Exactness for polynomials ---
    print("\n--- Exactness test: ∫₋₁¹ x^k dx ---")
    for k in range(8):
        f = lambda x, k=k: x**k
        exact = (1 - (-1)**(k+1)) / (k + 1)
        for n in [2, 3, 4]:
            result = gauss_quadrature(f, -1, 1, n)
            err = abs(result - exact)
            status = "✓" if err < 1e-14 else "✗"
            if n == 2:
                print(f"  x^{k}: n=2 err={err:.1e} {status}", end="")
            else:
                print(f"  n={n} err={err:.1e} {status}", end="")
        print()

    # --- Comparison with Simpson's ---
    print("\n--- Accuracy comparison: ∫₀^π sin(x) dx = 2 ---")
    from simpsons_rule import simpsons_rule
    
    print(f"{'Points':>6s} | {'Gauss':>14s} | {'Simpson':>14s}")
    print("-" * 42)
    for n in [2, 3, 4, 5, 8, 10]:
        g = gauss_quadrature(np.sin, 0, np.pi, n)
        s = simpsons_rule(np.sin, 0, np.pi, 2*n)  # Simpson uses 2n+1 points
        print(f"{n:6d} | {abs(g-2):14.4e} | {abs(s-2):14.4e}")

    # --- Gauss-Hermite: Gaussian integral ---
    print("\n--- Gauss-Hermite: ∫ x² e^{-x²} dx = √π/2 ---")
    exact_hermite = np.sqrt(np.pi) / 2
    for n in [2, 3, 4, 5]:
        nodes, weights = gauss_hermite_nodes_weights(n)
        result = np.sum(weights * nodes**2)
        print(f"  n={n}: {result:.12f}  error={abs(result - exact_hermite):.2e}")

    # --- Challenging integral ---
    print("\n--- ∫₀¹ √x dx = 2/3 (singular derivative at x=0) ---")
    exact = 2/3
    for n in [4, 8, 16, 32]:
        result = gauss_quadrature(np.sqrt, 0, 1, n)
        print(f"  n={n:3d}: error = {abs(result - exact):.4e}")

    # Plot nodes and weights
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Gauss-Legendre nodes for different n
        for n in [2, 4, 6, 8]:
            nodes, weights = gauss_legendre_nodes_weights(n)
            axes[0].scatter(nodes, [n]*len(nodes), s=weights*200, c='b', alpha=0.7)
        axes[0].set_xlabel('Nodes')
        axes[0].set_ylabel('n (number of points)')
        axes[0].set_title('Gauss-Legendre Nodes (size ∝ weight)')
        axes[0].set_xlim(-1.1, 1.1)
        axes[0].grid(True, alpha=0.3)

        # Convergence comparison
        ns = range(2, 30)
        errors_gauss = [abs(gauss_quadrature(np.sin, 0, np.pi, n) - 2) for n in ns]
        
        axes[1].semilogy(ns, errors_gauss, 'ro-', label='Gauss-Legendre', markersize=4)
        axes[1].set_xlabel('n (quadrature points)')
        axes[1].set_ylabel('|Error|')
        axes[1].set_title('Gauss Quadrature Convergence')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("gaussian_quadrature.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
