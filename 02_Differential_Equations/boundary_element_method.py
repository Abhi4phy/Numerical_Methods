"""
Boundary Element Method (BEM) — 1D/2D Introduction
=====================================================
Reformulates a PDE as an integral equation on the BOUNDARY only.

Key idea:
1. Use Green's function to convert the PDE into a boundary integral equation.
2. Discretize only the boundary (not the interior).
3. Solve the boundary integral equation.
4. Evaluate the solution anywhere in the interior using the boundary data.

Advantages:
- Reduces dimension by one (3D volume → 2D surface).
- Exact treatment of infinite domains.
- Very efficient for exterior problems (scattering, radiation).

Disadvantages:
- Requires knowledge of Green's function.
- Produces dense (not sparse) matrices.
- Less suitable for nonlinear or inhomogeneous problems.

Demo: 1D BEM for Laplace equation, 2D potential problem.
"""

import numpy as np


def greens_function_1d(x, xi):
    """
    Green's function for -u'' = δ(x - ξ) on the real line.
    G(x, ξ) = -|x - ξ| / 2
    """
    return -np.abs(x - xi) / 2.0


def bem_laplace_1d(f_func, x_range, n_boundary=2, n_interior=50):
    """
    Simplified 1D BEM for -u''(x) = f(x) on [a, b].
    
    In 1D, BEM is essentially using Green's function directly:
        u(x) = ∫_a^b G(x, ξ) f(ξ) dξ + boundary terms
    
    This demonstrates the fundamental concept: the solution at any
    interior point is determined by boundary values + source integral.
    
    Parameters
    ----------
    f_func : callable – source term
    x_range : tuple (a, b)
    n_boundary : int – not used in 1D (2 boundary points always)
    n_interior : int – points for evaluating solution
    
    Returns
    -------
    x : ndarray – evaluation points
    u : ndarray – solution
    """
    a, b = x_range

    # For 1D Poisson on [a,b] with u(a)=u(b)=0:
    # u(x) = ∫_a^b G_D(x, ξ) f(ξ) dξ
    # where G_D is the Dirichlet Green's function:
    # G_D(x, ξ) = (1/(b-a)) * { (x-a)(b-ξ) if x ≤ ξ
    #                           { (ξ-a)(b-x) if x > ξ

    x = np.linspace(a, b, n_interior + 2)  # Including boundary
    u = np.zeros(len(x))

    # Integration quadrature points
    n_quad = 200
    xi = np.linspace(a, b, n_quad)
    dxi = xi[1] - xi[0]

    for idx, x_pt in enumerate(x):
        if x_pt == a or x_pt == b:
            u[idx] = 0.0  # Dirichlet BC
            continue

        integrand = np.zeros(n_quad)
        for j, xi_pt in enumerate(xi):
            # Dirichlet Green's function
            if x_pt <= xi_pt:
                G = (x_pt - a) * (b - xi_pt) / (b - a)
            else:
                G = (xi_pt - a) * (b - x_pt) / (b - a)
            integrand[j] = G * f_func(xi_pt)

        # Trapezoidal integration
        u[idx] = np.trapz(integrand, xi)

    return x, u


def bem_potential_2d(charges, x_eval, y_eval):
    """
    2D BEM-like calculation: Potential from point charges using Green's function.
    
    Green's function for 2D Laplace: G(r, r') = -1/(2π) ln|r - r'|
    
    This demonstrates how BEM computes interior solutions from
    boundary/source data using Green's functions.
    
    Parameters
    ----------
    charges : list of (x, y, q) tuples – charge positions and strengths
    x_eval, y_eval : 2D mesh grids for evaluation points
    
    Returns
    -------
    phi : 2D potential field
    """
    phi = np.zeros_like(x_eval)
    for xc, yc, q in charges:
        r = np.sqrt((x_eval - xc)**2 + (y_eval - yc)**2)
        r = np.maximum(r, 1e-10)  # Avoid log(0)
        phi += -q / (2 * np.pi) * np.log(r)
    return phi


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BOUNDARY ELEMENT METHOD DEMO")
    print("=" * 60)

    # --- 1D BEM for Poisson equation ---
    print("\n--- 1D BEM: -u'' = π²sin(πx), u(0)=u(1)=0 ---")
    f = lambda x: np.pi**2 * np.sin(np.pi * x)
    u_exact = lambda x: np.sin(np.pi * x)

    x, u = bem_laplace_1d(f, (0, 1), n_interior=50)
    error = np.max(np.abs(u - u_exact(x)))
    print(f"  Max error: {error:.6e}")
    print(f"  u(0.5) computed: {u[26]:.6f}")
    print(f"  u(0.5) exact:    {u_exact(0.5):.6f}")

    # --- 2D Potential from charges ---
    print("\n--- 2D Potential from point charges ---")
    charges = [
        (0.3, 0.3, 1.0),   # Positive charge
        (-0.3, -0.3, -1.0), # Negative charge
    ]

    # Evaluation grid
    x_grid = np.linspace(-1, 1, 100)
    y_grid = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x_grid, y_grid)

    phi = bem_potential_2d(charges, X, Y)
    print(f"  Potential at origin: {phi[50, 50]:.6f}")
    print(f"  Max potential: {np.max(phi):.4f}")
    print(f"  Min potential: {np.min(phi):.4f}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1D BEM solution
        axes[0].plot(x, u, 'b-o', markersize=3, label='BEM')
        x_fine = np.linspace(0, 1, 200)
        axes[0].plot(x_fine, u_exact(x_fine), 'r--', label='Exact')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('u(x)')
        axes[0].set_title('1D BEM: Poisson Equation')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2D potential
        levels = np.linspace(np.percentile(phi, 5), np.percentile(phi, 95), 30)
        c = axes[1].contourf(X, Y, phi, levels=levels, cmap='RdBu_r')
        axes[1].contour(X, Y, phi, levels=levels, colors='k', linewidths=0.3)
        plt.colorbar(c, ax=axes[1], label='Potential φ')
        for xc, yc, q in charges:
            color = 'red' if q > 0 else 'blue'
            axes[1].plot(xc, yc, 'o', color=color, markersize=10)
        axes[1].set_xlim(-1, 1)
        axes[1].set_ylim(-1, 1)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_title('2D Potential (Dipole)')
        axes[1].set_aspect('equal')

        plt.tight_layout()
        plt.savefig("boundary_element_method.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
