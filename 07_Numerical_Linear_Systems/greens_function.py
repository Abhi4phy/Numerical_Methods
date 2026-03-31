"""
Green's Function Methods
========================
Solve inhomogeneous linear differential equations:

    L[u](x) = f(x)

where L is a linear differential operator.

The Green's function G(x, x') satisfies:

    L[G](x, x') = δ(x - x')

Then the solution is:

    u(x) = ∫ G(x, x') f(x') dx'

This converts a differential equation into an integral equation.

Key properties:
- G encodes the response of the system to a point source.
- Symmetry: G(x,x') = G(x',x) for self-adjoint operators.
- Boundary conditions are built into G.

Examples implemented:
1. 1D Poisson equation: -u'' = f(x), with various BCs.
2. 1D Helmholtz equation: -u'' + k²u = f(x).
3. Heat kernel (Green's function for diffusion).

Physics: electrostatics (Coulomb potential), quantum mechanics
(propagator), elasticity, wave propagation.
"""

import numpy as np


def greens_function_1d_poisson(x, x_prime, L=1.0):
    """
    Green's function for -u'' = f on [0, L] with u(0)=u(L)=0.
    
    G(x, x') = { x'(L-x)/(L)  if x' ≤ x
               { x(L-x')/(L)  if x' > x
    """
    x = np.atleast_1d(x)
    x_prime = np.atleast_1d(x_prime)
    
    G = np.zeros((len(x), len(x_prime)))
    for i, xi in enumerate(x):
        for j, xp in enumerate(x_prime):
            if xp <= xi:
                G[i, j] = xp * (L - xi) / L
            else:
                G[i, j] = xi * (L - xp) / L
    return G


def solve_poisson_greens(f, x, L=1.0):
    """
    Solve -u'' = f(x) on [0, L], u(0)=u(L)=0
    using the Green's function method.
    
    u(x_i) = ∫₀ᴸ G(x_i, x') f(x') dx'
    
    The integral is approximated by the trapezoidal rule.
    """
    N = len(x)
    dx = x[1] - x[0]
    
    # Build Green's function matrix
    G = greens_function_1d_poisson(x, x, L)
    
    # Source term
    f_vals = np.array([f(xi) for xi in x])
    
    # Quadrature: u(x_i) ≈ Σ G(x_i, x_j) f(x_j) dx
    u = G @ f_vals * dx
    
    return u


def greens_function_1d_helmholtz(x, x_prime, k, L=1.0):
    """
    Green's function for -u'' + k²u = f on [0, L], u(0)=u(L)=0.
    
    G(x, x') = sinh(k·min(x,x')) sinh(k·(L-max(x,x'))) / (k sinh(kL))
    """
    x = np.atleast_1d(x)
    x_prime = np.atleast_1d(x_prime)
    
    G = np.zeros((len(x), len(x_prime)))
    denom = k * np.sinh(k * L)
    
    for i, xi in enumerate(x):
        for j, xp in enumerate(x_prime):
            x_min = min(xi, xp)
            x_max = max(xi, xp)
            G[i, j] = np.sinh(k * x_min) * np.sinh(k * (L - x_max)) / denom
    
    return G


def solve_helmholtz_greens(f, x, k, L=1.0):
    """
    Solve -u'' + k²u = f(x) using Green's function.
    """
    dx = x[1] - x[0]
    G = greens_function_1d_helmholtz(x, x, k, L)
    f_vals = np.array([f(xi) for xi in x])
    return G @ f_vals * dx


def heat_kernel(x, t, D=1.0):
    """
    Green's function for the heat equation on infinite domain.
    
    G(x, t) = 1/√(4πDt) · exp(-x²/(4Dt))
    
    This is the fundamental solution: if u(x,0) = δ(x),
    then u(x,t) = G(x,t).
    
    For general initial condition u(x,0) = u₀(x):
        u(x,t) = ∫ G(x-x', t) u₀(x') dx'
    """
    if t <= 0:
        raise ValueError("t must be positive")
    return np.exp(-x**2 / (4 * D * t)) / np.sqrt(4 * np.pi * D * t)


def solve_heat_greens(u0_func, x, t, D=1.0, dx_prime=None):
    """
    Solve the heat equation ∂u/∂t = D ∂²u/∂x² with u(x,0) = u₀(x)
    on an effectively infinite domain using the heat kernel.
    
    u(x, t) = ∫ G(x-x', t) u₀(x') dx'
    """
    if dx_prime is None:
        dx_prime = x[1] - x[0]
    
    # Extend integration domain
    x_prime = np.linspace(x[0] - 5*np.sqrt(2*D*t), x[-1] + 5*np.sqrt(2*D*t), 1000)
    dx_p = x_prime[1] - x_prime[0]
    
    u = np.zeros_like(x)
    u0_vals = np.array([u0_func(xp) for xp in x_prime])
    
    for i, xi in enumerate(x):
        G = heat_kernel(xi - x_prime, t, D)
        u[i] = np.sum(G * u0_vals) * dx_p
    
    return u


def coulomb_potential_2d(x, y, charges, positions):
    """
    2D electrostatic potential from point charges.
    
    V(r) = Σ q_i / (2π) · ln(1/|r - r_i|)
    
    This is the Green's function for the 2D Laplacian.
    """
    V = np.zeros_like(x)
    for q, (xq, yq) in zip(charges, positions):
        r = np.sqrt((x - xq)**2 + (y - yq)**2)
        r = np.maximum(r, 1e-10)  # Avoid log(0)
        V += -q / (2 * np.pi) * np.log(r)
    return V


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GREEN'S FUNCTION METHODS DEMO")
    print("=" * 60)

    # --- Problem 1: Poisson equation ---
    # TO TEST: Change source f_poisson (for example sin(2*pi*x)) or grid size N, and observe max error against the corresponding analytical solution.
    print("\n--- Problem 1: -u'' = sin(πx), u(0)=u(1)=0 ---")
    L = 1.0
    N = 100
    x = np.linspace(0, L, N)
    
    f_poisson = lambda x: np.sin(np.pi * x)
    u_greens = solve_poisson_greens(f_poisson, x, L)
    u_exact = np.sin(np.pi * x) / np.pi**2
    
    err = np.max(np.abs(u_greens - u_exact))
    print(f"Max error vs exact: {err:.6e}")
    print(f"u(0.5) computed: {np.interp(0.5, x, u_greens):.8f}")
    print(f"u(0.5) exact:    {1/np.pi**2:.8f}")

    # --- Problem 2: Helmholtz equation ---
    # TO TEST: Vary k (for example 1.0, 3.0) and compare u_helm to u_helm_exact to observe stiffness and error changes.
    print("\n--- Problem 2: -u'' + 4u = 1, u(0)=u(1)=0 ---")
    k = 2.0
    f_const = lambda x: 1.0
    u_helm = solve_helmholtz_greens(f_const, x, k, L)
    
    # Exact: u = 1/k² (1 - cosh(k(x-L/2))/cosh(kL/2))
    u_helm_exact = (1/k**2) * (1 - np.cosh(k*(x - L/2)) / np.cosh(k*L/2))
    err_helm = np.max(np.abs(u_helm - u_helm_exact))
    print(f"Max error: {err_helm:.6e}")

    # --- Problem 3: Heat equation ---
    # TO TEST: Change diffusion D and times list, then observe peak decay and spread width against the Gaussian closed form.
    print("\n--- Problem 3: Heat equation, Gaussian initial condition ---")
    D = 0.1
    x_heat = np.linspace(-5, 5, 200)
    u0_func = lambda x: np.exp(-x**2)
    
    times = [0.1, 0.5, 1.0, 2.0]
    for t in times:
        u_t = solve_heat_greens(u0_func, x_heat, t, D)
        # Exact for Gaussian IC: u(x,t) = 1/√(1+4Dt) exp(-x²/(1+4Dt))
        u_exact_t = np.exp(-x_heat**2 / (1 + 4*D*t)) / np.sqrt(1 + 4*D*t)
        err_t = np.max(np.abs(u_t - u_exact_t))
        print(f"  t={t:.1f}: max error = {err_t:.6e}, peak = {np.max(u_t):.4f}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Poisson solution
        axes[0, 0].plot(x, u_greens, 'b-', linewidth=2, label="Green's function")
        axes[0, 0].plot(x, u_exact, 'r--', linewidth=1.5, label='Exact')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('u(x)')
        axes[0, 0].set_title("-u'' = sin(πx)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Green's function visualization
        x_gf = np.linspace(0.01, 0.99, 100)
        for xp in [0.25, 0.5, 0.75]:
            G_vals = greens_function_1d_poisson(x_gf, [xp], L).flatten()
            axes[0, 1].plot(x_gf, G_vals, label=f"x' = {xp}")
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel("G(x, x')")
        axes[0, 1].set_title("Poisson Green's Function")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Heat equation diffusion
        colors = ['b', 'g', 'orange', 'r']
        axes[1, 0].plot(x_heat, u0_func(x_heat), 'k-', linewidth=2, label='t=0')
        for t, c in zip(times, colors):
            u_t = solve_heat_greens(u0_func, x_heat, t, D)
            axes[1, 0].plot(x_heat, u_t, color=c, linewidth=1.5, label=f't={t}')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('u(x, t)')
        axes[1, 0].set_title('Heat Equation: Gaussian Diffusion')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 2D Coulomb potential
        xg = np.linspace(-3, 3, 200)
        yg = np.linspace(-3, 3, 200)
        X, Y = np.meshgrid(xg, yg)
        
        charges = [1, -1, 0.5]
        positions = [(1, 0), (-1, 0), (0, 1.5)]
        V = coulomb_potential_2d(X, Y, charges, positions)
        V = np.clip(V, -2, 2)
        
        c = axes[1, 1].contourf(X, Y, V, levels=30, cmap='RdBu_r')
        plt.colorbar(c, ax=axes[1, 1], label='V')
        for q, (xq, yq) in zip(charges, positions):
            marker = 'r+' if q > 0 else 'b_'
            axes[1, 1].plot(xq, yq, marker, markersize=15, markeredgewidth=3)
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        axes[1, 1].set_title('2D Coulomb Potential')
        axes[1, 1].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig("greens_function.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
