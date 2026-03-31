"""
Multigrid Methods
==================
Hierarchical solvers for large-scale linear systems Ax = b arising
from discretized PDEs.

**The Problem with Standard Iterative Methods:**
Jacobi/Gauss-Seidel quickly reduce high-frequency error components
but are very slow at reducing low-frequency (smooth) errors.

**Multigrid Insight:**
Smooth errors on a fine grid become oscillatory on a coarser grid,
where iterative methods can eliminate them efficiently.

**V-Cycle Algorithm:**
1. Pre-smooth on fine grid (few Gauss-Seidel sweeps)
2. Restrict residual to coarser grid
3. Solve (or recurse) on coarse grid
4. Prolongate (interpolate) correction back to fine grid
5. Post-smooth on fine grid

**Complexity:** O(N) for one V-cycle — optimal!
Compare: direct solvers O(N^{3/2}) for 2D, iterative O(N^2/log N).

**Grid transfer operators:**
- Restriction: fine → coarse (full weighting: [1,2,1]/4)
- Prolongation: coarse → fine (linear interpolation)

Applications: CFD, structural mechanics, electromagnetics, any PDE
discretized on large grids.
"""

import numpy as np


def restrict_fw(r_fine):
    """
    Full-weighting restriction: fine → coarse.
    
    r_coarse[i] = (r_fine[2i-1] + 2·r_fine[2i] + r_fine[2i+1]) / 4
    
    Reduces grid by factor 2.
    """
    n_fine = len(r_fine)
    n_coarse = (n_fine - 1) // 2 + 1
    r_coarse = np.zeros(n_coarse)
    
    r_coarse[0] = r_fine[0]
    r_coarse[-1] = r_fine[-1]
    
    for i in range(1, n_coarse - 1):
        r_coarse[i] = 0.25 * (r_fine[2*i - 1] + 2*r_fine[2*i] + r_fine[2*i + 1])
    
    return r_coarse


def prolongate(e_coarse):
    """
    Linear interpolation prolongation: coarse → fine.
    
    e_fine[2i]   = e_coarse[i]
    e_fine[2i+1] = (e_coarse[i] + e_coarse[i+1]) / 2
    """
    n_coarse = len(e_coarse)
    n_fine = 2 * (n_coarse - 1) + 1
    e_fine = np.zeros(n_fine)
    
    for i in range(n_coarse):
        e_fine[2*i] = e_coarse[i]
    
    for i in range(n_coarse - 1):
        e_fine[2*i + 1] = 0.5 * (e_coarse[i] + e_coarse[i+1])
    
    return e_fine


def gauss_seidel_smooth(u, f, h, nu):
    """
    ν sweeps of Gauss-Seidel for -u'' = f on uniform grid.
    
    Stencil: (-u[i-1] + 2u[i] - u[i+1]) / h² = f[i]
    → u[i] = 0.5 * (u[i-1] + u[i+1] + h² f[i])
    """
    n = len(u)
    for _ in range(nu):
        for i in range(1, n - 1):
            u[i] = 0.5 * (u[i-1] + u[i+1] + h**2 * f[i])
    return u


def compute_residual(u, f, h):
    """
    Compute residual r = f - Au for the 1D Poisson operator.
    
    r[i] = f[i] - (-u[i-1] + 2u[i] - u[i+1])/h²
    """
    n = len(u)
    r = np.zeros(n)
    for i in range(1, n - 1):
        r[i] = f[i] - (-u[i-1] + 2*u[i] - u[i+1]) / h**2
    return r


def v_cycle(u, f, h, nu1=2, nu2=2, level=0, max_levels=10):
    """
    Multigrid V-cycle for 1D Poisson equation -u'' = f.
    
    Parameters
    ----------
    u : current approximation
    f : right-hand side
    h : grid spacing
    nu1 : pre-smoothing sweeps
    nu2 : post-smoothing sweeps
    level : current level (0 = finest)
    max_levels : maximum recursion depth
    
    Returns
    -------
    u : improved approximation
    """
    n = len(u)
    
    # Coarsest level: solve directly
    if n <= 3 or level >= max_levels:
        # Direct solve on coarse grid
        u = gauss_seidel_smooth(u, f, h, 50)
        return u
    
    # 1. Pre-smooth
    u = gauss_seidel_smooth(u, f, h, nu1)
    
    # 2. Compute residual
    r = compute_residual(u, f, h)
    
    # 3. Restrict residual to coarse grid
    r_coarse = restrict_fw(r)
    
    # 4. Solve on coarse grid (recursive V-cycle)
    n_coarse = len(r_coarse)
    e_coarse = np.zeros(n_coarse)
    h_coarse = 2 * h
    e_coarse = v_cycle(e_coarse, r_coarse, h_coarse, nu1, nu2, level+1, max_levels)
    
    # 5. Prolongate correction and add to solution
    e_fine = prolongate(e_coarse)
    u += e_fine
    
    # 6. Post-smooth
    u = gauss_seidel_smooth(u, f, h, nu2)
    
    return u


def full_multigrid(f, h, nu1=2, nu2=2, n_vcycles=1, max_levels=10):
    """
    Full Multigrid (FMG) - uses nested iteration for a better initial guess.
    
    1. Start on the coarsest grid.
    2. Solve (approximately).
    3. Prolongate to next finer grid.
    4. Perform V-cycles.
    5. Repeat until finest grid.
    
    This achieves the solution in O(N) work to discretization accuracy!
    """
    n = len(f)
    
    # Build hierarchy of grid sizes
    grids = []
    n_curr = n
    h_curr = h
    while n_curr >= 3:
        grids.append((n_curr, h_curr))
        n_curr = (n_curr - 1) // 2 + 1
        h_curr *= 2
    
    # Start from coarsest
    n_c, h_c = grids[-1]
    f_levels = [f]
    f_curr = f
    for i in range(len(grids) - 1):
        f_curr = restrict_fw(f_curr)
        f_levels.append(f_curr)
    
    # Solve on coarsest
    u = np.zeros(grids[-1][0])
    u = gauss_seidel_smooth(u, f_levels[-1], grids[-1][1], 50)
    
    # Work up through levels
    for lev in range(len(grids) - 2, -1, -1):
        # Prolongate to finer grid
        u = prolongate(u)
        n_lev, h_lev = grids[lev]
        
        # Make sure sizes match
        if len(u) != n_lev:
            u_new = np.zeros(n_lev)
            u_new[:len(u)] = u
            u = u_new
        
        # V-cycles at this level
        for _ in range(n_vcycles):
            u = v_cycle(u, f_levels[lev], h_lev, nu1, nu2, lev, max_levels)
    
    return u


def multigrid_solve(f_func, N, L=1.0, tol=1e-10, max_cycles=50, nu1=2, nu2=2):
    """
    Complete multigrid solver for -u'' = f on [0, L], u(0)=u(L)=0.
    
    Parameters
    ----------
    f_func : callable or array — RHS
    N : int — number of grid points (should be 2^k + 1 for optimal MG)
    L : float — domain length
    tol : float — convergence tolerance
    max_cycles : int — maximum V-cycles
    
    Returns
    -------
    x : grid points
    u : solution
    residuals : list of residual norms
    """
    h = L / (N - 1)
    x = np.linspace(0, L, N)
    
    if callable(f_func):
        f = np.array([f_func(xi) for xi in x])
    else:
        f = np.array(f_func, dtype=float)
    
    f[0] = 0  # BC
    f[-1] = 0
    
    u = np.zeros(N)
    residuals = []
    
    for cycle in range(max_cycles):
        u = v_cycle(u.copy(), f, h, nu1, nu2)
        
        r = compute_residual(u, f, h)
        res_norm = np.linalg.norm(r) * h  # Weighted norm
        residuals.append(res_norm)
        
        if res_norm < tol:
            print(f"Multigrid converged in {cycle+1} V-cycles (res = {res_norm:.2e})")
            return x, u, residuals
    
    print(f"Multigrid: {max_cycles} V-cycles, final residual = {residuals[-1]:.2e}")
    return x, u, residuals


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MULTIGRID METHOD DEMO")
    print("=" * 60)

    # --- Problem 1: Simple Poisson ---
    # TO TEST: Change N (use 2^k+1 values) and smoothing sweeps nu1/nu2, then observe V-cycle count and max error versus exact sin(pi*x).
    print("\n--- Problem 1: -u'' = sin(πx), u(0)=u(1)=0 ---")
    
    N = 129  # 2^7 + 1
    f_func = lambda x: np.pi**2 * np.sin(np.pi * x)  # So exact u = sin(πx)
    
    x, u_mg, residuals = multigrid_solve(f_func, N, tol=1e-10)
    u_exact = np.sin(np.pi * x)
    err = np.max(np.abs(u_mg - u_exact))
    print(f"Max error vs exact: {err:.6e}")
    print(f"V-cycles needed: {len(residuals)}")

    # --- Comparison with Gauss-Seidel alone ---
    # TO TEST: Increase/decrease GS sweeps and compare residual trajectories to multigrid to observe low-frequency error reduction advantage.
    print("\n--- Comparison: Multigrid vs Gauss-Seidel ---")
    h = 1.0 / (N - 1)
    f = np.array([f_func(xi) for xi in x])
    f[0] = 0; f[-1] = 0
    
    u_gs = np.zeros(N)
    gs_residuals = []
    
    for sweep in range(500):
        u_gs = gauss_seidel_smooth(u_gs.copy(), f, h, 1)
        r = compute_residual(u_gs, f, h)
        gs_residuals.append(np.linalg.norm(r) * h)
    
    print(f"After 500 GS sweeps:  residual = {gs_residuals[-1]:.6e}")
    print(f"After {len(residuals)} MG V-cycles: residual = {residuals[-1]:.6e}")
    
    # Count total work (GS sweeps)
    total_gs_sweeps_in_mg = len(residuals) * (2 + 2) * 2  # Rough estimate
    print(f"MG total GS sweeps (est): ~{total_gs_sweeps_in_mg}")

    # --- Problem 2: Variable coefficient ---
    # TO TEST: Replace source with higher/lower frequency modes (for example sin(5*pi*x)), and observe impact on convergence cycles and error.
    print("\n--- Problem 2: Higher frequency source ---")
    f_func2 = lambda x: np.pi**2 * 9 * np.sin(3*np.pi*x)  # u = sin(3πx)
    
    x2, u_mg2, res2 = multigrid_solve(f_func2, N, tol=1e-10)
    u_exact2 = np.sin(3*np.pi * x2)
    err2 = np.max(np.abs(u_mg2 - u_exact2))
    print(f"Max error: {err2:.6e}, V-cycles: {len(res2)}")

    # --- Convergence rate ---
    print("\n--- Convergence rate analysis ---")
    for i in range(1, len(residuals)):
        if residuals[i-1] > 1e-15:
            ratio = residuals[i] / residuals[i-1]
            print(f"  Cycle {i}: res = {residuals[i]:.2e}, "
                  f"ratio = {ratio:.4f}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Solution
        axes[0, 0].plot(x, u_mg, 'b-', linewidth=2, label='Multigrid')
        axes[0, 0].plot(x, u_exact, 'r--', linewidth=1.5, label='Exact')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('u(x)')
        axes[0, 0].set_title("-u'' = π²sin(πx)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Convergence: MG vs GS
        axes[0, 1].semilogy(range(1, len(residuals)+1), residuals, 'b-o',
                           markersize=4, label=f'Multigrid ({len(residuals)} V-cycles)')
        axes[0, 1].semilogy(range(1, len(gs_residuals)+1), gs_residuals, 'r-',
                           alpha=0.7, label=f'Gauss-Seidel ({len(gs_residuals)} sweeps)')
        axes[0, 1].set_xlabel('Iteration/Cycle')
        axes[0, 1].set_ylabel('Residual norm')
        axes[0, 1].set_title('Convergence: Multigrid vs Gauss-Seidel')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error profile
        axes[1, 0].semilogy(x, np.abs(u_mg - u_exact) + 1e-16, 'b-')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('|error|')
        axes[1, 0].set_title('Pointwise Error')
        axes[1, 0].grid(True, alpha=0.3)
        
        # V-cycle diagram
        axes[1, 1].set_title('V-Cycle Structure')
        levels = 5
        # Draw V shape
        x_v = list(range(levels)) + list(range(levels-2, -1, -1))
        y_v = list(range(levels)) + list(range(levels-2, -1, -1))
        
        for i in range(len(x_v)):
            h_lev = 2**y_v[i]
            n_pts = 2**(levels - y_v[i])
            axes[1, 1].plot(i, y_v[i], 'bo', markersize=10)
            axes[1, 1].annotate(f'h={h_lev}h₀\nn={n_pts}', (i, y_v[i]),
                              textcoords="offset points", xytext=(0, 12),
                              ha='center', fontsize=7)
        
        axes[1, 1].plot(range(len(x_v)), y_v, 'b-', linewidth=1.5)
        
        # Annotate
        axes[1, 1].annotate('Pre-smooth', (0.5, 0.5), fontsize=8, color='green',
                           rotation=-50)
        axes[1, 1].annotate('Restrict', (1.5, 1.5), fontsize=8, color='red',
                           rotation=-50)
        axes[1, 1].annotate('Prolongate', (len(x_v)-2.5, 1.5), fontsize=8, 
                           color='purple', rotation=50)
        axes[1, 1].annotate('Post-smooth', (len(x_v)-1.5, 0.5), fontsize=8,
                           color='orange', rotation=50)
        
        axes[1, 1].set_xlabel('Operation sequence')
        axes[1, 1].set_ylabel('Grid level (coarser →)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("multigrid.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
