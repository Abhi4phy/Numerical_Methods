"""
Finite Difference Method (FDM)
===============================
Approximates derivatives by finite differences on a discrete grid.

Core idea:
    f'(x) ≈ [f(x+h) - f(x-h)] / (2h)          (central difference, O(h²))
    f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²   (second derivative, O(h²))

Replace derivatives in a PDE with these approximations → get a system of
algebraic equations that can be solved.

Applications demonstrated:
1. 1D Poisson equation: -u''(x) = f(x)
2. 1D Heat equation (time-dependent): ∂u/∂t = α ∂²u/∂x²
3. 1D Wave equation: ∂²u/∂t² = c² ∂²u/∂x²
"""

import numpy as np


def solve_poisson_1d(f_func, x_range, n_points, bc_left=0.0, bc_right=0.0):
    """
    Solve -u''(x) = f(x) on [a, b] with Dirichlet BCs.
    
    Discretization:
        -[u_{i-1} - 2u_i + u_{i+1}] / h² = f_i
    
    This gives a tridiagonal system Au = b.
    
    Parameters
    ----------
    f_func : callable – source term f(x)
    x_range : tuple (a, b)
    n_points : int – number of interior points
    bc_left, bc_right : float – boundary values
    
    Returns
    -------
    x : grid points (including boundaries)
    u : solution
    """
    a, b = x_range
    h = (b - a) / (n_points + 1)
    x_interior = np.linspace(a + h, b - h, n_points)

    # Build tridiagonal matrix (-1, 2, -1) / h²
    main_diag = 2.0 * np.ones(n_points) / h**2
    off_diag = -1.0 * np.ones(n_points - 1) / h**2

    A = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

    # Right-hand side
    rhs = f_func(x_interior)
    rhs[0] += bc_left / h**2       # Boundary contribution
    rhs[-1] += bc_right / h**2

    # Solve
    u_interior = np.linalg.solve(A, rhs)

    # Assemble full solution with boundaries
    x = np.concatenate([[a], x_interior, [b]])
    u = np.concatenate([[bc_left], u_interior, [bc_right]])

    return x, u


def heat_equation_1d(alpha, L, T, nx, nt, u0_func, bc_left=0.0, bc_right=0.0):
    """
    Solve the 1D heat equation ∂u/∂t = α ∂²u/∂x² using FTCS
    (Forward Time, Central Space) explicit scheme.
    
    Stability condition (CFL): α Δt / Δx² ≤ 0.5
    
    Parameters
    ----------
    alpha : float – thermal diffusivity
    L : float – domain length [0, L]
    T : float – final time
    nx : int – number of spatial points
    nt : int – number of time steps
    u0_func : callable – initial condition u(x, 0)
    """
    dx = L / (nx - 1)
    dt = T / nt
    r = alpha * dt / dx**2  # CFL number

    print(f"CFL number r = α·dt/dx² = {r:.4f}", end="")
    if r > 0.5:
        print(" ⚠️ UNSTABLE (r > 0.5)!")
    else:
        print(" ✓ Stable")

    x = np.linspace(0, L, nx)
    u = u0_func(x)
    u[0] = bc_left
    u[-1] = bc_right

    # Store solution at selected times
    save_times = np.linspace(0, T, 6)
    save_idx = 0
    solutions = [(0, u.copy())]

    for n in range(1, nt + 1):
        u_new = u.copy()
        # FTCS: u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
        u_new[0] = bc_left
        u_new[-1] = bc_right
        u = u_new

        t = n * dt
        if save_idx < len(save_times) - 1 and t >= save_times[save_idx + 1]:
            save_idx += 1
            solutions.append((t, u.copy()))

    return x, solutions


def wave_equation_1d(c, L, T, nx, nt, u0_func, v0_func=None):
    """
    Solve the 1D wave equation ∂²u/∂t² = c² ∂²u/∂x² using
    central differences in both space and time.
    
    u_i^{n+1} = 2u_i^n - u_i^{n-1} + r²(u_{i+1}^n - 2u_i^n + u_{i-1}^n)
    
    Stability condition (CFL): c·dt/dx ≤ 1
    """
    dx = L / (nx - 1)
    dt = T / nt
    r = c * dt / dx

    print(f"CFL number r = c·dt/dx = {r:.4f}", end="")
    if r > 1.0:
        print(" ⚠️ UNSTABLE (r > 1)!")
    else:
        print(" ✓ Stable")

    x = np.linspace(0, L, nx)

    # Initial conditions
    u_prev = u0_func(x)                   # u at t=0
    if v0_func is None:
        v0 = np.zeros(nx)
    else:
        v0 = v0_func(x)

    # First time step (using initial velocity)
    u_curr = u_prev.copy()
    u_curr[1:-1] = (u_prev[1:-1]
                    + dt * v0[1:-1]
                    + 0.5 * r**2 * (u_prev[2:] - 2*u_prev[1:-1] + u_prev[:-2]))
    u_curr[0] = 0
    u_curr[-1] = 0

    solutions = [(0, u_prev.copy())]
    save_times = np.linspace(0, T, 8)
    save_idx = 0

    for n in range(2, nt + 1):
        u_next = np.zeros(nx)
        u_next[1:-1] = (2*u_curr[1:-1] - u_prev[1:-1]
                        + r**2 * (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]))
        u_next[0] = 0
        u_next[-1] = 0

        u_prev = u_curr.copy()
        u_curr = u_next.copy()

        t = n * dt
        if save_idx < len(save_times) - 1 and t >= save_times[save_idx + 1]:
            save_idx += 1
            solutions.append((t, u_curr.copy()))

    return x, solutions


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FINITE DIFFERENCE METHOD DEMO")
    print("=" * 60)

    # --- 1D Poisson equation ---
    # TO TEST: Vary spatial resolution n_points (25, 50, 100) and boundary conditions (bc_left, bc_right).
    # Parameters: n_points (interior grid), domain [0,1], f(x)=π²sin(πx), BCs: u(0)=u(1)=0.
    # Initial values: n_points=50 (exact: u(x)=sin(πx)).
    # Observe: Max error scales as O(h²) where h=1/(n_points+1). Doubling n_points → error/4.
    # Try: Change boundary values to bc_left=1, bc_right=-1, or modify source f to x or exp(x).
    print("\n--- Poisson Equation: -u'' = sin(πx), u(0)=u(1)=0 ---")
    f = lambda x: np.sin(np.pi * x) * np.pi**2
    u_exact = lambda x: np.sin(np.pi * x)  # Exact solution

    x, u = solve_poisson_1d(f, (0, 1), n_points=50)
    error = np.max(np.abs(u - u_exact(x)))
    print(f"Max error (n=50): {error:.6e}")

    # --- Heat equation ---
    # TO TEST: Vary thermal diffusivity (alpha: 0.001, 0.01, 0.1), spatial/temporal resolution (nx, nt), final time T.
    # Parameters: alpha (diffusivity), L (domain), T (final time), nx,nt (grid/time steps), u0 (initial profile).
    # Initial values: alpha=0.01, L=1, T=2, nx=50, nt=5000 (watch the CFL number r=alpha*dt/dx²).
    # Observe: Solution should diffuse smoothly. Smaller alpha → slower diffusion. Check CFL condition r≤0.5!
    # Try: Set alpha=0.1, nt=100 (r>0.5) to see unstability (solution blows up). Then reduce nt to 1000.
    print("\n--- Heat Equation: ∂u/∂t = 0.01 ∂²u/∂x² ---")
    u0 = lambda x: np.sin(np.pi * x)
    x_heat, sols_heat = heat_equation_1d(
        alpha=0.01, L=1.0, T=2.0, nx=50, nt=5000, u0_func=u0
    )

    # --- Wave equation ---
    # TO TEST: Vary wave speed c (0.5, 1.0, 2.0), initial pulse shape (Gaussian width, position), domain length L.
    # Parameters: c (wave speed), L (domain), T (time), nx,nt (spatial/temporal resolution), u0 (initial displacement).
    # Initial values: c=1, L=1, T=1, nx=200, nt=400 (Gaussian pulse at x=0.3, width~0.1).
    # Observe: Pulse should split into left/right traveling waves. Monitor CFL: r=c*dt/dx must be ≤1!
    # Try: c=2 (faster wave), or u0=lambda x: np.sin(4*pi*x) (sinusoidal initial), or L=2 (larger domain).
    print("\n--- Wave Equation: ∂²u/∂t² = ∂²u/∂x² ---")
    u0_wave = lambda x: np.exp(-100 * (x - 0.3)**2)  # Gaussian pulse
    x_wave, sols_wave = wave_equation_1d(
        c=1.0, L=1.0, T=1.0, nx=200, nt=400, u0_func=u0_wave
    )

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Poisson
        axes[0].plot(x, u, 'b-o', markersize=3, label='FDM')
        x_fine = np.linspace(0, 1, 200)
        axes[0].plot(x_fine, u_exact(x_fine), 'r--', label='Exact')
        axes[0].set_title("Poisson Equation")
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('u(x)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Heat equation
        for t, u_h in sols_heat:
            axes[1].plot(x_heat, u_h, label=f't={t:.2f}')
        axes[1].set_title("Heat Equation")
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('u(x,t)')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # Wave equation
        for t, u_w in sols_wave:
            axes[2].plot(x_wave, u_w, label=f't={t:.2f}')
        axes[2].set_title("Wave Equation")
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('u(x,t)')
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("finite_difference_method.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
