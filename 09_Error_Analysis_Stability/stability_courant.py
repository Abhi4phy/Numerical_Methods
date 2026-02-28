"""
Stability Analysis & CFL Condition
=====================================
Numerical stability determines whether errors grow or decay during
time integration of PDEs.

**Von Neumann (Fourier) Stability Analysis:**
Insert a Fourier mode e^{ikx} into the difference scheme.
The amplification factor G(k) satisfies:

    u^{n+1}_j = G(k) · u^n_j

Stability requires |G(k)| ≤ 1 for all k.

**CFL Condition (Courant–Friedrichs–Lewy):**
For the advection equation u_t + a u_x = 0:

    C = |a| Δt / Δx ≤ 1

where C is the Courant number. This means the numerical domain
of dependence must contain the physical domain of dependence.

**Key stability results:**
| Scheme           | Equation  | Stability condition         |
|------------------|-----------|-----------------------------|
| FTCS advection   | u_t+au_x  | UNCONDITIONALLY UNSTABLE    |
| Upwind           | u_t+au_x  | C ≤ 1                      |
| Lax-Friedrichs   | u_t+au_x  | C ≤ 1                      |
| FTCS diffusion   | u_t=Du_xx | D·Δt/Δx² ≤ 1/2            |
| Crank-Nicolson   | u_t=Du_xx | UNCONDITIONALLY STABLE      |
| Leapfrog         | u_t+au_x  | C ≤ 1                      |

Physics: wave propagation, diffusion, fluid dynamics, electromagnetic
wave simulation (FDTD), plasma physics.
"""

import numpy as np


def amplification_factor_ftcs_advection(C, k_h):
    """
    Amplification factor for FTCS scheme on u_t + a u_x = 0.
    
    u^{n+1}_j = u^n_j - C/2 (u^n_{j+1} - u^n_{j-1})
    
    G = 1 - iC sin(kh)
    |G|² = 1 + C² sin²(kh) > 1  →  ALWAYS UNSTABLE!
    """
    return 1 - 1j * C * np.sin(k_h)


def amplification_factor_upwind(C, k_h):
    """
    Amplification factor for upwind scheme (a > 0).
    
    u^{n+1}_j = u^n_j - C(u^n_j - u^n_{j-1})
    
    G = 1 - C(1 - e^{-ikh})
    |G| ≤ 1 iff C ≤ 1.
    """
    return 1 - C * (1 - np.exp(-1j * k_h))


def amplification_factor_lax_friedrichs(C, k_h):
    """
    Lax-Friedrichs scheme.
    
    u^{n+1}_j = 0.5(u^n_{j+1} + u^n_{j-1}) - C/2(u^n_{j+1} - u^n_{j-1})
    
    G = cos(kh) - iC sin(kh)
    |G|² = cos²(kh) + C² sin²(kh) ≤ 1 iff C ≤ 1.
    """
    return np.cos(k_h) - 1j * C * np.sin(k_h)


def amplification_factor_ftcs_diffusion(r, k_h):
    """
    FTCS for diffusion u_t = D u_xx.
    
    u^{n+1}_j = u^n_j + r(u^n_{j+1} - 2u^n_j + u^n_{j-1})
    
    where r = D·Δt/Δx².
    
    G = 1 - 4r sin²(kh/2)
    |G| ≤ 1 iff r ≤ 1/2.
    """
    return 1 - 4 * r * np.sin(k_h / 2)**2


def advection_demo(scheme, C, N=200, n_steps=200):
    """
    Solve u_t + a u_x = 0 with periodic BC and Gaussian initial condition.
    
    Parameters
    ----------
    scheme : str — 'ftcs', 'upwind', 'lax_friedrichs', 'lax_wendroff'
    C : float — Courant number a·dt/dx
    N : number of grid points
    n_steps : time steps
    
    Returns
    -------
    x, u_final, u_initial
    """
    dx = 1.0 / N
    a = 1.0  # Wave speed
    dt = C * dx / a
    x = np.linspace(0, 1, N, endpoint=False)
    
    # Gaussian initial condition
    u = np.exp(-200 * (x - 0.5)**2)
    u0 = u.copy()
    
    for n in range(n_steps):
        u_old = u.copy()
        
        if scheme == 'ftcs':
            # FTCS: unconditionally unstable for advection!
            for j in range(N):
                u[j] = u_old[j] - 0.5*C*(u_old[(j+1)%N] - u_old[(j-1)%N])
        
        elif scheme == 'upwind':
            for j in range(N):
                u[j] = u_old[j] - C*(u_old[j] - u_old[(j-1)%N])
        
        elif scheme == 'lax_friedrichs':
            for j in range(N):
                u[j] = 0.5*(u_old[(j+1)%N] + u_old[(j-1)%N]) - \
                       0.5*C*(u_old[(j+1)%N] - u_old[(j-1)%N])
        
        elif scheme == 'lax_wendroff':
            for j in range(N):
                u[j] = u_old[j] - 0.5*C*(u_old[(j+1)%N] - u_old[(j-1)%N]) + \
                       0.5*C**2*(u_old[(j+1)%N] - 2*u_old[j] + u_old[(j-1)%N])
        
        # Check for blowup
        if np.max(np.abs(u)) > 1e10:
            print(f"  {scheme}: BLEW UP at step {n} (C={C})")
            return x, u, u0
    
    return x, u, u0


def diffusion_stability_demo(r, N=100, n_steps=200):
    """
    Solve u_t = D u_xx with FTCS. Stable if r = D·dt/dx² ≤ 0.5.
    """
    dx = 1.0 / N
    x = np.linspace(0, 1, N + 1)
    
    u = np.sin(np.pi * x)
    u0 = u.copy()
    
    for n in range(n_steps):
        u_new = u.copy()
        for j in range(1, N):
            u_new[j] = u[j] + r * (u[j+1] - 2*u[j] + u[j-1])
        u_new[0] = 0
        u_new[N] = 0
        u = u_new
        
        if np.max(np.abs(u)) > 1e10:
            return x, u, u0, False
    
    return x, u, u0, True


def cfl_check(a, dx, dt):
    """
    Check CFL condition for advection.
    
    Returns Courant number and stability status.
    """
    C = abs(a) * dt / dx
    stable = C <= 1.0
    return C, stable


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("STABILITY ANALYSIS & CFL CONDITION")
    print("=" * 60)

    # --- Von Neumann analysis ---
    print("\n--- Von Neumann Stability Analysis ---")
    k_h = np.linspace(0, np.pi, 200)
    
    print("\nAmplification factors |G| (max over all k·h):")
    for C in [0.5, 0.9, 1.0, 1.1]:
        g_ftcs = np.max(np.abs(amplification_factor_ftcs_advection(C, k_h)))
        g_upwind = np.max(np.abs(amplification_factor_upwind(C, k_h)))
        g_lf = np.max(np.abs(amplification_factor_lax_friedrichs(C, k_h)))
        print(f"  C = {C}: FTCS={g_ftcs:.4f}, Upwind={g_upwind:.4f}, "
              f"Lax-Friedrichs={g_lf:.4f}")

    # --- Advection demonstrations ---
    print("\n--- Advection Equation: u_t + u_x = 0 ---")
    
    for scheme in ['ftcs', 'upwind', 'lax_friedrichs', 'lax_wendroff']:
        for C in [0.5, 0.9, 1.1]:
            x, u, u0 = advection_demo(scheme, C, N=200, n_steps=100)
            max_val = np.max(np.abs(u))
            status = "STABLE" if max_val < 2 else "UNSTABLE"
            print(f"  {scheme:16s} C={C}: max|u|={max_val:.2e}  [{status}]")

    # --- Diffusion stability ---
    print("\n--- Diffusion: u_t = D u_xx (FTCS) ---")
    for r in [0.3, 0.5, 0.51, 0.6, 1.0]:
        x, u, u0, stable = diffusion_stability_demo(r, N=50, n_steps=100)
        max_u = np.max(np.abs(u))
        status = "STABLE" if stable and max_u < 10 else "UNSTABLE"
        print(f"  r = {r:.2f}: max|u| = {max_u:.2e}  [{status}]")

    # --- CFL check ---
    print("\n--- CFL Condition Check ---")
    dx = 0.01
    a = 340  # Speed of sound (m/s)
    
    for dt in [1e-5, 2.5e-5, 3e-5, 5e-5]:
        C, stable = cfl_check(a, dx, dt)
        print(f"  a={a}, dx={dx}, dt={dt:.1e}: C = {C:.3f} "
              f"({'STABLE' if stable else 'UNSTABLE'})")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Amplification factors
        k_h = np.linspace(0, np.pi, 200)
        for C, color in [(0.5, 'b'), (0.9, 'g'), (1.0, 'orange'), (1.5, 'r')]:
            g = np.abs(amplification_factor_upwind(C, k_h))
            axes[0, 0].plot(k_h/np.pi, g, color=color, label=f'C={C}')
        axes[0, 0].axhline(1, color='k', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('k·h / π')
        axes[0, 0].set_ylabel('|G|')
        axes[0, 0].set_title('Upwind: |G(k)|')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # FTCS always unstable
        for C in [0.1, 0.5, 1.0]:
            g = np.abs(amplification_factor_ftcs_advection(C, k_h))
            axes[0, 1].plot(k_h/np.pi, g, label=f'C={C}')
        axes[0, 1].axhline(1, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('k·h / π')
        axes[0, 1].set_ylabel('|G|')
        axes[0, 1].set_title('FTCS Advection: |G| > 1 always!')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Diffusion stability
        for r, color in [(0.3, 'b'), (0.5, 'g'), (0.6, 'r')]:
            g = np.abs(amplification_factor_ftcs_diffusion(r, k_h))
            axes[0, 2].plot(k_h/np.pi, g, color=color, label=f'r={r}')
        axes[0, 2].axhline(1, color='k', linestyle='--', alpha=0.5)
        axes[0, 2].set_xlabel('k·h / π')
        axes[0, 2].set_ylabel('|G|')
        axes[0, 2].set_title('FTCS Diffusion: stable iff r ≤ 0.5')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Advection solutions
        schemes = ['upwind', 'lax_friedrichs', 'lax_wendroff']
        for i, scheme in enumerate(schemes):
            x, u, u0 = advection_demo(scheme, 0.8, N=200, n_steps=100)
            axes[1, i].plot(x, u0, 'k--', alpha=0.5, label='Initial')
            axes[1, i].plot(x, u, 'b-', linewidth=1.5, label=f'C=0.8')
            
            # Exact solution (shifted Gaussian)
            shift = 0.8 * 100 * (1.0/200)  # C * n_steps * dx
            u_exact = np.exp(-200 * ((x - 0.5 - shift) % 1.0 - 0.5)**2)
            # Handle wrap-around
            u_exact2 = np.exp(-200 * ((x - 0.5 - shift + 1) % 1.0 - 0.5)**2)
            u_exact = np.maximum(u_exact, u_exact2)
            axes[1, i].plot(x, u_exact, 'r--', alpha=0.7, label='Exact')
            
            axes[1, i].set_xlabel('x')
            axes[1, i].set_ylabel('u')
            axes[1, i].set_title(f'{scheme}')
            axes[1, i].legend(fontsize=8)
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("stability_cfl.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
