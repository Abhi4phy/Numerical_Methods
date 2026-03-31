"""
Finite Volume Method (FVM)
===========================
Conservative discretization for hyperbolic and parabolic PDEs.

**Why Finite Volumes?**
The FVM directly discretizes the INTEGRAL form of conservation laws:

    ∂u/∂t + ∇·F(u) = 0  →  d/dt ∫_V u dV = -∮_∂V F · n dS

By working with cell averages and fluxes through cell faces,
the FVM automatically conserves quantities (mass, momentum, energy).

**Key Concepts:**
1. **Cell averages**: ū_i = (1/Δx) ∫ u dx over cell i
2. **Numerical flux**: F_{i+1/2} = flux at interface between cells i and i+1
3. **Update**: ū_i^{n+1} = ū_i^n - (Δt/Δx)[F_{i+1/2} - F_{i-1/2}]

**Flux Schemes (1D):**

1. **Upwind (Godunov)**: uses information from upstream
   F_{i+1/2} = F(u_i) if a > 0, else F(u_{i+1})

2. **Lax-Friedrichs**: simple but diffusive
   F = [F(u_L) + F(u_R)]/2 - (Δx/2Δt)(u_R - u_L)

3. **Lax-Wendroff**: 2nd order in time and space

4. **MUSCL + slope limiting**: high-resolution, TVD
   Reconstruct piecewise linear states, apply slope limiter.

5. **HLL / HLLC Riemann solver**: approximate Riemann solver

**Applications:**
Euler equations (compressible flow), shallow water, MHD,
traffic flow, neutron transport.

Where to start:
━━━━━━━━━━━━━━
Run the 1D advection example. Try different flux schemes and see
how upwind is stable but diffusive, Lax-Wendroff is accurate but
oscillatory near discontinuities, and MUSCL+limiter combines both.
Prerequisite: finite_difference_method.py
"""

import numpy as np


# ============================================================
# Numerical Flux Functions
# ============================================================

def flux_upwind(uL, uR, flux_func, wave_speed):
    """
    Upwind (Godunov) flux for linear advection.
    
    Uses information from the upwind direction.
    1st order, very stable but diffusive.
    """
    a = wave_speed
    if a >= 0:
        return flux_func(uL)
    else:
        return flux_func(uR)


def flux_lax_friedrichs(uL, uR, flux_func, alpha):
    """
    Lax-Friedrichs flux.
    
    F_{i+1/2} = [F(u_L) + F(u_R)]/2 - (α/2)(u_R - u_L)
    
    α = max |f'(u)| (maximum wave speed)
    Simple but very diffusive.
    """
    return 0.5 * (flux_func(uL) + flux_func(uR)) - 0.5 * alpha * (uR - uL)


def flux_lax_wendroff(uL, uR, flux_func, a, dt, dx):
    """
    Lax-Wendroff flux — 2nd order in time and space.
    
    F = [F(uL) + F(uR)]/2 - (a·dt/2dx)(F(uR) - F(uL))
    
    Accurate but oscillates near discontinuities (no TVD).
    """
    return 0.5 * (flux_func(uL) + flux_func(uR)) - 0.5 * a * dt / dx * (flux_func(uR) - flux_func(uL))


def flux_rusanov(uL, uR, flux_func, max_speed_func):
    """
    Rusanov (local Lax-Friedrichs) flux.
    
    Uses LOCAL maximum wave speed.
    Better than global LF for nonlinear problems.
    """
    smax = max(abs(max_speed_func(uL)), abs(max_speed_func(uR)))
    return 0.5 * (flux_func(uL) + flux_func(uR)) - 0.5 * smax * (uR - uL)


# ============================================================
# Slope Limiters (for MUSCL reconstruction)
# ============================================================

def minmod(a, b):
    """Minmod limiter — most diffusive TVD limiter."""
    return np.where(a * b > 0, 
                    np.sign(a) * np.minimum(np.abs(a), np.abs(b)),
                    0.0)


def van_leer(r):
    """Van Leer limiter — smooth, good general choice."""
    return np.where(r > 0, 2 * r / (1 + r), 0.0)


def superbee(r):
    """Superbee limiter — least diffusive TVD limiter."""
    return np.maximum(0, np.maximum(np.minimum(2*r, 1), np.minimum(r, 2)))


def mc_limiter(r):
    """MC (Monotonized Central) limiter."""
    return np.maximum(0, np.minimum(np.minimum(2*r, (1+r)/2), 2))


def slope_ratio(u, i):
    """Compute consecutive slope ratio r for limiter functions."""
    denom = u[i+1] - u[i]
    numer = u[i] - u[i-1]
    # Avoid division by zero
    r = np.where(np.abs(denom) > 1e-15, numer / denom, 0.0)
    return r


# ============================================================
# FVM Solvers
# ============================================================

def fvm_advection_1d(u0, a, dx, dt, n_steps, flux_scheme='upwind',
                     limiter='minmod', bc='periodic'):
    """
    1D linear advection: ∂u/∂t + a·∂u/∂x = 0
    
    Parameters
    ----------
    u0 : array
        Initial cell averages
    a : float
        Advection speed
    dx : float
        Cell size
    dt : float
        Time step
    n_steps : int
        Number of time steps
    flux_scheme : str
        'upwind', 'lax-friedrichs', 'lax-wendroff', 'muscl'
    limiter : str
        Slope limiter for MUSCL: 'minmod', 'van_leer', 'superbee', 'mc'
    bc : str
        'periodic' or 'outflow'
    """
    N = len(u0)
    u = u0.copy()
    flux_func = lambda u: a * u
    
    limiter_funcs = {
        'minmod': lambda r: minmod(r, np.ones_like(r)),
        'van_leer': van_leer,
        'superbee': superbee,
        'mc': mc_limiter,
    }
    
    for step in range(n_steps):
        # Apply boundary conditions (ghost cells)
        if bc == 'periodic':
            u_ext = np.concatenate([[u[-2], u[-1]], u, [u[0], u[1]]])
        else:
            u_ext = np.concatenate([[u[0], u[0]], u, [u[-1], u[-1]]])
        
        # Compute fluxes at interfaces
        fluxes = np.zeros(N + 1)
        
        for i in range(N + 1):
            iL = i + 1  # Left cell index in extended array
            iR = i + 2  # Right cell index
            
            if flux_scheme == 'upwind':
                fluxes[i] = flux_upwind(u_ext[iL], u_ext[iR], flux_func, a)
            
            elif flux_scheme == 'lax-friedrichs':
                fluxes[i] = flux_lax_friedrichs(u_ext[iL], u_ext[iR], 
                                                  flux_func, abs(a))
            
            elif flux_scheme == 'lax-wendroff':
                fluxes[i] = flux_lax_wendroff(u_ext[iL], u_ext[iR], 
                                                flux_func, a, dt, dx)
            
            elif flux_scheme == 'muscl':
                # MUSCL reconstruction with slope limiting
                lim = limiter_funcs.get(limiter, van_leer)
                
                # Slope ratios
                dL = u_ext[iL] - u_ext[iL-1]
                dR = u_ext[iR] - u_ext[iL]
                dRR = u_ext[iR+1] - u_ext[iR] if iR+1 < len(u_ext) else dR
                
                rL = dL / (dR + 1e-15) if abs(dR) > 1e-15 else 0.0
                rR = dR / (dRR + 1e-15) if abs(dRR) > 1e-15 else 0.0
                
                if np.isscalar(rL):
                    rL = np.array([rL])
                    rR = np.array([rR])
                
                phiL = lim(rL)[0] if len(lim(rL)) > 0 else 0
                phiR = lim(rR)[0] if len(lim(rR)) > 0 else 0
                
                # Reconstructed left and right states
                uL_face = u_ext[iL] + 0.5 * phiL * dR
                uR_face = u_ext[iR] - 0.5 * phiR * dR
                
                fluxes[i] = flux_rusanov(uL_face, uR_face, flux_func, 
                                          lambda u: a)
        
        # Update cell averages
        u = u - (dt / dx) * (fluxes[1:] - fluxes[:-1])
    
    return u


def fvm_burgers_1d(u0, dx, dt, n_steps, flux_scheme='rusanov'):
    """
    1D inviscid Burgers equation: ∂u/∂t + ∂(u²/2)/∂x = 0
    
    Nonlinear conservation law that develops shocks and rarefactions.
    """
    N = len(u0)
    u = u0.copy()
    flux_func = lambda u: 0.5 * u**2
    speed_func = lambda u: np.abs(u)
    
    for step in range(n_steps):
        # Periodic BC
        u_ext = np.concatenate([[u[-1]], u, [u[0]]])
        
        fluxes = np.zeros(N + 1)
        for i in range(N + 1):
            uL = u_ext[i]
            uR = u_ext[i+1]
            
            if flux_scheme == 'rusanov':
                fluxes[i] = flux_rusanov(uL, uR, flux_func, speed_func)
            elif flux_scheme == 'godunov':
                # Exact Godunov for Burgers
                if uL >= uR:  # Shock
                    if uL > 0 and uR > 0:
                        fluxes[i] = flux_func(uL)
                    elif uL < 0 and uR < 0:
                        fluxes[i] = flux_func(uR)
                    else:
                        fluxes[i] = max(flux_func(uL), flux_func(uR))
                else:  # Rarefaction
                    if uL >= 0:
                        fluxes[i] = flux_func(uL)
                    elif uR <= 0:
                        fluxes[i] = flux_func(uR)
                    else:
                        fluxes[i] = 0.0
        
        u = u - (dt / dx) * (fluxes[1:] - fluxes[:-1])
    
    return u


def euler_1d_fvm(rho0, vel0, pres0, dx, dt, n_steps, gamma=1.4):
    """
    1D Euler equations (compressible gas dynamics).
    
    Conservation form:
        ∂/∂t [ρ, ρu, E] + ∂/∂x [ρu, ρu²+p, u(E+p)] = 0
    
    Uses Rusanov flux with piecewise constant reconstruction.
    
    Parameters
    ----------
    rho0, vel0, pres0 : arrays
        Initial density, velocity, pressure
    gamma : float
        Ratio of specific heats (1.4 for air)
    """
    N = len(rho0)
    
    # Conservative variables: [ρ, ρu, E]
    U = np.zeros((N, 3))
    U[:, 0] = rho0
    U[:, 1] = rho0 * vel0
    U[:, 2] = pres0 / (gamma - 1) + 0.5 * rho0 * vel0**2
    
    def primitive(U):
        """Convert conservative → primitive variables."""
        rho = U[:, 0]
        vel = U[:, 1] / rho
        pres = (gamma - 1) * (U[:, 2] - 0.5 * rho * vel**2)
        return rho, vel, pres
    
    def flux(U):
        """Physical flux F(U)."""
        rho, vel, pres = primitive(U.reshape(-1, 3) if U.ndim == 1 else U)
        F = np.zeros_like(U.reshape(-1, 3) if U.ndim == 1 else U)
        E = U[:, 2] if U.ndim > 1 else U.reshape(-1, 3)[:, 2]
        F_flat = F if U.ndim > 1 else F
        F_flat[:, 0] = rho * vel
        F_flat[:, 1] = rho * vel**2 + pres
        F_flat[:, 2] = vel * (E + pres)
        return F_flat
    
    def max_wave_speed(UL, UR):
        """Maximum wave speed for Rusanov."""
        rhoL, velL, presL = primitive(UL.reshape(1, 3))
        rhoR, velR, presR = primitive(UR.reshape(1, 3))
        aL = np.sqrt(gamma * presL / rhoL)
        aR = np.sqrt(gamma * presR / rhoR)
        return max(abs(velL[0]) + aL[0], abs(velR[0]) + aR[0])
    
    for step in range(n_steps):
        # Transmissive (outflow) boundary
        U_ext = np.vstack([U[0], U, U[-1]])
        
        # Compute fluxes
        F_faces = np.zeros((N + 1, 3))
        
        for i in range(N + 1):
            UL = U_ext[i]
            UR = U_ext[i+1]
            
            smax = max_wave_speed(UL, UR)
            FL = flux(UL.reshape(1, 3))[0]
            FR = flux(UR.reshape(1, 3))[0]
            
            F_faces[i] = 0.5 * (FL + FR) - 0.5 * smax * (UR - UL)
        
        # Update
        U = U - (dt / dx) * (F_faces[1:] - F_faces[:-1])
    
    rho, vel, pres = primitive(U)
    return rho, vel, pres


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FINITE VOLUME METHOD DEMO")
    print("=" * 60)

    # TO TEST: Vary CFL, flux_scheme, limiter, and initial discontinuity width/shape.
    # Observe stability, diffusion, and oscillation tradeoffs via L2 error and profile sharpness.
    # --- 1. Linear Advection ---
    print("\n--- Linear Advection ---")
    N = 200
    dx = 1.0 / N
    a = 1.0
    CFL = 0.8
    dt = CFL * dx / a
    
    x = np.linspace(0.5*dx, 1.0-0.5*dx, N)
    
    # Initial condition: square wave + Gaussian
    u0 = np.where((x > 0.1) & (x < 0.3), 1.0, 0.0)
    u0 += np.exp(-((x - 0.7)/0.05)**2)
    
    # Advect for one period
    n_steps = int(1.0 / (a * dt))
    
    schemes = ['upwind', 'lax-friedrichs', 'lax-wendroff', 'muscl']
    results = {}
    
    for scheme in schemes:
        u_final = fvm_advection_1d(u0, a, dx, dt, n_steps, flux_scheme=scheme)
        err = np.sqrt(dx * np.sum((u_final - u0)**2))
        results[scheme] = (u_final, err)
        print(f"  {scheme:20s}: L2 error = {err:.6e}")

    # TO TEST: Change initial wave amplitude, final time, and solver flux (godunov/rusanov).
    # Observe shock steepening time and post-shock numerical dissipation behavior.
    # --- 2. Burgers Equation (shock formation) ---
    print("\n--- Inviscid Burgers Equation ---")
    N_b = 400
    dx_b = 2.0 / N_b
    x_b = np.linspace(-1 + 0.5*dx_b, 1 - 0.5*dx_b, N_b)
    
    # Sine wave → develops shock
    u0_b = np.sin(np.pi * x_b)
    
    CFL_b = 0.5
    dt_b = CFL_b * dx_b  # max speed = 1
    n_steps_b = int(0.5 / dt_b)
    
    u_burgers = fvm_burgers_1d(u0_b, dx_b, dt_b, n_steps_b, flux_scheme='godunov')
    print(f"  Shock formed at t = 1/π ≈ {1/np.pi:.4f}")
    print(f"  Simulated to t = {n_steps_b * dt_b:.4f}")
    print(f"  max(u) = {u_burgers.max():.4f}, min(u) = {u_burgers.min():.4f}")

    # TO TEST: Modify left/right states, gamma, dt, and resolution N_e.
    # Observe shock/contact/rarefaction structure and sensitivity to timestep/resolution.
    # --- 3. Sod Shock Tube (Euler equations) ---
    print("\n--- Sod Shock Tube (Euler Equations) ---")
    N_e = 400
    dx_e = 1.0 / N_e
    x_e = np.linspace(0.5*dx_e, 1.0 - 0.5*dx_e, N_e)
    
    # Initial conditions: left state / right state
    rho0 = np.where(x_e < 0.5, 1.0, 0.125)
    vel0 = np.zeros(N_e)
    pres0 = np.where(x_e < 0.5, 1.0, 0.1)
    
    dt_e = 0.0002
    n_steps_e = int(0.2 / dt_e)
    
    rho_f, vel_f, pres_f = euler_1d_fvm(rho0, vel0, pres0, dx_e, dt_e, n_steps_e)
    
    print(f"  Sod shock tube at t = {n_steps_e * dt_e:.3f}")
    print(f"  Density:  [{rho_f.min():.4f}, {rho_f.max():.4f}]")
    print(f"  Velocity: [{vel_f.min():.4f}, {vel_f.max():.4f}]")
    print(f"  Pressure: [{pres_f.min():.4f}, {pres_f.max():.4f}]")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Advection results
        ax = axes[0, 0]
        ax.plot(x, u0, 'k-', linewidth=2, label='Exact')
        for scheme in schemes:
            ax.plot(x, results[scheme][0], '--', linewidth=1, 
                   alpha=0.8, label=scheme)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title('Linear Advection (1 period)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Zoom on discontinuity
        ax = axes[0, 1]
        mask = (x > 0.05) & (x < 0.4)
        ax.plot(x[mask], u0[mask], 'k-', linewidth=2, label='Exact')
        for scheme in schemes:
            ax.plot(x[mask], results[scheme][0][mask], '--', linewidth=1.5, 
                   alpha=0.8, label=scheme)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title('Zoom: Square Wave')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Burgers
        ax = axes[0, 2]
        ax.plot(x_b, u0_b, 'b--', linewidth=1, label='t=0')
        ax.plot(x_b, u_burgers, 'r-', linewidth=2, label=f't={n_steps_b*dt_b:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title("Burgers' Equation (shock)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Sod shock tube: density
        ax = axes[1, 0]
        ax.plot(x_e, rho0, 'b--', linewidth=1, label='t=0')
        ax.plot(x_e, rho_f, 'r-', linewidth=2, label=f't={n_steps_e*dt_e:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('ρ')
        ax.set_title('Sod Shock Tube: Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Sod: velocity
        ax = axes[1, 1]
        ax.plot(x_e, vel0, 'b--', linewidth=1, label='t=0')
        ax.plot(x_e, vel_f, 'r-', linewidth=2, label=f't={n_steps_e*dt_e:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('v')
        ax.set_title('Sod Shock Tube: Velocity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Sod: pressure
        ax = axes[1, 2]
        ax.plot(x_e, pres0, 'b--', linewidth=1, label='t=0')
        ax.plot(x_e, pres_f, 'r-', linewidth=2, label=f't={n_steps_e*dt_e:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('p')
        ax.set_title('Sod Shock Tube: Pressure')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("finite_volume_method.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
