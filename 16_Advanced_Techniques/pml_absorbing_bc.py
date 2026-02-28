"""
Perfectly Matched Layer (PML)
=============================
Absorbing boundary condition for wave simulations that eliminates
reflections at domain boundaries — the wave just "disappears."

**The Problem:**
When simulating waves in a finite domain, waves hitting the boundary
reflect back, corrupting the solution. We need boundaries that absorb
outgoing waves perfectly.

**The Idea (Bérenger, 1994):**
Surround the computational domain with a layer where the wave
equation is modified to include DAMPING that increases with distance
into the layer. The key insight: the damping is applied in a
split-field formulation that makes the interface impedance-matched
(zero reflection at ANY angle and ANY frequency).

**For 1D wave equation:**
    ∂²u/∂t² = c²∂²u/∂x²

In PML region, replace ∂/∂x → (1/(1 + σ(x)/iω)) ∂/∂x

Equivalent to complex coordinate stretching:
    x → x̃ = x + (i/ω) ∫₀ˣ σ(s) ds

**Damping Profile:**
    σ(x) = σ_max · (d/L_pml)^p

where d = distance into PML, p = polynomial order (usually 2-3).

**Optimal σ_max:**
    σ_max ≈ -(p+1) · c · ln(R) / (2 · L_pml)

where R is the desired reflection coefficient (e.g., R = 10⁻⁶).

**Applications:**
- Electromagnetic simulations (FDTD)
- Seismic wave propagation
- Acoustic simulations
- Quantum mechanics (outgoing wave problems)
- Gravitational wave extraction

Where to start:
━━━━━━━━━━━━━━
1. Start with the 1D wave equation with PML
2. Compare: hard boundary (full reflection) vs PML (absorbed)
3. Then try 2D — watch a circular wave vanish at the edges
Prerequisites: finite_difference_method.py, spectral_methods.py
"""

import numpy as np


# ============================================================
# PML Damping Profiles
# ============================================================

def pml_profile_polynomial(x, x_pml, L_pml, sigma_max, p=3):
    """
    Polynomial PML damping profile.
    
    σ(x) = σ_max · (d/L_pml)^p for x inside PML
    σ(x) = 0 for x inside computational domain
    
    Parameters
    ----------
    x : array
        Grid positions
    x_pml : float
        Start of PML region
    L_pml : float
        PML layer thickness
    sigma_max : float
        Maximum damping
    p : int
        Polynomial order (2-4)
    """
    sigma = np.zeros_like(x)
    in_pml = np.abs(x) > x_pml
    d = np.abs(x[in_pml]) - x_pml
    sigma[in_pml] = sigma_max * (d / L_pml)**p
    return sigma


def optimal_sigma_max(c, L_pml, R=1e-6, p=3):
    """
    Compute optimal σ_max for target reflection coefficient R.
    
    σ_max = -(p+1) · c · ln(R) / (2 · L_pml)
    """
    return -(p + 1) * c * np.log(R) / (2 * L_pml)


# ============================================================
# 1D Wave Equation with PML
# ============================================================

def wave_1d_pml(Nx, L, L_pml, c, dt, n_steps,
                source_func=None, source_pos=0.0):
    """
    1D wave equation with PML using auxiliary differential equation (ADE).
    
    ∂²u/∂t² + (σ_x + σ_x) ∂u/∂t + σ_x·σ_x·u = c²∂²u/∂x² + c²σ_x ∂ψ/∂x
    
    Simplified split approach for 1D:
    We use a first-order system:
        ∂u/∂t = v
        ∂v/∂t = c²∂²u/∂x² - σ(x)·v
    
    The damping σ(x) = 0 in the interior and ramps up in the PML.
    
    Parameters
    ----------
    Nx : int
        Number of grid points
    L : float
        Half-domain size (domain is [-L, L])
    L_pml : float
        PML thickness
    c : float
        Wave speed
    dt : float
        Time step
    n_steps : int
        Number of time steps
    source_func : callable, optional
        Source function f(t) at source_pos
    source_pos : float
        Position of point source
    
    Returns
    -------
    x : array
        Grid points
    u_history : list of arrays
        Solution at selected time steps
    """
    dx = 2 * L / (Nx - 1)
    x = np.linspace(-L, L, Nx)
    
    # PML starts at x_pml = L - L_pml
    x_pml = L - L_pml
    sigma_max_val = optimal_sigma_max(c, L_pml)
    sigma = pml_profile_polynomial(x, x_pml, L_pml, sigma_max_val)
    
    # Fields
    u = np.zeros(Nx)       # Displacement
    v = np.zeros(Nx)       # Velocity (∂u/∂t)
    
    # Source position index
    i_src = np.argmin(np.abs(x - source_pos))
    
    u_history = []
    save_every = max(1, n_steps // 50)
    
    for step in range(n_steps):
        t = step * dt
        
        # Compute spatial derivative (central differences)
        d2u_dx2 = np.zeros(Nx)
        d2u_dx2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        
        # Update velocity (leapfrog-like)
        v += dt * (c**2 * d2u_dx2 - sigma * v)
        
        # Add source
        if source_func is not None:
            v[i_src] += dt * source_func(t)
        
        # Update displacement
        u += dt * v
        
        # Apply PML damping to displacement as well
        u *= np.exp(-sigma * dt)
        
        if step % save_every == 0:
            u_history.append(u.copy())
    
    return x, u_history


def wave_1d_no_pml(Nx, L, c, dt, n_steps, 
                    source_func=None, source_pos=0.0, bc='reflecting'):
    """
    1D wave equation WITHOUT PML (for comparison).
    
    bc : str
        'reflecting' (hard wall) or 'absorbing' (first-order ABC)
    """
    dx = 2 * L / (Nx - 1)
    x = np.linspace(-L, L, Nx)
    
    u = np.zeros(Nx)
    v = np.zeros(Nx)
    
    i_src = np.argmin(np.abs(x - source_pos))
    
    u_history = []
    save_every = max(1, n_steps // 50)
    
    for step in range(n_steps):
        t = step * dt
        
        d2u_dx2 = np.zeros(Nx)
        d2u_dx2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        
        v += dt * c**2 * d2u_dx2
        
        if source_func is not None:
            v[i_src] += dt * source_func(t)
        
        u += dt * v
        
        # Boundary conditions
        if bc == 'reflecting':
            u[0] = 0
            u[-1] = 0
        elif bc == 'absorbing':
            # First-order Mur ABC
            u[0] = u[1]   # Simple approximation
            u[-1] = u[-2]
        
        if step % save_every == 0:
            u_history.append(u.copy())
    
    return x, u_history


# ============================================================
# 2D Wave Equation with PML
# ============================================================

def wave_2d_pml(Nx, Ny, Lx, Ly, L_pml, c, dt, n_steps,
                source_func=None, source_pos=(0.0, 0.0)):
    """
    2D wave equation with PML.
    
    Uses split-field PML approach.
    
    Parameters
    ----------
    Nx, Ny : int
        Grid points in x and y
    Lx, Ly : float
        Half-domain sizes
    L_pml : float
        PML thickness
    c : float
        Wave speed
    """
    dx = 2 * Lx / (Nx - 1)
    dy = 2 * Ly / (Ny - 1)
    
    x = np.linspace(-Lx, Lx, Nx)
    y = np.linspace(-Ly, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    
    # PML damping in x and y
    x_pml = Lx - L_pml
    y_pml = Ly - L_pml
    sigma_max_val = optimal_sigma_max(c, L_pml)
    
    sigma_x = pml_profile_polynomial(x, x_pml, L_pml, sigma_max_val)
    sigma_y = pml_profile_polynomial(y, y_pml, L_pml, sigma_max_val)
    SIGMA_X = np.tile(sigma_x, (Ny, 1))
    SIGMA_Y = np.tile(sigma_y.reshape(-1, 1), (1, Nx))
    
    # Fields
    u = np.zeros((Ny, Nx))
    vx = np.zeros((Ny, Nx))  # Split velocity x-component
    vy = np.zeros((Ny, Nx))  # Split velocity y-component
    
    # Source
    i_src = np.argmin(np.abs(x - source_pos[0]))
    j_src = np.argmin(np.abs(y - source_pos[1]))
    
    u_history = []
    save_every = max(1, n_steps // 30)
    
    for step in range(n_steps):
        t = step * dt
        
        # Spatial derivatives
        d2u_dx2 = np.zeros((Ny, Nx))
        d2u_dy2 = np.zeros((Ny, Nx))
        
        d2u_dx2[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]) / dx**2
        d2u_dy2[1:-1, :] = (u[2:, :] - 2*u[1:-1, :] + u[:-2, :]) / dy**2
        
        # Update split velocities with PML damping
        vx += dt * (c**2 * d2u_dx2 - SIGMA_X * vx)
        vy += dt * (c**2 * d2u_dy2 - SIGMA_Y * vy)
        
        # Source
        if source_func is not None:
            vx[j_src, i_src] += dt * source_func(t) * 0.5
            vy[j_src, i_src] += dt * source_func(t) * 0.5
        
        # Update displacement
        u += dt * (vx + vy)
        
        # Extra PML damping on corners
        corner_damp = np.exp(-(SIGMA_X + SIGMA_Y) * dt * 0.5)
        u *= corner_damp
        
        if step % save_every == 0:
            u_history.append(u.copy())
    
    return x, y, u_history


# ============================================================
# Source Functions
# ============================================================

def ricker_wavelet(t, f0=5.0, t0=None):
    """
    Ricker wavelet (Mexican hat) source.
    
    Common in seismology. Centered at t0 with peak frequency f0.
    """
    if t0 is None:
        t0 = 1.5 / f0
    
    tau = (t - t0) * f0
    return (1 - 2 * np.pi**2 * tau**2) * np.exp(-np.pi**2 * tau**2)


def gaussian_pulse(t, t0=0.3, width=0.05):
    """Gaussian pulse source."""
    return np.exp(-(t - t0)**2 / (2 * width**2))


def harmonic_source(t, freq=5.0, t_ramp=0.5):
    """Harmonic sinusoidal source with smooth ramp-up."""
    ramp = np.minimum(t / t_ramp, 1.0)
    return ramp * np.sin(2 * np.pi * freq * t)


# ============================================================
# Analysis
# ============================================================

def measure_reflection(u_history, x, x_measure, x_pml_start):
    """
    Measure PML reflection coefficient.
    
    Compare wave amplitude at measurement point before and after
    reaching PML boundary.
    """
    idx = np.argmin(np.abs(x - x_measure))
    
    signal = np.array([u[idx] for u in u_history])
    
    # Find peak of incident wave
    peak_idx = np.argmax(np.abs(signal[:len(signal)//2]))
    incident_amp = np.abs(signal[peak_idx])
    
    # Find peak of reflected wave (after incident has passed)
    reflect_signal = signal[peak_idx + len(signal)//4:]
    if len(reflect_signal) > 0:
        reflected_amp = np.max(np.abs(reflect_signal))
    else:
        reflected_amp = 0.0
    
    if incident_amp > 0:
        return reflected_amp / incident_amp
    return 0.0


def energy_in_domain(u, x, x_pml_start, dx):
    """Compute wave energy inside the physical domain (excluding PML)."""
    mask = np.abs(x) < x_pml_start
    return np.sum(u[mask]**2) * dx


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PERFECTLY MATCHED LAYER (PML) DEMO")
    print("=" * 60)

    # --- 1. 1D comparison: PML vs reflecting ---
    print("\n--- 1D Wave: PML vs Reflecting Boundary ---")
    
    Nx = 500
    L = 5.0
    L_pml = 1.0
    c = 1.0
    dx = 2 * L / (Nx - 1)
    dt = 0.5 * dx / c  # CFL condition
    n_steps = 2000
    
    source = lambda t: ricker_wavelet(t, f0=3.0)
    
    # With PML
    x_pml, u_hist_pml = wave_1d_pml(Nx, L, L_pml, c, dt, n_steps,
                                     source_func=source)
    
    # Without PML (reflecting)
    x_ref, u_hist_ref = wave_1d_no_pml(Nx, L, c, dt, n_steps,
                                        source_func=source, bc='reflecting')
    
    # With first-order ABC
    x_abc, u_hist_abc = wave_1d_no_pml(Nx, L, c, dt, n_steps,
                                        source_func=source, bc='absorbing')
    
    # Energy comparison at late time
    x_pml_start = L - L_pml
    e_pml = energy_in_domain(u_hist_pml[-1], x_pml, x_pml_start, dx)
    e_ref = energy_in_domain(u_hist_ref[-1], x_ref, x_pml_start, dx)
    e_abc = energy_in_domain(u_hist_abc[-1], x_abc, x_pml_start, dx)
    
    print(f"  Grid: {Nx} points, domain: [-{L}, {L}]")
    print(f"  PML thickness: {L_pml}")
    print(f"  Residual energy at t_final:")
    print(f"    Reflecting BC: {e_ref:.6f}")
    print(f"    1st-order ABC: {e_abc:.6f}")
    print(f"    PML:           {e_pml:.6f}")
    if e_ref > 0:
        print(f"  PML reduction: {e_pml/e_ref:.2e}× reflecting")

    # --- 2. PML parameter study ---
    print("\n--- PML Parameter Study ---")
    
    print(f"  {'L_pml':>6s} {'σ_max':>10s} {'Residual Energy':>16s}")
    for L_pml_test in [0.25, 0.5, 1.0, 1.5, 2.0]:
        _, u_test = wave_1d_pml(Nx, L, L_pml_test, c, dt, n_steps,
                                 source_func=source)
        x_pml_s = L - L_pml_test
        e = energy_in_domain(u_test[-1], x_pml, x_pml_s, dx)
        sm = optimal_sigma_max(c, L_pml_test)
        print(f"  {L_pml_test:6.2f} {sm:10.2f} {e:16.2e}")

    # --- 3. Polynomial order study ---
    print("\n--- Damping Profile Order Study ---")
    print(f"  {'Order p':>8s} {'Residual Energy':>16s}")
    
    for p in [1, 2, 3, 4, 5]:
        # Manual PML with different orders
        dx_t = 2 * L / (Nx - 1)
        x_t = np.linspace(-L, L, Nx)
        x_pml_s = L - 1.0
        sm = optimal_sigma_max(c, 1.0, p=p)
        sigma_t = pml_profile_polynomial(x_t, x_pml_s, 1.0, sm, p=p)
        
        u_t = np.zeros(Nx)
        v_t = np.zeros(Nx)
        i_src = Nx // 2
        
        for step in range(n_steps):
            t = step * dt
            d2u = np.zeros(Nx)
            d2u[1:-1] = (u_t[2:] - 2*u_t[1:-1] + u_t[:-2]) / dx_t**2
            v_t += dt * (c**2 * d2u - sigma_t * v_t)
            v_t[i_src] += dt * source(t)
            u_t += dt * v_t
            u_t *= np.exp(-sigma_t * dt)
        
        e_p = energy_in_domain(u_t, x_t, x_pml_s, dx_t)
        print(f"  {p:8d} {e_p:16.2e}")

    # --- 4. 2D simulation ---
    print("\n--- 2D Wave with PML ---")
    
    Nx_2d = 100
    Ny_2d = 100
    Lx_2d = 3.0
    Ly_2d = 3.0
    L_pml_2d = 0.8
    dx_2d = 2 * Lx_2d / (Nx_2d - 1)
    dt_2d = 0.4 * dx_2d / (c * np.sqrt(2))
    
    source_2d = lambda t: ricker_wavelet(t, f0=4.0)
    
    x_2d, y_2d, u_hist_2d = wave_2d_pml(
        Nx_2d, Ny_2d, Lx_2d, Ly_2d, L_pml_2d, c, dt_2d, 
        n_steps=800, source_func=source_2d)
    
    # Energy in physical domain
    x_phys = Lx_2d - L_pml_2d
    mask_2d = (np.abs(x_2d) < x_phys)
    
    e_early = np.sum(u_hist_2d[len(u_hist_2d)//4][:, mask_2d]**2) * dx_2d**2
    e_late = np.sum(u_hist_2d[-1][:, mask_2d]**2) * dx_2d**2
    
    print(f"  Grid: {Nx_2d}×{Ny_2d}, PML = {L_pml_2d}")
    print(f"  Energy at t/4: {e_early:.6f}")
    print(f"  Energy at end: {e_late:.6f}")

    # --- 5. Ricker wavelet ---
    print("\n--- Ricker Wavelet Source ---")
    t_test = np.linspace(0, 1, 200)
    r_test = ricker_wavelet(t_test, f0=5.0)
    print(f"  Peak frequency: 5.0 Hz")
    print(f"  Peak time: {t_test[np.argmax(r_test)]:.3f}")
    print(f"  Max amplitude: {np.max(r_test):.3f}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1D comparison at multiple times
        ax = axes[0, 0]
        n_show = min(len(u_hist_ref), 8)
        for i in range(0, n_show, 2):
            ax.plot(x_ref, u_hist_ref[i], 'b-', alpha=0.3 + 0.7*i/n_show,
                   linewidth=0.8)
        ax.axvline(L - 1, color='r', linestyle='--', alpha=0.5, label='PML start')
        ax.axvline(-(L - 1), color='r', linestyle='--', alpha=0.5)
        ax.set_title('Reflecting BC')
        ax.set_xlabel('x')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        for i in range(0, min(len(u_hist_pml), 8), 2):
            ax.plot(x_pml, u_hist_pml[i], 'b-', 
                   alpha=0.3 + 0.7*i/n_show, linewidth=0.8)
        ax.axvline(L - 1, color='r', linestyle='--', alpha=0.5, label='PML start')
        ax.axvline(-(L - 1), color='r', linestyle='--', alpha=0.5)
        ax.set_title('With PML')
        ax.set_xlabel('x')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Final comparison
        ax = axes[0, 2]
        ax.plot(x_ref, u_hist_ref[-1], 'b-', label='Reflecting', linewidth=1.5)
        ax.plot(x_abc, u_hist_abc[-1], 'g-', label='1st-order ABC', linewidth=1.5)
        ax.plot(x_pml, u_hist_pml[-1], 'r-', label='PML', linewidth=1.5)
        ax.set_title('Final State Comparison')
        ax.set_xlabel('x')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # PML damping profile
        ax = axes[1, 0]
        x_prof = np.linspace(-L, L, 500)
        for p in [1, 2, 3, 4]:
            sm = optimal_sigma_max(c, 1.0, p=p)
            sigma_prof = pml_profile_polynomial(x_prof, L-1, 1.0, sm, p=p)
            ax.plot(x_prof, sigma_prof, label=f'p={p}')
        ax.set_xlabel('x')
        ax.set_ylabel('σ(x)')
        ax.set_title('PML Damping Profiles')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2D snapshots
        if len(u_hist_2d) >= 2:
            ax = axes[1, 1]
            snap_idx = min(len(u_hist_2d)//3, len(u_hist_2d)-1)
            vmax = np.max(np.abs(u_hist_2d[snap_idx]))
            ax.imshow(u_hist_2d[snap_idx], cmap='RdBu',
                     vmin=-vmax, vmax=vmax,
                     extent=[-Lx_2d, Lx_2d, -Ly_2d, Ly_2d])
            rect_x = [-x_phys, x_phys, x_phys, -x_phys, -x_phys]
            rect_y = [-x_phys, -x_phys, x_phys, x_phys, -x_phys]
            ax.plot(rect_x, rect_y, 'k--', linewidth=1.5, label='PML boundary')
            ax.set_title('2D Wave (early)')
            ax.legend(fontsize=8)
            
            ax = axes[1, 2]
            vmax2 = max(np.max(np.abs(u_hist_2d[-1])), 1e-10)
            ax.imshow(u_hist_2d[-1], cmap='RdBu',
                     vmin=-vmax2, vmax=vmax2,
                     extent=[-Lx_2d, Lx_2d, -Ly_2d, Ly_2d])
            ax.plot(rect_x, rect_y, 'k--', linewidth=1.5, label='PML boundary')
            ax.set_title('2D Wave (late)')
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig("pml_absorbing_bc.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
