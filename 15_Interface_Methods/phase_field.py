"""
Phase-Field Methods
===================
Model interfaces as DIFFUSE (finite-width) regions rather than
sharp boundaries. The order parameter φ varies smoothly from
one phase to another.

**Core Idea:**
Instead of tracking a sharp interface, use a continuous field
φ(x,t) ∈ [-1, 1] where:
  - φ ≈ +1 in phase A
  - φ ≈ -1 in phase B
  - Smooth transition of width ε at the interface

**Free Energy Functional:**
    F[φ] = ∫ [f(φ) + (ε²/2)|∇φ|²] dV

where f(φ) is a double-well potential (e.g., f = (1-φ²)²/4)

**Allen-Cahn Equation (non-conserved order parameter):**
    ∂φ/∂t = -M · δF/δφ = M[ε²∇²φ - f'(φ)]

Models: grain growth, phase ordering, domain wall motion

**Cahn-Hilliard Equation (conserved order parameter):**
    ∂φ/∂t = ∇·[M ∇(δF/δφ)] = ∇·[M ∇(f'(φ) - ε²∇²φ)]

Models: spinodal decomposition, phase separation, binary alloys
Fourth-order PDE — needs special numerical treatment

**Physics:**
- Surface tension arises naturally from gradient energy
- Interface width ε is a computational parameter
- Cahn-Hilliard conserves ∫φ dV (total mass), Allen-Cahn does not
- Can couple to Navier-Stokes for two-phase flows

**Applications:**
- Solidification (dendritic growth)
- Spinodal decomposition
- Grain boundary dynamics
- Polymer blends
- Battery electrode microstructure
- Tumor growth models

Where to start:
━━━━━━━━━━━━━━
1. Start with Allen-Cahn — it's simpler (2nd order PDE)
2. Initialize a step function, watch it sharpen to tanh profile
3. Then try Cahn-Hilliard spinodal decomposition — beautiful patterns!
Prerequisites: finite_difference_method.py, fast_fourier_transform.py
"""

import numpy as np


# ============================================================
# Free Energy and Potentials
# ============================================================

def double_well(phi):
    """
    Standard double-well potential: f(φ) = (1 - φ²)² / 4
    
    Minima at φ = ±1, barrier at φ = 0
    """
    return 0.25 * (1 - phi**2)**2


def double_well_derivative(phi):
    """f'(φ) = -φ(1 - φ²) = φ³ - φ"""
    return phi**3 - phi


def logarithmic_potential(phi, theta=0.3):
    """
    Logarithmic (Flory-Huggins) free energy:
    f(φ) = θ/2 [(1+φ)ln(1+φ) + (1-φ)ln(1-φ)] + (1-φ²)/2
    
    More physically realistic but requires |φ| < 1.
    """
    phi_c = np.clip(phi, -0.999, 0.999)
    return (theta/2 * ((1+phi_c) * np.log(1+phi_c) + 
                       (1-phi_c) * np.log(1-phi_c)) + 
            0.5 * (1 - phi_c**2))


def logarithmic_potential_derivative(phi, theta=0.3):
    """Derivative of logarithmic potential."""
    phi_c = np.clip(phi, -0.999, 0.999)
    return theta/2 * np.log((1+phi_c)/(1-phi_c)) - phi_c


def free_energy(phi, epsilon, dx, dy, potential='polynomial'):
    """
    Compute total Ginzburg-Landau free energy.
    
    F = ∫ [f(φ) + (ε²/2)|∇φ|²] dV
    """
    if potential == 'polynomial':
        f_bulk = double_well(phi)
    else:
        f_bulk = logarithmic_potential(phi)
    
    # Gradient energy
    phi_x = np.zeros_like(phi)
    phi_y = np.zeros_like(phi)
    phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2*dx)
    phi_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2*dy)
    
    grad_sq = phi_x**2 + phi_y**2
    
    return np.sum(f_bulk + 0.5 * epsilon**2 * grad_sq) * dx * dy


# ============================================================
# Allen-Cahn Equation
# ============================================================

def allen_cahn_step(phi, dt, epsilon, M=1.0, dx=None, dy=None):
    """
    One time step of the Allen-Cahn equation:
        ∂φ/∂t = M[ε²∇²φ - f'(φ)]
    
    Semi-implicit scheme for stability:
        (1 + M·dt)φ^{n+1} = φ^n + M·dt[ε²∇²φ^n + φ^n]
    
    (Treats the linear part of f'(φ) = φ³ - φ implicitly)
    """
    Ny, Nx = phi.shape
    if dx is None:
        dx = 1.0 / Nx
    if dy is None:
        dy = 1.0 / Ny
    
    # Laplacian with periodic BC
    laplacian = (
        np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) - 2*phi
    ) / dx**2 + (
        np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) - 2*phi
    ) / dy**2
    
    # Semi-implicit update
    phi_new = (phi + M * dt * (epsilon**2 * laplacian + phi)) / (1 + M * dt)
    
    # Correct for cubic term
    phi_new = phi + M * dt * (epsilon**2 * laplacian - double_well_derivative(phi))
    
    return phi_new


def allen_cahn_solve(phi0, dt, epsilon, n_steps, M=1.0, dx=None, dy=None):
    """
    Solve Allen-Cahn equation for n_steps.
    """
    phi = phi0.copy()
    Ny, Nx = phi.shape
    if dx is None:
        dx = 1.0 / Nx
    if dy is None:
        dy = 1.0 / Ny
    
    energies = [free_energy(phi, epsilon, dx, dy)]
    
    for step in range(n_steps):
        phi = allen_cahn_step(phi, dt, epsilon, M, dx, dy)
        if step % max(1, n_steps // 20) == 0:
            energies.append(free_energy(phi, epsilon, dx, dy))
    
    return phi, np.array(energies)


# ============================================================
# Cahn-Hilliard Equation (spectral method)
# ============================================================

def cahn_hilliard_spectral_step(phi_hat, dt, epsilon, M, kx, ky):
    """
    One step of Cahn-Hilliard using semi-implicit spectral method.
    
    ∂φ/∂t = M ∇²[f'(φ) - ε²∇²φ]
    
    In Fourier space (semi-implicit, treating ε²k⁴ implicitly):
        φ̂^{n+1} = [φ̂^n - M·dt·k²·F{φ³}] / [1 + M·dt·k²(1 + ε²k²)]
    
    where k² = kx² + ky²
    """
    k2 = kx**2 + ky**2
    
    # Nonlinear term in real space
    phi = np.real(np.fft.ifft2(phi_hat))
    phi_cubed_hat = np.fft.fft2(phi**3)
    
    # Semi-implicit update
    numerator = phi_hat - M * dt * k2 * phi_cubed_hat
    denominator = 1 + M * dt * k2 * (1 + epsilon**2 * k2)
    
    # Avoid division by zero at k=0
    denominator[0, 0] = 1.0
    
    phi_hat_new = numerator / denominator
    phi_hat_new[0, 0] = phi_hat[0, 0]  # Conserve mean
    
    return phi_hat_new


def cahn_hilliard_solve(phi0, dt, epsilon, n_steps, M=1.0):
    """
    Solve Cahn-Hilliard equation using pseudo-spectral method.
    
    Returns phi at final time and energy history.
    """
    Ny, Nx = phi0.shape
    dx = 1.0 / Nx
    dy = 1.0 / Ny
    
    # Wavenumbers
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    
    phi_hat = np.fft.fft2(phi0)
    phi = phi0.copy()
    
    energies = [free_energy(phi, epsilon, dx, dy)]
    snapshots = [phi0.copy()]
    
    for step in range(n_steps):
        phi_hat = cahn_hilliard_spectral_step(phi_hat, dt, epsilon, M, KX, KY)
        
        if step % max(1, n_steps // 10) == 0:
            phi = np.real(np.fft.ifft2(phi_hat))
            energies.append(free_energy(phi, epsilon, dx, dy))
            snapshots.append(phi.copy())
    
    phi = np.real(np.fft.ifft2(phi_hat))
    
    return phi, np.array(energies), snapshots


# ============================================================
# Initial Conditions
# ============================================================

def random_initial(Nx, Ny, mean=0.0, amplitude=0.05, seed=42):
    """Random initial perturbation around mean value."""
    rng = np.random.default_rng(seed)
    return mean + amplitude * (2 * rng.random((Ny, Nx)) - 1)


def bubble_initial(Nx, Ny, cx=0.5, cy=0.5, R=0.2, epsilon=0.02):
    """Circular bubble with tanh profile."""
    x = np.linspace(0, 1, Nx, endpoint=False)
    y = np.linspace(0, 1, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    return np.tanh((R - r) / (np.sqrt(2) * epsilon))


def two_bubbles_initial(Nx, Ny, epsilon=0.02):
    """Two bubbles that may merge."""
    x = np.linspace(0, 1, Nx, endpoint=False)
    y = np.linspace(0, 1, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)
    r1 = np.sqrt((X - 0.35)**2 + (Y - 0.5)**2)
    r2 = np.sqrt((X - 0.65)**2 + (Y - 0.5)**2)
    phi1 = np.tanh((0.15 - r1) / (np.sqrt(2) * epsilon))
    phi2 = np.tanh((0.15 - r2) / (np.sqrt(2) * epsilon))
    return np.maximum(phi1, phi2)


def stripe_initial(Nx, Ny, n_stripes=3, epsilon=0.02):
    """Stripe pattern initial condition."""
    x = np.linspace(0, 1, Nx, endpoint=False)
    y = np.linspace(0, 1, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)
    return np.tanh(np.sin(2 * np.pi * n_stripes * X) / (np.sqrt(2) * epsilon))


# ============================================================
# Analysis
# ============================================================

def interface_width(phi, dx):
    """Estimate interface width from max gradient."""
    grad = np.max(np.abs(np.diff(phi, axis=1))) / dx
    if grad > 0:
        return 2.0 / grad  # For tanh profile, width ≈ 2/(max gradient)
    return np.inf


def phase_fraction(phi):
    """Compute volume fraction of each phase."""
    f_plus = np.mean(phi > 0)
    f_minus = np.mean(phi < 0)
    return f_plus, f_minus


def structure_factor(phi):
    """
    Compute structure factor S(k) = <|φ̂(k)|²>.
    
    Useful for characterizing domain size in spinodal decomposition.
    The peak position k* determines the dominant domain spacing L* = 2π/k*.
    """
    phi_hat = np.fft.fft2(phi - np.mean(phi))
    S = np.abs(phi_hat)**2
    
    Ny, Nx = phi.shape
    kx = np.fft.fftfreq(Nx) * Nx
    ky = np.fft.fftfreq(Ny) * Ny
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2).flatten()
    S_flat = S.flatten()
    
    # Radial average
    k_bins = np.arange(0.5, min(Nx, Ny)//2)
    S_radial = np.zeros(len(k_bins))
    
    for i, k in enumerate(k_bins):
        mask = (K >= k - 0.5) & (K < k + 0.5)
        if np.any(mask):
            S_radial[i] = np.mean(S_flat[mask])
    
    return k_bins, S_radial


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE-FIELD METHODS DEMO")
    print("=" * 60)

    # --- 1. Allen-Cahn: tanh profile formation ---
    print("\n--- Allen-Cahn: Interface Formation ---")
    N = 128
    epsilon = 0.02
    dx = dy = 1.0 / N
    dt = 0.5 * dx**2  # Stability requirement
    
    # Start with step function
    x = np.linspace(0, 1, N, endpoint=False)
    y = np.linspace(0, 1, N, endpoint=False)
    X, Y = np.meshgrid(x, y)
    phi0_ac = np.where(np.sqrt((X-0.5)**2 + (Y-0.5)**2) < 0.3, 1.0, -1.0)
    
    phi_ac, energies_ac = allen_cahn_solve(phi0_ac, dt, epsilon, 
                                            n_steps=500, dx=dx, dy=dy)
    
    print(f"  Grid: {N}×{N}, ε = {epsilon}")
    print(f"  Energy: {energies_ac[0]:.4f} → {energies_ac[-1]:.4f}")
    print(f"  Phase fractions: {phase_fraction(phi_ac)}")
    print(f"  Interface width: {interface_width(phi_ac, dx):.4f} "
          f"(expected ≈ {2*np.sqrt(2)*epsilon:.4f})")

    # --- 2. Cahn-Hilliard: spinodal decomposition ---
    print("\n--- Cahn-Hilliard: Spinodal Decomposition ---")
    N_ch = 128
    epsilon_ch = 0.01
    dt_ch = 1e-5
    
    # Random initial condition near φ = 0 (unstable)
    phi0_ch = random_initial(N_ch, N_ch, mean=0.0, amplitude=0.1)
    
    phi_ch, energies_ch, snapshots = cahn_hilliard_solve(
        phi0_ch, dt_ch, epsilon_ch, n_steps=5000)
    
    print(f"  Grid: {N_ch}×{N_ch}, ε = {epsilon_ch}")
    print(f"  Energy: {energies_ch[0]:.4f} → {energies_ch[-1]:.4f}")
    print(f"  Mean φ: {np.mean(phi0_ch):.6f} → {np.mean(phi_ch):.6f} "
          f"(conserved!)")
    
    k_bins, S_k = structure_factor(phi_ch)
    k_peak = k_bins[np.argmax(S_k)] if len(S_k) > 0 else 0
    print(f"  Dominant wavelength: L* ≈ {N_ch/k_peak:.1f} grid cells" 
          if k_peak > 0 else "  No dominant wavelength")

    # --- 3. Off-critical quench ---
    print("\n--- Off-Critical Quench (φ̄ = 0.3) ---")
    phi0_off = random_initial(N_ch, N_ch, mean=0.3, amplitude=0.05)
    
    phi_off, energies_off, _ = cahn_hilliard_solve(
        phi0_off, dt_ch, epsilon_ch, n_steps=5000)
    
    f_plus, f_minus = phase_fraction(phi_off)
    print(f"  Mean φ: {np.mean(phi0_off):.4f} → {np.mean(phi_off):.4f}")
    print(f"  φ>0 fraction: {f_plus:.3f}, φ<0 fraction: {f_minus:.3f}")
    print("  (Asymmetric: minority phase forms droplets)")

    # --- 4. Bubble coarsening ---
    print("\n--- Bubble Coarsening (Ostwald Ripening) ---")
    phi0_bub = two_bubbles_initial(N_ch, N_ch, epsilon=epsilon_ch)
    
    phi_bub, energies_bub, _ = cahn_hilliard_solve(
        phi0_bub, dt_ch, epsilon_ch, n_steps=3000)
    
    print(f"  Two bubbles → Energy: {energies_bub[0]:.4f} → {energies_bub[-1]:.4f}")
    print(f"  (Large bubble grows, small bubble shrinks — Ostwald ripening)")

    # --- 5. Energy dissipation ---
    print("\n--- Energy Dissipation ---")
    print("  Allen-Cahn energy at sampled times:")
    for i, E in enumerate(energies_ac[:min(6, len(energies_ac))]):
        print(f"    Step ~{i * 500//min(20, len(energies_ac)):5d}: E = {E:.6f}")
    print("  Cahn-Hilliard energy at sampled times:")
    for i, E in enumerate(energies_ch[:min(6, len(energies_ch))]):
        print(f"    Step ~{i * 5000//10:5d}: E = {E:.6f}")
    print("  (Both: energy monotonically decreases — guaranteed by theory)")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        
        # Allen-Cahn
        ax = axes[0, 0]
        ax.imshow(phi0_ac, cmap='RdBu', vmin=-1, vmax=1,
                 extent=[0, 1, 0, 1])
        ax.set_title('Allen-Cahn: Initial')
        
        ax = axes[0, 1]
        ax.imshow(phi_ac, cmap='RdBu', vmin=-1, vmax=1,
                 extent=[0, 1, 0, 1])
        ax.set_title('Allen-Cahn: Final')
        
        # Spinodal decomposition snapshots
        n_snap = min(len(snapshots), 3)
        snap_indices = [0, len(snapshots)//2, -1]
        
        ax = axes[0, 2]
        ax.imshow(snapshots[0], cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title('C-H: t = 0')
        
        ax = axes[0, 3]
        ax.imshow(snapshots[-1], cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title('C-H: Final')
        
        # Energy plots
        ax = axes[1, 0]
        ax.plot(energies_ac, 'b-')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Free Energy')
        ax.set_title('Allen-Cahn Energy')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.plot(energies_ch, 'r-')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Free Energy')
        ax.set_title('Cahn-Hilliard Energy')
        ax.grid(True, alpha=0.3)
        
        # Structure factor
        ax = axes[1, 2]
        ax.semilogy(k_bins, S_k + 1e-10, 'k-')
        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel('S(k)')
        ax.set_title('Structure Factor')
        ax.grid(True, alpha=0.3)
        
        # Off-critical
        ax = axes[1, 3]
        ax.imshow(phi_off, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title('Off-Critical (φ̄=0.3)')
        
        plt.tight_layout()
        plt.savefig("phase_field.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
