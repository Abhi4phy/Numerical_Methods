"""
Split-Operator Method for the Schrödinger Equation
====================================================
Solving the time-dependent Schrödinger equation:

    iℏ ∂ψ/∂t = [-ℏ²/(2m) ∇² + V(x)] ψ = [T + V] ψ

**The Split-Operator Idea:**
The formal solution is ψ(t+Δt) = exp(-i(T+V)Δt/ℏ) ψ(t).
Since T and V don't commute, we split:

    exp(-i(T+V)Δt) ≈ exp(-iV·Δt/2) · exp(-iT·Δt) · exp(-iV·Δt/2) + O(Δt³)

Key insight: exp(-iT·Δt) is DIAGONAL in momentum space (Fourier),
             exp(-iV·Δt) is DIAGONAL in position space.

Algorithm:
    1. Multiply ψ(x) by exp(-iV(x)Δt/2)        [position space]
    2. FFT → ψ̃(k)
    3. Multiply ψ̃(k) by exp(-iℏk²Δt/(2m))      [momentum space]
    4. IFFT → ψ(x)
    5. Multiply ψ(x) by exp(-iV(x)Δt/2)        [position space]

**Properties:**
- Unitary (conserves norm exactly)
- 2nd order in time, spectral accuracy in space
- O(N log N) per step via FFT
- Symplectic (for real-time evolution)

**Also includes:**
- Crank-Nicolson: implicit, unconditionally stable
- Imaginary time evolution: finds ground state

Physics: quantum tunneling, wave packet dynamics, bound states.

Where to start:
━━━━━━━━━━━━━━
Run the Gaussian wave packet scattering off a barrier — see tunneling!
Prerequisite: FFT (07/fast_fourier_transform.py), finite_difference_method.py
"""

import numpy as np


class QuantumSolver1D:
    """
    1D time-dependent Schrödinger equation solver.
    
    Uses the split-operator FFT method for time evolution.
    """
    
    def __init__(self, x_min, x_max, N, mass=1.0, hbar=1.0):
        """
        Parameters
        ----------
        x_min, x_max : float
            Spatial domain
        N : int
            Number of grid points (should be power of 2 for FFT)
        mass : float
            Particle mass
        hbar : float
            Reduced Planck constant
        """
        self.N = N
        self.mass = mass
        self.hbar = hbar
        
        # Position grid
        self.dx = (x_max - x_min) / N
        self.x = np.linspace(x_min, x_max - self.dx, N)
        
        # Momentum grid (FFT frequencies)
        self.dk = 2 * np.pi / (N * self.dx)
        self.k = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        
        # Kinetic energy in k-space: ℏ²k²/(2m)
        self.T_k = self.hbar**2 * self.k**2 / (2 * self.mass)
    
    def gaussian_wavepacket(self, x0, sigma, k0):
        """
        Create a Gaussian wave packet:
        ψ(x) = (2πσ²)^{-1/4} exp(-(x-x₀)²/(4σ²) + ik₀x)
        
        Parameters
        ----------
        x0 : float
            Center position
        sigma : float
            Width
        k0 : float
            Central momentum (ℏk₀)
        """
        psi = ((2 * np.pi * sigma**2)**(-0.25) 
               * np.exp(-(self.x - x0)**2 / (4 * sigma**2))
               * np.exp(1j * k0 * self.x))
        return psi
    
    def split_operator_step(self, psi, V, dt):
        """
        One step of the split-operator method.
        
        ψ(t+dt) = exp(-iVdt/2) · FFT⁻¹[exp(-iTdt/ℏ) · FFT[exp(-iVdt/2) · ψ]]
        
        2nd order (Strang splitting), unitary, O(N log N).
        """
        # Half-step in V (position space)
        psi = psi * np.exp(-1j * V * dt / (2 * self.hbar))
        
        # Full step in T (momentum space)
        psi_k = np.fft.fft(psi)
        psi_k = psi_k * np.exp(-1j * self.T_k * dt / self.hbar)
        psi = np.fft.ifft(psi_k)
        
        # Half-step in V (position space)
        psi = psi * np.exp(-1j * V * dt / (2 * self.hbar))
        
        return psi
    
    def evolve(self, psi0, V, dt, n_steps, store_every=1):
        """
        Time evolution using split-operator method.
        
        Returns snapshots of the wavefunction.
        """
        psi = psi0.copy()
        snapshots = [psi.copy()]
        times = [0.0]
        
        for step in range(1, n_steps + 1):
            psi = self.split_operator_step(psi, V, dt)
            
            if step % store_every == 0:
                snapshots.append(psi.copy())
                times.append(step * dt)
        
        return np.array(times), np.array(snapshots)
    
    def imaginary_time_evolution(self, V, dt_imag, n_steps, psi0=None):
        """
        Imaginary time evolution: t → -iτ.
        
        Projects out the ground state: ψ → exp(-Hτ/ℏ)ψ → ground state.
        We renormalize after each step to prevent exponential decay.
        
        Parameters
        ----------
        V : array
            Potential
        dt_imag : float
            Imaginary time step
        n_steps : int
            Number of steps
        psi0 : array, optional
            Initial guess (random if None)
        """
        if psi0 is None:
            psi0 = np.random.randn(self.N) + 0.0j
        
        psi = psi0.copy()
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
        
        energies = []
        
        for step in range(n_steps):
            # Split-operator with τ instead of it
            # exp(-Vτ/2) · exp(-Tτ) · exp(-Vτ/2)
            psi = psi * np.exp(-V * dt_imag / (2 * self.hbar))
            
            psi_k = np.fft.fft(psi)
            psi_k = psi_k * np.exp(-self.T_k * dt_imag / self.hbar)
            psi = np.fft.ifft(psi_k)
            
            psi = psi * np.exp(-V * dt_imag / (2 * self.hbar))
            
            # Renormalize
            norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
            psi /= norm
            
            # Compute energy expectation value
            E = self.compute_energy(psi, V)
            energies.append(E)
        
        return psi, np.array(energies)
    
    def compute_energy(self, psi, V):
        """
        Compute ⟨ψ|H|ψ⟩ = ⟨T⟩ + ⟨V⟩.
        """
        # Kinetic energy in k-space
        psi_k = np.fft.fft(psi)
        KE = np.real(np.sum(np.conj(psi_k) * self.T_k * psi_k)) / self.N * self.dx
        
        # Potential energy in x-space
        PE = np.real(np.sum(np.conj(psi) * V * psi)) * self.dx
        
        return KE + PE
    
    def compute_observables(self, psi):
        """Compute ⟨x⟩, ⟨p⟩, σ_x, σ_p."""
        prob = np.abs(psi)**2 * self.dx
        
        x_mean = np.sum(self.x * prob)
        x2_mean = np.sum(self.x**2 * prob)
        sigma_x = np.sqrt(x2_mean - x_mean**2)
        
        psi_k = np.fft.fft(psi)
        prob_k = np.abs(psi_k)**2 / self.N * self.dx
        p_mean = np.sum(self.hbar * self.k * prob_k)
        
        return x_mean, p_mean, sigma_x
    
    def norm(self, psi):
        """Compute ∫|ψ|² dx."""
        return np.sum(np.abs(psi)**2) * self.dx


def crank_nicolson_step(psi, V, dx, dt, hbar=1.0, mass=1.0):
    """
    Crank-Nicolson method for 1D Schrödinger equation.
    
    Implicit, unconditionally stable, unitary, 2nd order.
    
    (1 + iHΔt/2ℏ) ψ^{n+1} = (1 - iHΔt/2ℏ) ψ^n
    
    Uses tridiagonal solver.
    """
    N = len(psi)
    r = 1j * hbar * dt / (4 * mass * dx**2)
    
    # Build tridiagonal matrices
    # Diagonal of (1 + iHΔt/2)
    d_lhs = (1 + 2*r) * np.ones(N) + 1j * V * dt / (2*hbar)
    # Off-diagonal
    o_lhs = -r * np.ones(N - 1)
    
    # RHS: (1 - iHΔt/2) ψ
    d_rhs = (1 - 2*r) * np.ones(N) - 1j * V * dt / (2*hbar)
    
    # Compute RHS vector
    rhs = d_rhs * psi
    rhs[:-1] += r * psi[1:]
    rhs[1:] += r * psi[:-1]
    
    # Solve tridiagonal system (Thomas algorithm)
    psi_new = tridiag_solve(o_lhs, d_lhs, o_lhs, rhs)
    
    return psi_new


def tridiag_solve(a, b, c, d):
    """Thomas algorithm for tridiagonal system."""
    n = len(b)
    c_prime = np.zeros(n, dtype=complex)
    d_prime = np.zeros(n, dtype=complex)
    
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n):
        m = a[i-1] / (b[i] - a[i-1] * c_prime[i-1]) if i < len(a) else 0
        if i < n - 1:
            c_prime[i] = c[i] / (b[i] - a[i-1] * c_prime[i-1])
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / (b[i] - a[i-1] * c_prime[i-1])
    
    x = np.zeros(n, dtype=complex)
    x[-1] = d_prime[-1]
    
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SPLIT-OPERATOR SCHRÖDINGER EQUATION DEMO")
    print("=" * 60)

    # --- 1. Free particle wave packet ---
    print("\n--- Free Particle Gaussian Wave Packet ---")
    solver = QuantumSolver1D(x_min=-30, x_max=30, N=1024)
    
    psi0 = solver.gaussian_wavepacket(x0=-5.0, sigma=1.0, k0=3.0)
    V_free = np.zeros(solver.N)
    
    dt = 0.01
    n_steps = 500
    times, snapshots = solver.evolve(psi0, V_free, dt, n_steps, store_every=50)
    
    print(f"  Initial norm:   {solver.norm(snapshots[0]):.10f}")
    print(f"  Final norm:     {solver.norm(snapshots[-1]):.10f}")
    print(f"  Norm conserved: {abs(solver.norm(snapshots[-1]) - 1.0):.2e}")
    
    # Track wave packet center
    x0_obs, p0_obs, sig0 = solver.compute_observables(snapshots[0])
    xf_obs, pf_obs, sigf = solver.compute_observables(snapshots[-1])
    print(f"  ⟨x⟩: {x0_obs:.2f} → {xf_obs:.2f} (expected: {-5.0 + 3.0*times[-1]:.2f})")
    print(f"  σ_x: {sig0:.3f} → {sigf:.3f} (spreading)")

    # --- 2. Quantum tunneling ---
    print("\n--- Quantum Tunneling through Barrier ---")
    solver2 = QuantumSolver1D(x_min=-30, x_max=30, N=2048)
    
    # Rectangular barrier
    barrier_height = 5.0
    barrier_width = 1.0
    V_barrier = np.where(np.abs(solver2.x) < barrier_width/2, barrier_height, 0.0)
    
    # Wave packet with energy E ≈ k₀²/2 < barrier
    k0 = 2.5  # E ≈ 3.125 < 5.0 (barrier)
    psi_tunnel = solver2.gaussian_wavepacket(x0=-8.0, sigma=1.5, k0=k0)
    
    times_t, snaps_t = solver2.evolve(psi_tunnel, V_barrier, 0.005, 3000, store_every=300)
    
    # Compute transmission probability
    transmitted = np.sum(np.abs(snaps_t[-1, solver2.x > 2])**2) * solver2.dx
    reflected = np.sum(np.abs(snaps_t[-1, solver2.x < -2])**2) * solver2.dx
    print(f"  Barrier: V₀={barrier_height}, width={barrier_width}")
    print(f"  Particle energy: E ≈ {k0**2/2:.2f}")
    print(f"  Transmission: {transmitted:.4f}")
    print(f"  Reflection:   {reflected:.4f}")
    print(f"  Total:        {transmitted + reflected:.6f}")

    # --- 3. Harmonic oscillator ground state ---
    print("\n--- Imaginary Time: Harmonic Oscillator Ground State ---")
    solver3 = QuantumSolver1D(x_min=-10, x_max=10, N=512)
    
    omega = 1.0
    V_ho = 0.5 * omega**2 * solver3.x**2
    
    psi_gs, energies_gs = solver3.imaginary_time_evolution(
        V_ho, dt_imag=0.01, n_steps=5000)
    
    # Exact ground state: E₀ = ℏω/2, ψ₀ = (mω/πℏ)^{1/4} exp(-mωx²/2ℏ)
    E_exact = 0.5 * solver3.hbar * omega
    psi_exact = (omega / np.pi)**0.25 * np.exp(-omega * solver3.x**2 / 2)
    
    print(f"  Ground state energy:")
    print(f"    Computed: {energies_gs[-1]:.8f}")
    print(f"    Exact:    {E_exact:.8f}")
    print(f"    Error:    {abs(energies_gs[-1] - E_exact):.2e}")
    
    # Phase alignment for comparison
    psi_gs_real = np.real(psi_gs)
    if psi_gs_real[len(psi_gs_real)//2] < 0:
        psi_gs_real = -psi_gs_real
    
    overlap = abs(np.sum(np.conj(psi_exact / np.sqrt(np.sum(psi_exact**2) * solver3.dx)) 
                         * psi_gs) * solver3.dx)
    print(f"    |⟨ψ_exact|ψ_computed⟩| = {overlap:.8f}")

    # --- 4. Double well ---
    print("\n--- Double-Well Potential ---")
    solver4 = QuantumSolver1D(x_min=-10, x_max=10, N=512)
    
    V_dw = 0.5 * (solver4.x**2 - 3)**2  # Double well
    
    # Find ground state
    psi_dw, E_dw = solver4.imaginary_time_evolution(V_dw, 0.005, 10000)
    print(f"  Double-well ground state E₀ ≈ {E_dw[-1]:.6f}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Free particle snapshots
        ax = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))
        for i, (t_i, snap) in enumerate(zip(times, snapshots)):
            ax.plot(solver.x, np.abs(snap)**2, color=colors[i], 
                   label=f't={t_i:.1f}')
        ax.set_xlabel('x')
        ax.set_ylabel('|ψ|²')
        ax.set_title('Free Particle Wave Packet')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # Tunneling
        ax = axes[0, 1]
        ax.fill_between(solver2.x, 0, V_barrier/barrier_height * 0.5, 
                       alpha=0.3, color='gray', label='Barrier')
        colors_t = plt.cm.plasma(np.linspace(0, 1, len(snaps_t)))
        for i, (t_i, snap) in enumerate(zip(times_t, snaps_t)):
            ax.plot(solver2.x, np.abs(snap)**2, color=colors_t[i],
                   linewidth=1, label=f't={t_i:.1f}' if i % 3 == 0 else '')
        ax.set_xlabel('x')
        ax.set_ylabel('|ψ|²')
        ax.set_title(f'Quantum Tunneling (E={k0**2/2:.1f} < V₀={barrier_height})')
        ax.set_xlim(-20, 20)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # Harmonic oscillator ground state
        ax = axes[0, 2]
        ax.plot(solver3.x, np.abs(psi_exact)**2, 'k-', linewidth=2, 
               label='Exact ψ₀')
        ax.plot(solver3.x, np.abs(psi_gs)**2, 'r--', linewidth=1.5,
               label='Computed')
        ax.plot(solver3.x, V_ho / V_ho.max() * np.max(np.abs(psi_exact)**2), 
               'b:', alpha=0.5, label='V(x)')
        ax.set_xlabel('x')
        ax.set_ylabel('|ψ|²')
        ax.set_title('HO Ground State (Imaginary Time)')
        ax.set_xlim(-5, 5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Energy convergence
        ax = axes[1, 0]
        ax.semilogy(np.abs(energies_gs - E_exact))
        ax.set_xlabel('Imaginary time step')
        ax.set_ylabel('|E - E_exact|')
        ax.set_title('Energy Convergence')
        ax.grid(True, alpha=0.3)
        
        # Double well
        ax = axes[1, 1]
        ax.plot(solver4.x, V_dw, 'b-', linewidth=2, label='V(x)')
        ax.plot(solver4.x, np.abs(psi_dw)**2 * 50 + E_dw[-1], 'r-', 
               linewidth=2, label='|ψ₀|² (scaled)')
        ax.axhline(E_dw[-1], color='r', linestyle='--', alpha=0.5,
                   label=f'E₀ = {E_dw[-1]:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('Energy / |ψ|²')
        ax.set_title('Double-Well Ground State')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-1, 10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Norm conservation
        ax = axes[1, 2]
        norms = [solver.norm(s) for s in snapshots]
        times_t_norms = [solver2.norm(s) for s in snaps_t]
        ax.plot(times, np.abs(np.array(norms) - 1.0), 'b-o', label='Free particle')
        ax.plot(times_t, np.abs(np.array(times_t_norms) - 1.0), 'r-s', label='Tunneling')
        ax.set_xlabel('Time')
        ax.set_ylabel('|‖ψ‖ - 1|')
        ax.set_title('Norm Conservation (Split-Operator)')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("split_operator_schrodinger.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
