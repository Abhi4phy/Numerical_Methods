"""
Symplectic Integrators
=======================
Time-integration methods that exactly preserve the symplectic structure
of Hamiltonian systems (and thus conserve energy on average).

**Why Symplectic?**
Standard RK4 introduces a systematic drift in energy over long times.
Symplectic integrators keep phase-space volume conserved — critical for:
  - Planetary / orbital mechanics
  - Molecular dynamics
  - Charged particle tracking (Boris pusher!)
  - Statistical mechanics sampling

**The Key Idea:**
For Hamiltonian H(q, p) = T(p) + V(q):
  ṗ = -∂V/∂q (force)      ←  "kick" step
  q̇ = ∂T/∂p (velocity)    ←  "drift" step

Splitting the Hamiltonian into kinetic + potential parts and alternating
exact sub-steps gives symplectic methods automatically.

**Methods Implemented:**

1. **Symplectic Euler** (1st order):
   p_{n+1} = p_n + dt·F(q_n)
   q_{n+1} = q_n + dt·p_{n+1}/m

2. **Störmer-Verlet / Leapfrog** (2nd order):
   p_{n+1/2} = p_n + (dt/2)·F(q_n)
   q_{n+1}   = q_n + dt·p_{n+1/2}/m
   p_{n+1}   = p_{n+1/2} + (dt/2)·F(q_{n+1})

3. **Yoshida 4th order** (4th order, 3 stages):
   Uses weighted Verlet sub-steps with special coefficients.

4. **Forest-Ruth** (4th order, equivalent to Yoshida):
   q₁ = q + θ·dt/2·v
   v₁ = v + θ·dt·a(q₁)
   ...

Physics: N-body, molecular dynamics, plasma simulation, accelerator physics.

Where to start:
━━━━━━━━━━━━━━
Run the harmonic oscillator example. Compare energy drift of RK4 vs
Verlet over 10000 periods — the difference is dramatic.
prerequisite: Euler method (02_Differential_Equations/euler_method.py)
"""

import numpy as np


def symplectic_euler(force, q0, p0, m, dt, n_steps):
    """
    Symplectic (semi-implicit) Euler method.
    
    First-order symplectic integrator.
    Uses the NEW momentum to update position → preserves phase space.
    
    Parameters
    ----------
    force : callable
        force(q) → np.array, the force = -dV/dq
    q0, p0 : np.array
        Initial position and momentum
    m : float
        Mass
    dt : float
        Time step
    n_steps : int
        Number of steps
    
    Returns
    -------
    t, q, p : arrays
        Time, position, and momentum histories
    """
    q = np.zeros((n_steps + 1, len(q0)))
    p = np.zeros((n_steps + 1, len(p0)))
    q[0] = q0
    p[0] = p0
    t = np.arange(n_steps + 1) * dt
    
    for i in range(n_steps):
        p[i+1] = p[i] + dt * force(q[i])           # Kick
        q[i+1] = q[i] + dt * p[i+1] / m            # Drift (using NEW p)
    
    return t, q, p


def velocity_verlet(force, q0, p0, m, dt, n_steps):
    """
    Störmer-Verlet (velocity form) — 2nd order symplectic.
    
    The workhorse of molecular dynamics and orbital mechanics.
    
    Algorithm:
        1. p_{n+1/2} = p_n + (dt/2) · F(q_n)         (half-kick)
        2. q_{n+1}   = q_n + dt · p_{n+1/2} / m       (full drift)
        3. p_{n+1}   = p_{n+1/2} + (dt/2) · F(q_{n+1}) (half-kick)
    
    Time-reversible and symplectic.
    """
    dim = len(q0)
    q = np.zeros((n_steps + 1, dim))
    p = np.zeros((n_steps + 1, dim))
    q[0] = q0
    p[0] = p0
    t = np.arange(n_steps + 1) * dt
    
    for i in range(n_steps):
        # Half-kick
        p_half = p[i] + 0.5 * dt * force(q[i])
        # Full drift
        q[i+1] = q[i] + dt * p_half / m
        # Half-kick
        p[i+1] = p_half + 0.5 * dt * force(q[i+1])
    
    return t, q, p


def leapfrog(force, q0, p0, m, dt, n_steps):
    """
    Leapfrog method — positions and momenta offset by half-step.
    
    Mathematically equivalent to velocity Verlet.
    
    Standard form for particle-in-cell codes:
        p at integer times, q at half-integer times.
    """
    dim = len(q0)
    q = np.zeros((n_steps + 1, dim))
    p = np.zeros((n_steps + 1, dim))
    q[0] = q0
    p[0] = p0
    t = np.arange(n_steps + 1) * dt
    
    # Initial half-step for momentum
    p_half = p[0] + 0.5 * dt * force(q[0])
    
    for i in range(n_steps):
        q[i+1] = q[i] + dt * p_half / m
        
        if i < n_steps - 1:
            p_half_new = p_half + dt * force(q[i+1])
            p[i+1] = p_half + 0.5 * dt * force(q[i+1])  # Sync to integer
            p_half = p_half_new
        else:
            p[i+1] = p_half + 0.5 * dt * force(q[i+1])
    
    return t, q, p


def yoshida4(force, q0, p0, m, dt, n_steps):
    """
    Yoshida 4th-order symplectic integrator.
    
    Composes 3 Verlet steps with special coefficients:
        c₁ = c₃ = 1 / (2 - 2^{1/3})
        c₂ = -2^{1/3} / (2 - 2^{1/3})
    
    4th order accuracy while remaining exactly symplectic.
    """
    # Yoshida coefficients
    cbrt2 = 2.0**(1.0/3.0)
    w0 = -cbrt2 / (2.0 - cbrt2)
    w1 = 1.0 / (2.0 - cbrt2)
    
    # Position and momentum sub-step coefficients
    c = np.array([w1/2, (w0+w1)/2, (w0+w1)/2, w1/2])
    d = np.array([w1, w0, w1, 0.0])
    
    dim = len(q0)
    q = np.zeros((n_steps + 1, dim))
    p = np.zeros((n_steps + 1, dim))
    q[0] = q0
    p[0] = p0
    t = np.arange(n_steps + 1) * dt
    
    for i in range(n_steps):
        qi = q[i].copy()
        pi = p[i].copy()
        
        for s in range(len(c)):
            qi = qi + c[s] * dt * pi / m
            if d[s] != 0:
                pi = pi + d[s] * dt * force(qi)
        
        q[i+1] = qi
        p[i+1] = pi
    
    return t, q, p


def rk4_step(deriv, state, t, dt):
    """Standard RK4 for comparison (NOT symplectic)."""
    k1 = deriv(state, t)
    k2 = deriv(state + 0.5*dt*k1, t + 0.5*dt)
    k3 = deriv(state + 0.5*dt*k2, t + 0.5*dt)
    k4 = deriv(state + dt*k3, t + dt)
    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


def compute_energy(q, p, m, V):
    """Total energy E = p²/(2m) + V(q) for each timestep."""
    KE = np.sum(p**2, axis=1) / (2*m)
    PE = np.array([V(qi) for qi in q])
    return KE + PE


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SYMPLECTIC INTEGRATORS DEMO")
    print("=" * 60)

    # === 1. Harmonic Oscillator ===
    print("\n--- Harmonic Oscillator ---")
    omega = 1.0
    m = 1.0
    force_ho = lambda q: -omega**2 * q
    V_ho = lambda q: 0.5 * omega**2 * np.sum(q**2)
    
    q0 = np.array([1.0])
    p0 = np.array([0.0])
    dt = 0.1
    n_periods = 100
    n_steps = int(2 * np.pi / omega / dt * n_periods)
    
    methods = {
        "Symplectic Euler": symplectic_euler,
        "Velocity Verlet":  velocity_verlet,
        "Leapfrog":        leapfrog,
        "Yoshida 4th":     yoshida4,
    }
    
    results = {}
    for name, method in methods.items():
        t, q, p = method(force_ho, q0, p0, m, dt, n_steps)
        E = compute_energy(q, p, m, V_ho)
        dE = np.abs(E - E[0])
        results[name] = (t, q, p, E)
        print(f"  {name:20s}: max |ΔE/E₀| = {np.max(dE)/E[0]:.3e}")
    
    # RK4 for comparison
    def deriv_ho(state, t):
        q, p = state[:1], state[1:]
        return np.array([p[0]/m, -omega**2 * q[0]])
    
    state = np.concatenate([q0, p0])
    E_rk4 = []
    q_rk4_list = []
    for i in range(n_steps):
        state = rk4_step(deriv_ho, state, i*dt, dt)
        E_rk4.append(0.5*state[1]**2/m + 0.5*omega**2*state[0]**2)
        q_rk4_list.append(state[0])
    dE_rk4 = np.abs(np.array(E_rk4) - E_rk4[0])
    print(f"  {'RK4 (non-sympl.)':20s}: max |ΔE/E₀| = {np.max(dE_rk4)/E_rk4[0]:.3e}")
    
    # === 2. Kepler Problem ===
    print("\n--- Kepler Problem (elliptical orbit) ---")
    GM = 1.0
    
    def force_kepler(q):
        r = np.sqrt(q[0]**2 + q[1]**2)
        return -GM * q / r**3
    
    def V_kepler(q):
        r = np.sqrt(q[0]**2 + q[1]**2)
        return -GM / r
    
    # Elliptical orbit (eccentricity 0.5)
    e = 0.5
    q0_kep = np.array([1.0 - e, 0.0])  # perihelion
    v0 = np.sqrt(GM * (1 + e) / (1 - e))
    p0_kep = np.array([0.0, v0])
    
    dt_kep = 0.01
    n_steps_kep = int(50 * 2 * np.pi / dt_kep)  # 50 orbits
    
    for name in ["Velocity Verlet", "Yoshida 4th"]:
        t, q, p = methods[name](force_kepler, q0_kep, p0_kep, m, dt_kep, n_steps_kep)
        E = compute_energy(q, p, m, V_kepler)
        L = q[:, 0] * p[:, 1] - q[:, 1] * p[:, 0]
        print(f"  {name}: ΔE/E₀ = {np.max(np.abs(E-E[0])/np.abs(E[0])):.3e}, "
              f"ΔL/L₀ = {np.max(np.abs(L-L[0])/np.abs(L[0])):.3e}")
    
    # === 3. Double Pendulum (chaotic) ===
    print("\n--- Double Pendulum (energy conservation in chaos) ---")

    def force_dp(q):
        """Simplified double spring force."""
        k = 1.0
        return -k * q  # Coupled harmonic
    
    q0_dp = np.array([1.0, 0.5])
    p0_dp = np.array([0.0, 0.3])
    
    t_dp, q_dp, p_dp = velocity_verlet(force_dp, q0_dp, p0_dp, 1.0, 0.01, 100000)
    V_dp = lambda q: 0.5 * np.sum(q**2)
    E_dp = compute_energy(q_dp, p_dp, 1.0, V_dp)
    print(f"  Verlet: 100k steps, max |ΔE| = {np.max(np.abs(E_dp - E_dp[0])):.3e}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Harmonic oscillator energy conservation
        ax = axes[0, 0]
        for name, (t, q, p, E) in results.items():
            ax.semilogy(t[:10000], np.abs(E[:10000] - E[0]) / E[0] + 1e-16, 
                       label=name, alpha=0.7)
        ax.semilogy(np.arange(1, len(E_rk4)+1)*dt, dE_rk4/E_rk4[0] + 1e-16,
                   label='RK4', alpha=0.7, linestyle='--')
        ax.set_xlabel('Time')
        ax.set_ylabel('|ΔE/E₀|')
        ax.set_title('Energy Conservation: Harmonic Oscillator')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Phase space (harmonic oscillator, Verlet)
        ax = axes[0, 1]
        t_v, q_v, p_v = methods["Velocity Verlet"](force_ho, q0, p0, m, 0.1, 5000)
        ax.plot(q_v[:, 0], p_v[:, 0], 'b-', alpha=0.3, linewidth=0.5)
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta) * omega, 'r--', linewidth=2, label='Exact')
        ax.set_xlabel('q')
        ax.set_ylabel('p')
        ax.set_title('Phase Space (Verlet, 5000 steps)')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Kepler orbit
        ax = axes[0, 2]
        t_k, q_k, p_k = methods["Velocity Verlet"](
            force_kepler, q0_kep, p0_kep, m, 0.01, int(10 * 2*np.pi/0.01))
        ax.plot(q_k[:, 0], q_k[:, 1], 'b-', linewidth=0.5)
        ax.plot(0, 0, 'yo', markersize=10, label='Sun')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Kepler Orbit (Verlet, 10 orbits)')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Kepler energy
        ax = axes[1, 0]
        E_k = compute_energy(q_k, p_k, m, V_kepler)
        ax.plot(t_k, (E_k - E_k[0])/np.abs(E_k[0]))
        ax.set_xlabel('Time')
        ax.set_ylabel('ΔE/|E₀|')
        ax.set_title('Kepler Energy Conservation')
        ax.grid(True, alpha=0.3)
        
        # Order comparison
        ax = axes[1, 1]
        dts = np.logspace(-3, -0.5, 10)
        for name, method in [("Sym. Euler", symplectic_euler), 
                              ("Verlet", velocity_verlet),
                              ("Yoshida4", yoshida4)]:
            errs = []
            for dt_test in dts:
                n_test = int(2*np.pi / dt_test)
                t_test, q_test, p_test = method(force_ho, q0, p0, m, dt_test, n_test)
                q_exact = np.cos(omega * t_test[-1])
                errs.append(abs(q_test[-1, 0] - q_exact))
            ax.loglog(dts, errs, 'o-', label=name)
        ax.set_xlabel('dt')
        ax.set_ylabel('|q(2π) - q_exact|')
        ax.set_title('Order of Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Long-time energy drift
        ax = axes[1, 2]
        t_long, q_long, p_long = velocity_verlet(force_ho, q0, p0, m, 0.1, 100000)
        E_long = compute_energy(q_long, p_long, m, V_ho)
        ax.plot(t_long, (E_long - E_long[0])/E_long[0])
        ax.set_xlabel('Time')
        ax.set_ylabel('ΔE/E₀')
        ax.set_title('Verlet: Long-time Energy (dt=0.1)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("symplectic_integrators.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
