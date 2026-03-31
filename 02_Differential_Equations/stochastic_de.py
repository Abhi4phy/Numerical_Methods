"""
Stochastic Differential Equations (SDEs)
=========================================
Numerical integration of differential equations driven by random noise.

**The Model:**
    dX = a(X, t) dt + b(X, t) dW

where dW ~ N(0, dt) is a Wiener increment (Brownian motion).

**Physics Applications:**
- Langevin dynamics: m dv = -γv dt - ∇V dt + √(2γkT) dW
- Brownian motion and diffusion
- Fluctuating hydrodynamics
- Stochastic quantization
- Financial physics (Black-Scholes)
- Noise in electronic circuits

**Itô vs Stratonovich:**
Two interpretations of the noise integral that converge differently:
- Itô: natural for filtering, discrete-time limits
- Stratonovich: preserves chain rule, physical white noise limits

**Methods Implemented:**

1. **Euler-Maruyama** (strong order 0.5):
   X_{n+1} = X_n + a(X_n)·Δt + b(X_n)·ΔW_n

2. **Milstein** (strong order 1.0):
   X_{n+1} = X_n + a·Δt + b·ΔW + ½·b·b'·(ΔW² - Δt)
   The extra term captures the Itô correction.

3. **Stochastic Runge-Kutta** (order 1.0):
   Derivative-free alternative to Milstein.

4. **Heun (Stratonovich)** (strong order 1.0):
   For Stratonovich SDEs, a predictor-corrector approach.

Where to start:
━━━━━━━━━━━━━━
Run the geometric Brownian motion example first (analytic solution exists).
Compare Euler-Maruyama vs Milstein strong convergence rates.
Prerequisite: euler_method.py, monte_carlo_integration.py
"""

import numpy as np


def euler_maruyama(a, b, x0, t_span, dt, n_paths=1, rng=None):
    """
    Euler-Maruyama method for SDE: dX = a(X,t)dt + b(X,t)dW
    
    Strong order 0.5, weak order 1.0.
    Simplest SDE integrator — the "Euler method" for stochastic ODEs.
    
    Parameters
    ----------
    a : callable
        Drift: a(x, t) → float or array
    b : callable
        Diffusion: b(x, t) → float or array
    x0 : float
        Initial condition
    t_span : tuple
        (t_start, t_end)
    dt : float
        Time step
    n_paths : int
        Number of Monte Carlo paths
    rng : np.random.Generator, optional
        Random number generator
    
    Returns
    -------
    t : array, shape (n_steps+1,)
    X : array, shape (n_paths, n_steps+1)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    t0, tf = t_span
    n_steps = int((tf - t0) / dt)
    t = np.linspace(t0, tf, n_steps + 1)
    
    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = x0
    
    sqrt_dt = np.sqrt(dt)
    
    for i in range(n_steps):
        dW = sqrt_dt * rng.standard_normal(n_paths)
        X[:, i+1] = X[:, i] + a(X[:, i], t[i]) * dt + b(X[:, i], t[i]) * dW
    
    return t, X


def milstein(a, b, b_prime, x0, t_span, dt, n_paths=1, rng=None):
    """
    Milstein method — strong order 1.0.
    
    Includes the Itô correction term:
        X_{n+1} = X_n + a·Δt + b·ΔW + ½·b·b'·(ΔW² - Δt)
    
    Parameters
    ----------
    a : callable
        Drift coefficient a(x, t)
    b : callable
        Diffusion coefficient b(x, t)
    b_prime : callable
        Derivative of b w.r.t. x: db/dx(x, t)
    x0 : float
        Initial condition
    t_span : tuple
        (t_start, t_end)
    dt : float
        Time step
    n_paths : int
        Number of paths
    rng : np.random.Generator, optional
    """
    if rng is None:
        rng = np.random.default_rng()
    
    t0, tf = t_span
    n_steps = int((tf - t0) / dt)
    t = np.linspace(t0, tf, n_steps + 1)
    
    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = x0
    
    sqrt_dt = np.sqrt(dt)
    
    for i in range(n_steps):
        dW = sqrt_dt * rng.standard_normal(n_paths)
        bval = b(X[:, i], t[i])
        X[:, i+1] = (X[:, i] 
                     + a(X[:, i], t[i]) * dt 
                     + bval * dW 
                     + 0.5 * bval * b_prime(X[:, i], t[i]) * (dW**2 - dt))
    
    return t, X


def stochastic_rk(a, b, x0, t_span, dt, n_paths=1, rng=None):
    """
    Stochastic Runge-Kutta (Platen's scheme).
    
    Achieves strong order 1.0 WITHOUT needing db/dx.
    Uses a supporting value to estimate the correction:
        X̃ = X_n + b(X_n)·√Δt
        correction = [b(X̃) - b(X_n)] · (ΔW² - Δt) / (2√Δt)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    t0, tf = t_span
    n_steps = int((tf - t0) / dt)
    t = np.linspace(t0, tf, n_steps + 1)
    
    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = x0
    
    sqrt_dt = np.sqrt(dt)
    
    for i in range(n_steps):
        dW = sqrt_dt * rng.standard_normal(n_paths)
        
        bval = b(X[:, i], t[i])
        X_tilde = X[:, i] + bval * sqrt_dt
        
        X[:, i+1] = (X[:, i] 
                     + a(X[:, i], t[i]) * dt 
                     + bval * dW
                     + (b(X_tilde, t[i]) - bval) * (dW**2 - dt) / (2 * sqrt_dt))
    
    return t, X


def heun_stratonovich(a, b, x0, t_span, dt, n_paths=1, rng=None):
    """
    Heun method for Stratonovich SDE: dX = a(X,t)dt + b(X,t)∘dW
    
    Predictor-corrector that preserves Stratonovich interpretation.
    Useful when the SDE comes from a physical white-noise limit.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    t0, tf = t_span
    n_steps = int((tf - t0) / dt)
    t = np.linspace(t0, tf, n_steps + 1)
    
    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = x0
    
    sqrt_dt = np.sqrt(dt)
    
    for i in range(n_steps):
        dW = sqrt_dt * rng.standard_normal(n_paths)
        
        # Predictor (Euler)
        X_pred = X[:, i] + a(X[:, i], t[i]) * dt + b(X[:, i], t[i]) * dW
        
        # Corrector (average drift and diffusion)
        X[:, i+1] = (X[:, i] 
                     + 0.5 * (a(X[:, i], t[i]) + a(X_pred, t[i+1])) * dt
                     + 0.5 * (b(X[:, i], t[i]) + b(X_pred, t[i+1])) * dW)
    
    return t, X


def langevin_overdamped(V_prime, D, x0, dt, n_steps, n_paths=1, rng=None):
    """
    Overdamped Langevin dynamics (Brownian dynamics):
        dX = -V'(X)·dt + √(2D)·dW
    
    Steady-state distribution: ρ(x) ∝ exp(-V(x)/D)
    Used for sampling from Boltzmann distributions.
    
    Parameters
    ----------
    V_prime : callable
        Gradient of potential V'(x)
    D : float
        Diffusion coefficient (= kT/γ)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = x0
    
    sqrt_2D_dt = np.sqrt(2 * D * dt)
    
    for i in range(n_steps):
        dW = rng.standard_normal(n_paths)
        X[:, i+1] = X[:, i] - V_prime(X[:, i]) * dt + sqrt_2D_dt * dW
    
    t = np.arange(n_steps + 1) * dt
    return t, X


def strong_convergence_test(method_func, exact_func, x0, t_span, 
                            dts, n_paths=1000, rng_seed=42):
    """
    Test strong convergence: E[|X(T) - X_exact(T)|] vs dt.
    
    For strong order p, the error scales as dt^p.
    """
    errors = []
    
    for dt in dts:
        rng = np.random.default_rng(rng_seed)
        t, X = method_func(x0, t_span, dt, n_paths, rng)
        X_exact = exact_func(t[-1], x0, rng_seed, n_paths, dt)
        
        mean_error = np.mean(np.abs(X[:, -1] - X_exact))
        errors.append(mean_error)
    
    return np.array(errors)


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("STOCHASTIC DIFFERENTIAL EQUATIONS DEMO")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # --- 1. Geometric Brownian Motion ---
    # TO TEST: Modify drift (mu), volatility (sigma), initial price (S0), and time horizon (T).
    # Parameters: mu (0.05=5% drift), sigma (0.3=30% volatility), S0 (100), T (2 years), dt, n_paths.
    # Initial values: S0=100, mu=0.05, sigma=0.3 (typical stock-like dynamics).
    # Observe: Mean and variance of S(T) vs exact values. Increase n_paths for better estimates.
    # Note: Try mu > sigma/2 for upward trend, or mu < -sigma/2 for downward trend.
    # dS = μ·S·dt + σ·S·dW
    # Exact: S(t) = S₀·exp((μ - σ²/2)t + σ·W(t))
    print("\n--- Geometric Brownian Motion ---")
    mu, sigma = 0.05, 0.3
    S0 = 100.0
    T = 2.0
    
    a_gbm = lambda x, t: mu * x
    b_gbm = lambda x, t: sigma * x
    b_prime_gbm = lambda x, t: sigma * np.ones_like(x)
    
    dt = 0.001
    n_paths = 10000
    
    t_em, S_em = euler_maruyama(a_gbm, b_gbm, S0, (0, T), dt, n_paths, rng)
    
    # Statistics
    E_exact = S0 * np.exp(mu * T)
    Var_exact = S0**2 * np.exp(2*mu*T) * (np.exp(sigma**2*T) - 1)
    
    print(f"  E[S(T)]:  exact={E_exact:.2f}, MC={np.mean(S_em[:, -1]):.2f}")
    print(f"  Var[S(T)]: exact={Var_exact:.1f}, MC={np.var(S_em[:, -1]):.1f}")

    # --- 2. Strong Convergence Test ---
    # TO TEST: Compare Euler-Maruyama (order 0.5) vs Milstein (order 1.0) vs Stochastic RK (order 1.0).
    # Parameters: dts_test (time steps), n_paths (number of Monte Carlo paths), x0 (initial condition).
    # Initial values: x0=1.0, dts=[0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005].
    # Observe: Error scales as dt^0.5 for EM, dt^1.0 for higher-order methods (on log-log plot).
    # Try: Increase n_paths to 10000 for smoother convergence curves.
    print("\n--- Strong Convergence Test ---")
    dts_test = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    n_test = 5000
    
    # Reference solution at fine dt
    rng_ref = np.random.default_rng(123)
    dt_fine = 0.001
    n_fine = int(T / dt_fine)
    
    # Generate reference Brownian path
    dW_fine = np.sqrt(dt_fine) * rng_ref.standard_normal((n_test, n_fine))
    W_fine = np.cumsum(dW_fine, axis=1)
    S_exact_final = S0 * np.exp((mu - 0.5*sigma**2)*T + sigma*W_fine[:, -1])
    
    err_em = []
    err_mil = []
    
    for dt_test in dts_test:
        stride = int(dt_test / dt_fine)
        n_steps_test = int(T / dt_test)
        
        S_em_t = np.full(n_test, S0)
        S_mil_t = np.full(n_test, S0)
        
        for i in range(n_steps_test):
            # Aggregate Brownian increments
            dW_agg = np.sum(dW_fine[:, i*stride:(i+1)*stride], axis=1)
            
            # Euler-Maruyama
            S_em_t = S_em_t + mu*S_em_t*dt_test + sigma*S_em_t*dW_agg
            
            # Milstein
            S_mil_t = (S_mil_t + mu*S_mil_t*dt_test + sigma*S_mil_t*dW_agg
                      + 0.5*sigma**2*S_mil_t*(dW_agg**2 - dt_test))
        
        err_em.append(np.mean(np.abs(S_em_t - S_exact_final)))
        err_mil.append(np.mean(np.abs(S_mil_t - S_exact_final)))
    
    # Fit convergence rates
    log_dt = np.log(dts_test)
    slope_em = np.polyfit(log_dt, np.log(err_em), 1)[0]
    slope_mil = np.polyfit(log_dt, np.log(err_mil), 1)[0]
    
    print(f"  Euler-Maruyama: strong order ≈ {slope_em:.2f} (expected 0.5)")
    print(f"  Milstein:       strong order ≈ {slope_mil:.2f} (expected 1.0)")

    # --- 3. Ornstein-Uhlenbeck Process ---
    # dX = -θ(X - μ)dt + σ dW
    print("\n--- Ornstein-Uhlenbeck Process ---")
    theta_ou, mu_ou, sigma_ou = 1.0, 0.0, 0.5
    
    a_ou = lambda x, t: -theta_ou * (x - mu_ou)
    b_ou = lambda x, t: sigma_ou * np.ones_like(x)
    
    t_ou, X_ou = euler_maruyama(a_ou, b_ou, 2.0, (0, 10), 0.01, 5000, rng)
    
    # Steady state: mean = μ, var = σ²/(2θ)
    var_exact = sigma_ou**2 / (2 * theta_ou)
    print(f"  Steady-state mean:  exact={mu_ou:.2f}, MC={np.mean(X_ou[:, -1]):.3f}")
    print(f"  Steady-state var:   exact={var_exact:.4f}, MC={np.var(X_ou[:, -1]):.4f}")

    # --- 4. Langevin in Double-Well ---
    print("\n--- Overdamped Langevin: Double-Well Potential ---")
    # V(x) = (x² - 1)²  → minima at x = ±1
    V_dw = lambda x: (x**2 - 1)**2
    V_prime_dw = lambda x: 4 * x * (x**2 - 1)
    D = 0.3  # Temperature
    
    t_lv, X_lv = langevin_overdamped(V_prime_dw, D, 0.0, 0.001, 500000, 
                                      n_paths=1, rng=rng)
    
    # Histogram → should approximate Boltzmann distribution
    print(f"  Simulated {len(t_lv)} steps, x ranges [{X_lv[0].min():.2f}, "
          f"{X_lv[0].max():.2f}]")
    
    x_hist = np.linspace(-2.5, 2.5, 100)
    rho_exact = np.exp(-V_dw(x_hist) / D)
    rho_exact /= np.trapz(rho_exact, x_hist)
    
    hist, bin_edges = np.histogram(X_lv[0, 10000:], bins=80, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Check peaks near ±1
    peak_idx = np.argmax(hist)
    print(f"  Most probable x ≈ {bin_centers[peak_idx]:.2f} (expected near ±1)")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # GBM sample paths
        ax = axes[0, 0]
        for i in range(min(20, n_paths)):
            ax.plot(t_em, S_em[i], alpha=0.3, linewidth=0.5)
        ax.plot(t_em, S0 * np.exp(mu * t_em), 'k-', linewidth=2, label='E[S]')
        ax.set_xlabel('Time')
        ax.set_ylabel('S(t)')
        ax.set_title('Geometric Brownian Motion (20 paths)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Strong convergence
        ax = axes[0, 1]
        ax.loglog(dts_test, err_em, 'bo-', label=f'EM (slope={slope_em:.2f})')
        ax.loglog(dts_test, err_mil, 'rs-', label=f'Milstein (slope={slope_mil:.2f})')
        ax.loglog(dts_test, np.array(dts_test)**0.5 * err_em[0]/dts_test[0]**0.5, 
                 'b--', alpha=0.3, label='O(dt^0.5)')
        ax.loglog(dts_test, np.array(dts_test)**1.0 * err_mil[0]/dts_test[0]**1.0,
                 'r--', alpha=0.3, label='O(dt^1.0)')
        ax.set_xlabel('dt')
        ax.set_ylabel('E[|X(T) - X_exact(T)|]')
        ax.set_title('Strong Convergence')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # OU process
        ax = axes[0, 2]
        for i in range(5):
            ax.plot(t_ou, X_ou[i], alpha=0.5, linewidth=0.5)
        ax.axhline(mu_ou, color='k', linestyle='--', label=f'μ = {mu_ou}')
        ax.fill_between(t_ou, mu_ou - 2*np.sqrt(var_exact), 
                        mu_ou + 2*np.sqrt(var_exact), alpha=0.1, color='gray',
                        label='±2σ_∞')
        ax.set_xlabel('Time')
        ax.set_ylabel('X(t)')
        ax.set_title('Ornstein-Uhlenbeck Process')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Double well potential
        ax = axes[1, 0]
        ax.plot(x_hist, V_dw(x_hist), 'b-', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('V(x)')
        ax.set_title('Double-Well Potential')
        ax.grid(True, alpha=0.3)
        
        # Langevin histogram
        ax = axes[1, 1]
        ax.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0], 
               alpha=0.6, label='Simulation')
        ax.plot(x_hist, rho_exact, 'r-', linewidth=2, label='Boltzmann')
        ax.set_xlabel('x')
        ax.set_ylabel('ρ(x)')
        ax.set_title('Langevin → Boltzmann Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Langevin trajectory
        ax = axes[1, 2]
        t_plot = t_lv[:50000]
        X_plot = X_lv[0, :50000]
        ax.plot(t_plot, X_plot, 'b-', alpha=0.3, linewidth=0.3)
        ax.axhline(1, color='r', linestyle='--', alpha=0.5)
        ax.axhline(-1, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('X(t)')
        ax.set_title('Langevin Trajectory (transitions)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("stochastic_de.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
