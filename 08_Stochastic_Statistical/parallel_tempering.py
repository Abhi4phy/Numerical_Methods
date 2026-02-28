"""
Parallel Tempering (Replica Exchange Monte Carlo)
==================================================
Advanced MCMC technique for sampling multimodal distributions
where standard Metropolis gets trapped in local minima.

**The Problem:**
Many physical systems have rugged energy landscapes with high barriers
between metastable states. Standard MCMC at low temperature gets
trapped and cannot explore the full configuration space.

**The Solution:**
Run multiple replicas at different temperatures T₁ < T₂ < ... < T_K.
Periodically attempt to SWAP configurations between adjacent temperatures:

    Accept swap(i, j) with probability:
    P_swap = min(1, exp((β_i - β_j)(E_i - E_j)))

where β = 1/(kT).

**Why it works:**
- Hot replicas explore freely (low barriers relative to kT)
- Cold replicas sample accurately near minima
- Swaps transfer hot exploration → cold accuracy
- Satisfies detailed balance → correct equilibrium

**Applications:**
- Protein folding
- Spin glasses
- Atomic cluster optimization
- Crystal structure prediction
- Bayesian inference with multimodal posteriors

Where to start:
━━━━━━━━━━━━━━
First understand basic Metropolis (08_Stochastic_Statistical/monte_carlo_metropolis.py).
Then run the double-well example below to see how PT overcomes barriers.
"""

import numpy as np


def metropolis_step(x, energy_func, beta, step_size, rng):
    """
    Single Metropolis-Hastings step.
    
    Parameters
    ----------
    x : np.array
        Current configuration
    energy_func : callable
        E(x) → float
    beta : float
        Inverse temperature 1/(kT)
    step_size : float
        Proposal step size
    rng : np.random.Generator
    
    Returns
    -------
    x_new : np.array
        New configuration (may be same as x if rejected)
    accepted : bool
    """
    x_new = x + step_size * rng.standard_normal(len(x))
    dE = energy_func(x_new) - energy_func(x)
    
    if dE < 0 or rng.random() < np.exp(-beta * dE):
        return x_new, True
    return x.copy(), False


def parallel_tempering(energy_func, x0, temperatures, n_steps, 
                       step_size=0.5, swap_interval=10, rng=None):
    """
    Parallel Tempering / Replica Exchange Monte Carlo.
    
    Parameters
    ----------
    energy_func : callable
        Energy function E(x)
    x0 : np.array
        Initial configuration (same for all replicas)
    temperatures : array-like
        Temperatures for each replica [T₁, T₂, ..., T_K]
        T₁ < T₂ < ... < T_K (coldest to hottest)
    n_steps : int
        Total MC steps per replica
    step_size : float
        Proposal step size
    swap_interval : int
        Attempt swaps every this many steps
    rng : np.random.Generator, optional
    
    Returns
    -------
    samples : dict
        {temp: array of samples} for each temperature
    energies : dict
        {temp: array of energies}
    swap_stats : dict
        Acceptance rates for swaps between adjacent temperatures
    """
    if rng is None:
        rng = np.random.default_rng()
    
    temperatures = np.array(temperatures)
    n_replicas = len(temperatures)
    betas = 1.0 / temperatures
    dim = len(x0)
    
    # Initialize replicas
    configs = [x0.copy() for _ in range(n_replicas)]
    current_E = [energy_func(x0) for _ in range(n_replicas)]
    
    # Storage
    samples = {T: [] for T in temperatures}
    energies = {T: [] for T in temperatures}
    
    # Swap statistics
    swap_attempts = np.zeros(n_replicas - 1)
    swap_accepts = np.zeros(n_replicas - 1)
    
    # MC acceptance stats
    mc_accepts = np.zeros(n_replicas)
    
    for step in range(n_steps):
        # --- Standard MC step for each replica ---
        for r in range(n_replicas):
            x_new = configs[r] + step_size * rng.standard_normal(dim)
            E_new = energy_func(x_new)
            dE = E_new - current_E[r]
            
            if dE < 0 or rng.random() < np.exp(-betas[r] * dE):
                configs[r] = x_new
                current_E[r] = E_new
                mc_accepts[r] += 1
        
        # --- Replica exchange (swap) ---
        if step % swap_interval == 0 and step > 0:
            # Alternate even/odd pairs to satisfy detailed balance
            start = (step // swap_interval) % 2
            
            for i in range(start, n_replicas - 1, 2):
                swap_attempts[i] += 1
                
                delta = (betas[i] - betas[i+1]) * (current_E[i] - current_E[i+1])
                
                if delta < 0 or rng.random() < np.exp(-delta):
                    # Swap configurations
                    configs[i], configs[i+1] = configs[i+1], configs[i]
                    current_E[i], current_E[i+1] = current_E[i+1], current_E[i]
                    swap_accepts[i] += 1
        
        # Store samples
        for r in range(n_replicas):
            samples[temperatures[r]].append(configs[r].copy())
            energies[temperatures[r]].append(current_E[r])
    
    # Convert to arrays
    for T in temperatures:
        samples[T] = np.array(samples[T])
        energies[T] = np.array(energies[T])
    
    # Compute swap rates
    swap_rates = {}
    for i in range(n_replicas - 1):
        if swap_attempts[i] > 0:
            swap_rates[(temperatures[i], temperatures[i+1])] = (
                swap_accepts[i] / swap_attempts[i])
        else:
            swap_rates[(temperatures[i], temperatures[i+1])] = 0.0
    
    mc_rates = mc_accepts / n_steps
    
    return samples, energies, swap_rates, mc_rates


def standard_mcmc(energy_func, x0, temperature, n_steps, step_size=0.5, rng=None):
    """
    Standard single-temperature MCMC for comparison.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    beta = 1.0 / temperature
    dim = len(x0)
    x = x0.copy()
    E = energy_func(x)
    
    samples = [x.copy()]
    energies_list = [E]
    accepts = 0
    
    for step in range(n_steps):
        x_new = x + step_size * rng.standard_normal(dim)
        E_new = energy_func(x_new)
        dE = E_new - E
        
        if dE < 0 or rng.random() < np.exp(-beta * dE):
            x = x_new
            E = E_new
            accepts += 1
        
        samples.append(x.copy())
        energies_list.append(E)
    
    return np.array(samples), np.array(energies_list), accepts / n_steps


def optimal_temperatures(T_min, T_max, n_replicas):
    """
    Generate geometrically-spaced temperatures.
    
    Geometric spacing gives roughly equal swap acceptance rates
    for Gaussian-like energy distributions.
    """
    return np.geomspace(T_min, T_max, n_replicas)


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PARALLEL TEMPERING (REPLICA EXCHANGE) DEMO")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # --- Double-well potential ---
    # V(x) = (x² - a²)² / (4a²) — barrier height ~ a²/4
    
    def double_well_1d(x, a=2.0, barrier=4.0):
        """1D double well with minima at x = ±a."""
        return barrier * (x[0]**2 / a**2 - 1)**2

    a = 2.0
    barrier = 4.0
    
    print(f"\n--- 1D Double Well ---")
    print(f"  Minima at x = ±{a}, barrier height = {barrier}")
    
    # Low temperature standard MCMC
    T_low = 0.5
    x0 = np.array([a])  # Start at one minimum
    
    print(f"\n  Standard MCMC at T = {T_low}:")
    samples_std, E_std, rate_std = standard_mcmc(
        double_well_1d, x0, T_low, 100000, step_size=0.3, rng=rng)
    
    # Count transitions between wells
    transitions_std = np.sum(np.diff(np.sign(samples_std[:, 0])) != 0) // 2
    print(f"    Accept rate: {rate_std:.3f}")
    print(f"    Well transitions: {transitions_std}")
    
    # Parallel tempering
    print(f"\n  Parallel Tempering:")
    temperatures = optimal_temperatures(T_low, 5.0, 6)
    print(f"    Temperatures: {temperatures.round(2)}")
    
    samples_pt, E_pt, swap_rates, mc_rates = parallel_tempering(
        double_well_1d, x0, temperatures, 100000, 
        step_size=0.3, swap_interval=5, rng=rng)
    
    transitions_pt = np.sum(
        np.diff(np.sign(samples_pt[temperatures[0]][:, 0])) != 0) // 2
    
    print(f"    MC accept rates: {mc_rates.round(3)}")
    print(f"    Swap rates: {dict((f'{k[0]:.2f}-{k[1]:.2f}', f'{v:.3f}') for k, v in swap_rates.items())}")
    print(f"    Cold replica transitions: {transitions_pt}")
    print(f"    Improvement: {transitions_pt}x vs {transitions_std}x (standard)")

    # --- 2D Multimodal ---
    print("\n--- 2D Three-Gaussian Mixture ---")
    
    def three_wells_2d(x):
        """Three minima in 2D."""
        centers = [(-2, -2), (2, -2), (0, 2.5)]
        wells = [3.0, 2.5, 3.5]
        E = 0.0
        for (cx, cy), w in zip(centers, wells):
            r2 = (x[0] - cx)**2 + (x[1] - cy)**2
            E -= w * np.exp(-0.5 * r2)
        return E
    
    x0_2d = np.array([-2.0, -2.0])
    temps_2d = optimal_temperatures(0.3, 3.0, 8)
    
    samples_2d, E_2d, swaps_2d, mc_2d = parallel_tempering(
        three_wells_2d, x0_2d, temps_2d, 50000,
        step_size=0.4, swap_interval=5, rng=rng)
    
    cold_samples = samples_2d[temps_2d[0]]
    print(f"  Cold T = {temps_2d[0]:.3f}")
    print(f"  Sample mean: ({cold_samples[:, 0].mean():.2f}, {cold_samples[:, 1].mean():.2f})")
    print(f"  Swap rates: {[f'{v:.3f}' for v in swaps_2d.values()]}")

    # --- Temperature optimization ---
    print("\n--- Optimal Temperature Spacing ---")
    for n_rep in [4, 6, 8, 12]:
        temps = optimal_temperatures(0.5, 5.0, n_rep)
        ratios = temps[1:] / temps[:-1]
        print(f"  {n_rep} replicas: ratio = {ratios[0]:.3f}, "
              f"T = [{temps[0]:.2f}, ..., {temps[-1]:.2f}]")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1D trajectory comparison
        ax = axes[0, 0]
        burnin = 10000
        ax.plot(samples_std[burnin:burnin+20000, 0], alpha=0.5, linewidth=0.3,
               label=f'Standard (T={T_low})')
        ax.axhline(a, color='r', linestyle='--', alpha=0.5)
        ax.axhline(-a, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('x')
        ax.set_title('Standard MCMC: Trapped!')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        cold = samples_pt[temperatures[0]]
        ax.plot(cold[burnin:burnin+20000, 0], alpha=0.5, linewidth=0.3,
               color='green', label=f'PT cold (T={temperatures[0]:.2f})')
        ax.axhline(a, color='r', linestyle='--', alpha=0.5)
        ax.axhline(-a, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('x')
        ax.set_title('Parallel Tempering: Free!')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Histograms comparison
        ax = axes[0, 2]
        x_hist = np.linspace(-5, 5, 200)
        boltz = np.exp(-double_well_1d(x_hist.reshape(-1,1).T[:1], a, barrier) / T_low)
        
        ax.hist(samples_std[burnin:, 0], bins=80, density=True, alpha=0.5,
               label='Standard', color='blue')
        ax.hist(cold[burnin:, 0], bins=80, density=True, alpha=0.5,
               label='PT', color='green')
        
        # Exact Boltzmann
        Z = np.trapz(np.exp(-np.array([double_well_1d(np.array([xi])) 
                     for xi in x_hist]) / T_low), x_hist)
        rho_exact = np.exp(-np.array([double_well_1d(np.array([xi])) 
                   for xi in x_hist]) / T_low) / Z
        ax.plot(x_hist, rho_exact, 'k-', linewidth=2, label='Boltzmann')
        ax.set_xlabel('x')
        ax.set_ylabel('ρ(x)')
        ax.set_title('Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2D scatter plot
        ax = axes[1, 0]
        cs = cold_samples[burnin:]
        ax.scatter(cs[::5, 0], cs[::5, 1], s=1, alpha=0.3, c='green')
        centers = [(-2, -2), (2, -2), (0, 2.5)]
        for cx, cy in centers:
            ax.plot(cx, cy, 'r*', markersize=15)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('2D PT: Cold Replica Samples')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Energy landscape 2D
        ax = axes[1, 1]
        xx = np.linspace(-4, 4, 100)
        yy = np.linspace(-4, 5, 100)
        XX, YY = np.meshgrid(xx, yy)
        ZZ = np.array([[three_wells_2d(np.array([xi, yi])) 
                       for xi in xx] for yi in yy])
        c = ax.contourf(XX, YY, ZZ, levels=30, cmap='RdYlBu_r')
        ax.scatter(cs[::10, 0], cs[::10, 1], s=1, alpha=0.5, c='white')
        plt.colorbar(c, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Energy Landscape + Samples')
        ax.set_aspect('equal')
        
        # Energy traces at different temperatures
        ax = axes[1, 2]
        for T in temperatures[:4]:
            ax.plot(E_pt[T][burnin:burnin+10000], alpha=0.5, linewidth=0.5,
                   label=f'T={T:.2f}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Traces (different T)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("parallel_tempering.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
