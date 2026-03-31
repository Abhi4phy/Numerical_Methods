"""
Monte Carlo & Metropolis-Hastings Algorithm
=============================================
Sample from probability distributions using random walks.

**Metropolis-Hastings Algorithm:**
Generate samples from a target distribution π(x) using:

1. Start at x_0.
2. Propose x' ~ q(x'|x_t) (proposal distribution).
3. Accept with probability:
       α = min(1, π(x')q(x_t|x') / (π(x_t)q(x'|x_t)))
4. If accepted: x_{t+1} = x'. Else: x_{t+1} = x_t.
5. Repeat.

For symmetric proposals q(x'|x) = q(x|x'):
       α = min(1, π(x')/π(x_t))

**Properties:**
- Ergodic: visits all states proportional to π.
- Satisfies detailed balance: π(x)P(x→x') = π(x')P(x'→x).
- After burn-in, samples approximate the target distribution.
- Autocorrelation: consecutive samples are correlated.

**Physics Applications:**
- Ising model: spin configurations at temperature T.
- Partition functions in statistical mechanics.
- Boltzmann distribution sampling.
- Path integrals (quantum Monte Carlo).
"""

import numpy as np


def metropolis_hastings(log_target, x0, proposal_std=1.0,
                        n_samples=10000, burn_in=1000):
    """
    Metropolis-Hastings with Gaussian proposal.
    
    Parameters
    ----------
    log_target : callable — log of target distribution (unnormalized OK)
    x0 : float or ndarray — initial state
    proposal_std : float — standard deviation of Gaussian proposal
    n_samples : int — total samples (including burn-in)
    burn_in : int — samples to discard
    
    Returns
    -------
    samples : array of accepted samples (after burn-in)
    acceptance_rate : float
    all_samples : full chain including burn-in
    """
    x = np.array(x0, dtype=float)
    dim = x.size
    
    all_samples = np.zeros((n_samples, dim))
    all_samples[0] = x
    n_accepted = 0
    
    for t in range(1, n_samples):
        # Propose
        x_prop = x + proposal_std * np.random.randn(dim)
        
        # Acceptance ratio (in log space for numerical stability)
        log_alpha = log_target(x_prop) - log_target(x)
        
        if np.log(np.random.rand()) < log_alpha:
            x = x_prop
            n_accepted += 1
        
        all_samples[t] = x
    
    acceptance_rate = n_accepted / (n_samples - 1)
    samples = all_samples[burn_in:]
    
    return samples, acceptance_rate, all_samples


def metropolis_ising_2d(L, T, n_sweeps=1000, burn_in=200):
    """
    Metropolis algorithm for the 2D Ising model.
    
    H = -J Σ_{<ij>} s_i s_j
    
    Each sweep: visit all L×L sites and attempt a spin flip.
    Accept with Boltzmann probability: min(1, exp(-ΔE/T)).
    
    Parameters
    ----------
    L : int — lattice size (L × L)
    T : float — temperature (in units of J/k_B)
    n_sweeps : int — number of Monte Carlo sweeps
    burn_in : int — sweeps to discard
    
    Returns
    -------
    magnetization : array — |M|/N per sweep (after burn-in)
    energy : array — E/N per sweep (after burn-in)
    final_config : L×L spin array
    """
    J = 1.0  # Coupling constant
    beta = 1.0 / T
    
    # Random initial configuration
    spins = np.random.choice([-1, 1], size=(L, L))
    
    magnetization = []
    energy = []
    
    def compute_energy(s):
        E = 0
        for i in range(L):
            for j in range(L):
                # Nearest neighbors with periodic BC
                E -= J * s[i, j] * (s[(i+1)%L, j] + s[i, (j+1)%L])
        return E
    
    for sweep in range(n_sweeps):
        for _ in range(L * L):
            # Random site
            i = np.random.randint(L)
            j = np.random.randint(L)
            
            # Energy change from flipping spin (i,j)
            nn_sum = (spins[(i+1)%L, j] + spins[(i-1)%L, j] +
                      spins[i, (j+1)%L] + spins[i, (j-1)%L])
            dE = 2 * J * spins[i, j] * nn_sum
            
            # Metropolis criterion
            if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
                spins[i, j] *= -1
        
        if sweep >= burn_in:
            M = np.abs(np.sum(spins)) / L**2
            E = compute_energy(spins) / L**2
            magnetization.append(M)
            energy.append(E)
    
    return np.array(magnetization), np.array(energy), spins


def simulated_annealing(objective, x0, T_init=100, T_min=1e-6,
                         alpha=0.99, max_iter=10000, step_size=1.0):
    """
    Simulated annealing for global optimization.
    
    Like Metropolis, but temperature decreases over time,
    making acceptance of worse solutions less likely.
    
    T_{k+1} = α · T_k  (geometric cooling)
    
    Parameters
    ----------
    objective : callable — function to minimize
    x0 : starting point
    T_init : initial temperature
    T_min : final temperature
    alpha : cooling factor (0 < α < 1)
    
    Returns
    -------
    x_best, f_best, history
    """
    x = np.array(x0, dtype=float)
    f = objective(x)
    x_best = x.copy()
    f_best = f
    T = T_init
    
    history = [(x.copy(), f, T)]
    
    for k in range(max_iter):
        if T < T_min:
            break
        
        # Propose
        x_new = x + step_size * np.random.randn(len(x))
        f_new = objective(x_new)
        
        # Accept/reject
        df = f_new - f
        if df < 0 or np.random.rand() < np.exp(-df / T):
            x = x_new
            f = f_new
        
        # Update best
        if f < f_best:
            x_best = x.copy()
            f_best = f
        
        # Cool
        T *= alpha
        history.append((x.copy(), f, T))
    
    return x_best, f_best, history


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MONTE CARLO & METROPOLIS-HASTINGS DEMO")
    print("=" * 60)

    # --- Example 1: Sample from a bimodal distribution ---
    # TO TEST: Vary proposal_std, x0, and burn_in; observe acceptance rate, mode-hopping frequency, and histogram agreement with true density.
    print("\n--- Example 1: Bimodal Gaussian mixture ---")
    
    def log_bimodal(x):
        """Log of mixture: 0.3·N(-3,1) + 0.7·N(3,0.5)"""
        p1 = 0.3 * np.exp(-0.5 * (x[0] + 3)**2)
        p2 = 0.7 * np.exp(-0.5 * ((x[0] - 3)/0.5)**2) / 0.5
        return np.log(p1 + p2 + 1e-300)
    
    samples, acc_rate, all_samples = metropolis_hastings(
        log_bimodal, x0=[0.0], proposal_std=1.5, n_samples=50000, burn_in=5000
    )
    
    print(f"Acceptance rate: {acc_rate:.3f}")
    print(f"Sample mean: {np.mean(samples):.3f}")
    print(f"Sample std:  {np.std(samples):.3f}")

    # --- Example 2: 2D Ising model ---
    # TO TEST: Change lattice size L and temperatures around Tc (about 2.269), and observe shifts in average magnetization and energy.
    print("\n--- Example 2: 2D Ising Model ---")
    L = 20
    
    # Below critical temperature (Tc ≈ 2.269 for 2D Ising)
    T_low = 1.5
    mag_low, eng_low, config_low = metropolis_ising_2d(L, T_low, n_sweeps=500, burn_in=100)
    print(f"T = {T_low} (ordered):   <|M|> = {np.mean(mag_low):.3f}, "
          f"<E> = {np.mean(eng_low):.3f}")
    
    # Above critical temperature
    T_high = 4.0
    mag_high, eng_high, config_high = metropolis_ising_2d(L, T_high, n_sweeps=500, burn_in=100)
    print(f"T = {T_high} (disordered): <|M|> = {np.mean(mag_high):.3f}, "
          f"<E> = {np.mean(eng_high):.3f}")
    
    # Near critical
    T_c = 2.269
    mag_c, eng_c, config_c = metropolis_ising_2d(L, T_c, n_sweeps=500, burn_in=100)
    print(f"T = {T_c:.3f} (critical):  <|M|> = {np.mean(mag_c):.3f}, "
          f"<E> = {np.mean(eng_c):.3f}")

    # --- Example 3: Simulated annealing ---
    # TO TEST: Change T_init, alpha, step_size, or dimension, and observe final best value quality versus convergence speed.
    print("\n--- Example 3: Simulated Annealing on Rastrigin ---")
    
    def rastrigin(x):
        """Rastrigin function — many local minima, global min at origin."""
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2*np.pi*x))
    
    x0 = np.random.uniform(-5, 5, size=5)
    x_best, f_best, history = simulated_annealing(
        rastrigin, x0, T_init=100, alpha=0.995, max_iter=20000, step_size=0.5
    )
    print(f"Best f = {f_best:.6f} (global min = 0)")
    print(f"Best x ≈ {np.round(x_best, 3)}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # MCMC histogram
        axes[0, 0].hist(samples[:, 0], bins=100, density=True, alpha=0.7, label='MCMC samples')
        x_plot = np.linspace(-8, 8, 500)
        p_true = 0.3*np.exp(-0.5*(x_plot+3)**2) + 0.7*np.exp(-0.5*((x_plot-3)/0.5)**2)/0.5
        p_true /= np.trapz(p_true, x_plot)
        axes[0, 0].plot(x_plot, p_true, 'r-', linewidth=2, label='True density')
        axes[0, 0].set_title('MH: Bimodal Sampling')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MCMC trace
        axes[0, 1].plot(all_samples[:2000, 0], 'b-', alpha=0.5, linewidth=0.5)
        axes[0, 1].axvline(1000, color='r', linestyle='--', label='Burn-in end')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('x')
        axes[0, 1].set_title('MCMC Trace')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # SA convergence
        f_hist = [h[1] for h in history]
        T_hist = [h[2] for h in history]
        ax_sa = axes[0, 2]
        ax_sa.semilogy(f_hist, 'b-', alpha=0.5, linewidth=0.5, label='f(x)')
        ax_sa2 = ax_sa.twinx()
        ax_sa2.semilogy(T_hist, 'r-', alpha=0.5, linewidth=0.5, label='T')
        ax_sa.set_xlabel('Iteration')
        ax_sa.set_ylabel('Objective', color='b')
        ax_sa2.set_ylabel('Temperature', color='r')
        ax_sa.set_title('Simulated Annealing')
        ax_sa.grid(True, alpha=0.3)
        
        # Ising configurations
        for idx, (config, T_val, title) in enumerate([
            (config_low, T_low, f'T={T_low} (ordered)'),
            (config_c, T_c, f'T={T_c:.2f} (critical)'),
            (config_high, T_high, f'T={T_high} (disordered)')
        ]):
            axes[1, idx].imshow(config, cmap='binary', interpolation='nearest')
            axes[1, idx].set_title(f'Ising {L}×{L}: {title}')
            axes[1, idx].set_xticks([])
            axes[1, idx].set_yticks([])
        
        plt.tight_layout()
        plt.savefig("metropolis_hastings.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
