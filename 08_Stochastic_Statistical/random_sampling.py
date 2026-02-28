"""
Random Number Generation & Statistical Sampling
=================================================
Methods for generating random numbers and sampling from distributions.

**Pseudo-Random Number Generators (PRNGs):**
1. Linear Congruential Generator (LCG):
       x_{n+1} = (a · x_n + c) mod m
   Simple but has known weaknesses (hyperplane structure).

2. Mersenne Twister: the standard PRNG (period 2^{19937} - 1).
   Used by Python's random module and NumPy.

**Sampling Methods:**
1. Inverse Transform Sampling: if F is the CDF, then F⁻¹(U) ~ target.
2. Box-Muller Transform: uniform → Gaussian.
3. Rejection Sampling: sample from proposal, accept/reject.
4. Importance Sampling: reweight samples from wrong distribution.

**Quasi-Random (Low-Discrepancy) Sequences:**
- Halton, Sobol sequences
- Better coverage than pseudo-random for integration
- O(1/N) convergence vs O(1/√N) for pseudo-random

Physics: Monte Carlo simulations, stochastic differential equations,
         statistical mechanics, uncertainty propagation.
"""

import numpy as np


# ============================================================
# Pseudo-Random Number Generators
# ============================================================
class LCG:
    """
    Linear Congruential Generator.
    
    x_{n+1} = (a * x_n + c) mod m
    
    Common parameters (Numerical Recipes):
        a = 1664525, c = 1013904223, m = 2^32
    """
    
    def __init__(self, seed=42, a=1664525, c=1013904223, m=2**32):
        self.state = seed
        self.a = a
        self.c = c
        self.m = m
    
    def next_int(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state
    
    def next_uniform(self):
        """Return U(0, 1)."""
        return self.next_int() / self.m
    
    def sample(self, n):
        """Generate n uniform samples."""
        return np.array([self.next_uniform() for _ in range(n)])


class XorShift:
    """
    XorShift PRNG — fast and reasonably good quality.
    
    Uses bitwise XOR and shifts.
    """
    
    def __init__(self, seed=42):
        self.state = seed if seed != 0 else 1  # State must be non-zero
    
    def next_int(self):
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 7) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self.state = x & 0xFFFFFFFFFFFFFFFF
        return self.state
    
    def next_uniform(self):
        return self.next_int() / 2**64
    
    def sample(self, n):
        return np.array([self.next_uniform() for _ in range(n)])


# ============================================================
# Sampling Methods
# ============================================================
def inverse_transform_exponential(n, lam=1.0, rng=None):
    """
    Sample from Exponential(λ) using inverse CDF method.
    
    CDF: F(x) = 1 - exp(-λx)
    Inverse: F⁻¹(u) = -ln(1-u)/λ
    """
    if rng is None:
        u = np.random.rand(n)
    else:
        u = rng.sample(n)
    return -np.log(1 - u) / lam


def box_muller(n):
    """
    Box-Muller transform: uniform → standard normal.
    
    Given U₁, U₂ ~ U(0,1):
        Z₁ = √(-2 ln U₁) cos(2π U₂)
        Z₂ = √(-2 ln U₁) sin(2π U₂)
    
    Then Z₁, Z₂ ~ N(0,1) independently.
    """
    # Generate pairs
    n_pairs = (n + 1) // 2
    u1 = np.random.rand(n_pairs)
    u2 = np.random.rand(n_pairs)
    
    r = np.sqrt(-2 * np.log(u1))
    theta = 2 * np.pi * u2
    
    z1 = r * np.cos(theta)
    z2 = r * np.sin(theta)
    
    samples = np.empty(2 * n_pairs)
    samples[0::2] = z1
    samples[1::2] = z2
    
    return samples[:n]


def rejection_sampling(target_pdf, proposal_sampler, proposal_pdf,
                        M, n_samples):
    """
    Rejection sampling.
    
    Sample from target f(x) using proposal g(x) where f(x) ≤ M·g(x).
    
    1. Sample x ~ g(x)
    2. Sample u ~ U(0, 1)
    3. Accept if u < f(x) / (M · g(x))
    
    Efficiency = 1/M.
    
    Parameters
    ----------
    target_pdf : callable — target density
    proposal_sampler : callable() — returns one sample from g
    proposal_pdf : callable — proposal density g(x)
    M : float — bounding constant
    n_samples : int
    
    Returns
    -------
    samples, n_total_proposed
    """
    samples = []
    n_proposed = 0
    
    while len(samples) < n_samples:
        x = proposal_sampler()
        u = np.random.rand()
        n_proposed += 1
        
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            samples.append(x)
    
    return np.array(samples), n_proposed


def importance_sampling(h, target_log_pdf, proposal_sampler, proposal_log_pdf,
                        n_samples):
    """
    Importance sampling to estimate E_π[h(x)].
    
    E_π[h(x)] ≈ Σ w_i h(x_i) / Σ w_i
    
    where x_i ~ q(x), w_i = π(x_i) / q(x_i).
    
    Self-normalized version doesn't need π to be normalized.
    
    Parameters
    ----------
    h : callable — function to compute expectation of
    target_log_pdf : callable — log of target (unnormalized OK)
    proposal_sampler : callable(n) — sample n from proposal
    proposal_log_pdf : callable — log of proposal density
    n_samples : int
    
    Returns
    -------
    estimate, effective_n, samples, weights
    """
    x = proposal_sampler(n_samples)
    
    log_w = np.array([target_log_pdf(xi) - proposal_log_pdf(xi) for xi in x])
    log_w -= np.max(log_w)  # Numerical stability
    w = np.exp(log_w)
    
    h_vals = np.array([h(xi) for xi in x])
    estimate = np.sum(w * h_vals) / np.sum(w)
    
    # Effective sample size
    w_norm = w / np.sum(w)
    n_eff = 1.0 / np.sum(w_norm**2)
    
    return estimate, n_eff, x, w


# ============================================================
# Quasi-Random Sequences
# ============================================================
def halton_sequence(n, base=2):
    """
    Generate Halton sequence — a quasi-random low-discrepancy sequence.
    
    For index i, compute the radical inverse in the given base.
    Different bases for different dimensions.
    """
    seq = np.zeros(n)
    for i in range(1, n + 1):
        f = 1.0
        result = 0.0
        idx = i
        while idx > 0:
            f /= base
            result += f * (idx % base)
            idx //= base
        seq[i-1] = result
    return seq


def halton_sequence_nd(n, dim):
    """
    Multi-dimensional Halton sequence.
    Uses first `dim` primes as bases.
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    if dim > len(primes):
        raise ValueError(f"Max {len(primes)} dimensions supported")
    
    points = np.zeros((n, dim))
    for d in range(dim):
        points[:, d] = halton_sequence(n, primes[d])
    return points


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RANDOM SAMPLING METHODS DEMO")
    print("=" * 60)

    # --- LCG test ---
    print("\n--- LCG: Linear Congruential Generator ---")
    lcg = LCG(seed=12345)
    lcg_samples = lcg.sample(10000)
    print(f"LCG: mean = {np.mean(lcg_samples):.4f} (expect 0.5)")
    print(f"     std  = {np.std(lcg_samples):.4f} (expect {1/np.sqrt(12):.4f})")

    # --- Box-Muller ---
    print("\n--- Box-Muller Transform ---")
    bm_samples = box_muller(100000)
    print(f"Box-Muller: mean = {np.mean(bm_samples):.4f} (expect 0)")
    print(f"            std  = {np.std(bm_samples):.4f} (expect 1)")
    print(f"            skew = {np.mean(bm_samples**3):.4f} (expect 0)")

    # --- Inverse Transform ---
    print("\n--- Inverse Transform: Exponential(λ=2) ---")
    exp_samples = inverse_transform_exponential(100000, lam=2.0)
    print(f"Mean: {np.mean(exp_samples):.4f} (expect {1/2.0})")
    print(f"Std:  {np.std(exp_samples):.4f} (expect {1/2.0})")

    # --- Rejection Sampling ---
    print("\n--- Rejection Sampling: Beta(2,5) ---")
    from scipy.stats import beta as beta_dist
    
    target_pdf = lambda x: beta_dist.pdf(x, 2, 5)
    proposal_sampler = lambda: np.random.rand()  # U(0,1)
    proposal_pdf = lambda x: 1.0
    M = 2.5  # Max of Beta(2,5) pdf ≈ 2.2
    
    beta_samples, n_proposed = rejection_sampling(
        target_pdf, proposal_sampler, proposal_pdf, M, n_samples=10000
    )
    efficiency = 10000 / n_proposed
    print(f"Efficiency: {efficiency:.3f} (theoretical max: {1/M:.3f})")
    print(f"Sample mean: {np.mean(beta_samples):.4f} (true: {2/7:.4f})")

    # --- Importance Sampling ---
    print("\n--- Importance Sampling ---")
    print("Estimate P(X > 3) where X ~ N(0,1) using N(3,1) proposal")
    
    h_func = lambda x: 1.0 if x > 3 else 0.0
    target_logpdf = lambda x: -0.5 * x**2
    prop_sampler = lambda n: np.random.normal(3, 1, n)
    prop_logpdf = lambda x: -0.5 * (x - 3)**2
    
    est, n_eff, _, _ = importance_sampling(h_func, target_logpdf,
                                            prop_sampler, prop_logpdf,
                                            n_samples=50000)
    from scipy.stats import norm
    true_val = 1 - norm.cdf(3)
    print(f"Estimated P(X>3): {est:.6f}")
    print(f"True P(X>3):      {true_val:.6f}")
    print(f"Effective samples: {n_eff:.0f} / 50000")

    # --- Halton vs Random ---
    print("\n--- Halton vs Pseudo-Random ---")
    n = 1000
    halton_2d = halton_sequence_nd(n, 2)
    random_2d = np.random.rand(n, 2)
    
    # Integration test: ∫∫ sin(πx)sin(πy) dxdy over [0,1]² = (2/π)² ≈ 0.4053
    f = lambda p: np.sin(np.pi * p[0]) * np.sin(np.pi * p[1])
    true_int = (2/np.pi)**2
    
    est_halton = np.mean([f(p) for p in halton_2d])
    est_random = np.mean([f(p) for p in random_2d])
    
    print(f"True integral: {true_int:.6f}")
    print(f"Halton (n={n}): {est_halton:.6f}, error = {abs(est_halton-true_int):.6f}")
    print(f"Random (n={n}): {est_random:.6f}, error = {abs(est_random-true_int):.6f}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # LCG scatter (pairs)
        axes[0, 0].scatter(lcg_samples[:-1], lcg_samples[1:], s=0.5, alpha=0.3)
        axes[0, 0].set_xlabel('x_n')
        axes[0, 0].set_ylabel('x_{n+1}')
        axes[0, 0].set_title('LCG: Sequential Pairs')
        
        # Box-Muller histogram
        axes[0, 1].hist(bm_samples, bins=100, density=True, alpha=0.7, color='steelblue')
        x_norm = np.linspace(-4, 4, 200)
        axes[0, 1].plot(x_norm, np.exp(-x_norm**2/2)/np.sqrt(2*np.pi), 'r-', linewidth=2)
        axes[0, 1].set_title('Box-Muller: Standard Normal')
        
        # Rejection sampling
        axes[0, 2].hist(beta_samples, bins=50, density=True, alpha=0.7, color='steelblue',
                       label='Samples')
        x_beta = np.linspace(0, 1, 200)
        axes[0, 2].plot(x_beta, beta_dist.pdf(x_beta, 2, 5), 'r-', linewidth=2,
                       label='Beta(2,5)')
        axes[0, 2].set_title(f'Rejection Sampling (eff={efficiency:.2f})')
        axes[0, 2].legend()
        
        # Halton vs Random
        axes[1, 0].scatter(random_2d[:, 0], random_2d[:, 1], s=2, alpha=0.5)
        axes[1, 0].set_title('Pseudo-Random (n=1000)')
        axes[1, 0].set_xlim(0, 1); axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_aspect('equal')
        
        axes[1, 1].scatter(halton_2d[:, 0], halton_2d[:, 1], s=2, alpha=0.5, c='red')
        axes[1, 1].set_title('Halton Sequence (n=1000)')
        axes[1, 1].set_xlim(0, 1); axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_aspect('equal')
        
        # Convergence comparison for integration
        ns = np.logspace(1, 4, 30).astype(int)
        errors_random = []
        errors_halton = []
        
        for ni in ns:
            pts_r = np.random.rand(ni, 2)
            pts_h = halton_sequence_nd(ni, 2)
            est_r = np.mean([f(p) for p in pts_r])
            est_h = np.mean([f(p) for p in pts_h])
            errors_random.append(abs(est_r - true_int))
            errors_halton.append(abs(est_h - true_int))
        
        axes[1, 2].loglog(ns, errors_random, 'b.', alpha=0.6, label='Pseudo-random')
        axes[1, 2].loglog(ns, errors_halton, 'r.', alpha=0.6, label='Halton')
        axes[1, 2].loglog(ns, 1/np.sqrt(ns), 'b--', alpha=0.5, label='O(1/√N)')
        axes[1, 2].loglog(ns, 1/ns, 'r--', alpha=0.5, label='O(1/N)')
        axes[1, 2].set_xlabel('N')
        axes[1, 2].set_ylabel('Integration Error')
        axes[1, 2].set_title('Convergence: Random vs Quasi-Random')
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("random_sampling.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
