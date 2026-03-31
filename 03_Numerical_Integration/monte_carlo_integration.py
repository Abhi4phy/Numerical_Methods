"""
Monte Carlo Integration
========================
Uses random sampling to estimate integrals. Particularly powerful for
high-dimensional integrals where deterministic methods fail (curse of
dimensionality).

Basic idea:
    ∫_Ω f(x) dx ≈ V(Ω)/N Σ f(xᵢ)  where xᵢ are random samples in Ω

Properties:
- Convergence rate: O(1/√N) regardless of dimension.
- Error ∝ σ(f)/√N where σ is the standard deviation of f.
- Trivially parallelizable.
- Can handle very complex integration domains.

Variance reduction techniques:
- Importance sampling: sample more where f is large.
- Stratified sampling: divide domain into strata.
- Antithetic variates: use correlated samples to reduce variance.

Physics applications:
- Quantum field theory (path integrals), statistical mechanics.
- Radiation transport, particle physics.
"""

import numpy as np


def monte_carlo_1d(f, a, b, n_samples=100000):
    """
    Monte Carlo integration of f over [a, b].
    
    ∫_a^b f(x) dx ≈ (b-a)/N Σ f(xᵢ), xᵢ ~ Uniform[a, b]
    
    Returns
    -------
    estimate : float – integral estimate
    error : float – estimated standard error
    """
    x = np.random.uniform(a, b, n_samples)
    f_values = f(x)
    
    estimate = (b - a) * np.mean(f_values)
    error = (b - a) * np.std(f_values) / np.sqrt(n_samples)
    
    return estimate, error


def monte_carlo_nd(f, bounds, n_samples=100000):
    """
    Monte Carlo integration in arbitrary dimensions.
    
    Parameters
    ----------
    f : callable – f(x) where x is ndarray of shape (d,)
    bounds : list of (low, high) tuples for each dimension
    n_samples : int
    
    Returns
    -------
    estimate : float
    error : float
    """
    d = len(bounds)
    volume = np.prod([b - a for a, b in bounds])
    
    # Generate random points
    samples = np.zeros((n_samples, d))
    for i, (low, high) in enumerate(bounds):
        samples[:, i] = np.random.uniform(low, high, n_samples)
    
    # Evaluate function
    f_values = np.array([f(s) for s in samples])
    
    estimate = volume * np.mean(f_values)
    error = volume * np.std(f_values) / np.sqrt(n_samples)
    
    return estimate, error


def importance_sampling(f, g, g_sample, g_pdf, n_samples=100000):
    """
    Importance Sampling: Sample from g(x) instead of uniform.
    
    ∫ f(x) dx = ∫ [f(x)/g(x)] g(x) dx ≈ (1/N) Σ f(xᵢ)/g(xᵢ)
    where xᵢ ~ g(x)
    
    Choose g(x) ∝ |f(x)| for minimum variance.
    
    Parameters
    ----------
    f : callable – integrand
    g_sample : callable – returns samples from g
    g_pdf : callable – probability density of g
    """
    x = g_sample(n_samples)
    ratios = f(x) / g_pdf(x)
    
    estimate = np.mean(ratios)
    error = np.std(ratios) / np.sqrt(n_samples)
    
    return estimate, error


def stratified_sampling(f, a, b, n_strata, samples_per_stratum):
    """
    Stratified Sampling: Divide [a,b] into strata and sample within each.
    
    Reduces variance compared to simple MC when f varies across the domain.
    """
    h = (b - a) / n_strata
    total = 0.0
    var_sum = 0.0
    
    for i in range(n_strata):
        low = a + i * h
        high = low + h
        x = np.random.uniform(low, high, samples_per_stratum)
        f_values = f(x)
        total += h * np.mean(f_values)
        var_sum += (h * np.std(f_values))**2 / samples_per_stratum
    
    return total, np.sqrt(var_sum)


def hit_or_miss(f, a, b, f_max, n_samples=100000):
    """
    Hit-or-miss Monte Carlo (acceptance-rejection).
    
    Geometric interpretation: fraction of random points (x,y) that
    fall under the curve gives the integral.
    
    ∫_a^b f(x) dx ≈ (b-a) × f_max × (hits / total)
    """
    x = np.random.uniform(a, b, n_samples)
    y = np.random.uniform(0, f_max, n_samples)
    
    hits = np.sum(y <= f(x))
    area = (b - a) * f_max
    
    estimate = area * hits / n_samples
    error = area * np.sqrt(hits * (n_samples - hits) / n_samples**3)
    
    return estimate, error


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 60)
    print("MONTE CARLO INTEGRATION DEMO")
    print("=" * 60)

    # --- 1D integral ---
    # TO TEST: Verify convergence rate O(1/√N). Error should decrease by factor ~3.16 when N→10N.
    # Parameters: f=sin(x), domain [0,π], exact=2, N=[100, 1000, 10000, 100000, 1000000].
    # Initial values: Random seed=42 (reproducible) for RNG consistency.
    # Observe: Both statistical error (σ/√N) and actual error vs exact value=2.
    # Try: N=[10, 100, 1000] for faster run, or different f like exp(x) or 1/(1+x²).
    print("\n--- 1D: ∫₀^π sin(x) dx = 2 ---")
    for n in [100, 1000, 10000, 100000, 1000000]:
        est, err = monte_carlo_1d(np.sin, 0, np.pi, n)
        print(f"  N={n:>8d}  estimate={est:.6f}  error={err:.6f}  "
              f"actual_err={abs(est-2):.6f}")

    # --- High-dimensional integral ---
    # TO TEST: Compute volumes/integrals in high dimensions where grid methods fail. Try 4D, 8D, 10D.
    # Parameters: Indicator function in_Nball(x) (1 if |x|≤1, else 0), dimension=6, n_samples=500000.
    # Initial values: 6D unit ball with exact volume V₆=π³/6≈5.1677, bounds=[-1,1]^6.
    # Observe: Estimate ± error vs exact. Even with 0.5M samples, error may be ~0.1 (curse of dimensionality).
    # Try: Reduce n_samples to 50K to see larger error, or increase to 2M for tighter estimate.
    print("\n--- 6D: Volume of unit 6-ball (exact = π³/6 ≈ 5.1677) ---")
    exact_6ball = np.pi**3 / 6
    
    def in_6ball(x):
        return 1.0 if np.sum(x**2) <= 1.0 else 0.0
    
    bounds_6d = [(-1, 1)] * 6
    est, err = monte_carlo_nd(in_6ball, bounds_6d, n_samples=500000)
    print(f"  Estimate: {est:.4f} ± {err:.4f}")
    print(f"  Exact:    {exact_6ball:.4f}")
    print(f"  Error:    {abs(est - exact_6ball):.4f}")

    # --- Importance sampling ---
    # TO TEST: Compare importance sampling vs naive MC. Importance should have much smaller variance.
    # Parameters: f=x²exp(-x) (unbounded but integrable), use exponential dist (1-exp(-x)) as proposal.
    # Initial values: Exponential proposal g(x)=exp(-x), n_samples=100000, exact integral=2.
    # Observe: Importance sampling error should be much smaller than naive MC on [0,20].
    # Insight: Proposal g should match shape of |f|. Try other integrals: ∫x⁴exp(-x), ∫sqrt(x)exp(-x).
    print("\n--- Importance Sampling: ∫₀^∞ x² e^{-x} dx = 2 ---")
    # Use exponential distribution as importance distribution
    f_is = lambda x: x**2 * np.exp(-x)
    g_sample = lambda n: np.random.exponential(1, n)
    g_pdf = lambda x: np.exp(-x)

    est_is, err_is = importance_sampling(
        lambda x: x**2 * np.exp(-x),
        None,
        g_sample,
        g_pdf,
        n_samples=100000
    )
    print(f"  Importance sampling: {est_is:.6f} ± {err_is:.6f}")
    
    # Compare with naive MC (using truncated domain)
    est_naive, err_naive = monte_carlo_1d(
        lambda x: x**2 * np.exp(-x), 0, 20, 100000
    )
    print(f"  Naive MC [0,20]:    {est_naive:.6f} ± {err_naive:.6f}")

    # --- Convergence rate ---
    print("\n--- Convergence rate: O(1/√N) ---")
    ns = [10**k for k in range(2, 7)]
    for n in ns:
        _, err = monte_carlo_1d(np.sin, 0, np.pi, n)
        print(f"  N={n:>8d}  σ/√N = {err:.6f}  1/√N = {1/np.sqrt(n):.6f}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Convergence
        ns = np.logspace(1, 6, 30).astype(int)
        errors = []
        for n in ns:
            estimates = [monte_carlo_1d(np.sin, 0, np.pi, n)[0] for _ in range(20)]
            errors.append(np.std(estimates))
        
        axes[0].loglog(ns, errors, 'bo', alpha=0.6, markersize=4, label='MC std')
        axes[0].loglog(ns, 1.0/np.sqrt(ns), 'r-', label='1/√N')
        axes[0].set_xlabel('N (samples)')
        axes[0].set_ylabel('Standard deviation')
        axes[0].set_title('Monte Carlo Convergence O(1/√N)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Hit-or-miss visualization
        n_vis = 5000
        x = np.random.uniform(0, np.pi, n_vis)
        y = np.random.uniform(0, 1, n_vis)
        hits = y <= np.sin(x)
        
        axes[1].scatter(x[hits], y[hits], c='green', s=1, alpha=0.5, label='Hit')
        axes[1].scatter(x[~hits], y[~hits], c='red', s=1, alpha=0.5, label='Miss')
        t = np.linspace(0, np.pi, 200)
        axes[1].plot(t, np.sin(t), 'k-', linewidth=2)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_title('Hit-or-Miss Monte Carlo')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("monte_carlo_integration.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
