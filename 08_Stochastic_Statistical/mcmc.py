"""
Markov Chain Monte Carlo (MCMC) for Bayesian Inference
=======================================================
Use MCMC to sample from posterior distributions:

    P(θ|data) ∝ P(data|θ) · P(θ)
                likelihood · prior

When the posterior can't be computed analytically,
MCMC provides samples that approximate it.

**Methods implemented:**

1. **Metropolis-Hastings** — general-purpose MCMC
2. **Gibbs Sampling** — sample each parameter conditional on the rest
3. **Hamiltonian Monte Carlo (HMC)** — use gradient information for
   efficient exploration of parameter space

**Diagnostics:**
- Trace plots — visual check for convergence
- Autocorrelation — how correlated are sequential samples?
- Effective sample size (ESS) — independent samples
- Gelman-Rubin R̂ — multiple chain convergence

Physics: Bayesian parameter estimation, model selection,
uncertainty quantification in simulations.
"""

import numpy as np


def mcmc_bayesian(log_likelihood, log_prior, x0, proposal_std=0.1,
                   n_samples=20000, burn_in=5000):
    """
    MCMC for Bayesian inference using Metropolis-Hastings.
    
    Samples from posterior ∝ likelihood × prior.
    
    Parameters
    ----------
    log_likelihood : callable(θ) — log-likelihood
    log_prior : callable(θ) — log-prior
    x0 : ndarray — initial parameter values
    
    Returns
    -------
    samples : posterior samples (after burn-in)
    log_posteriors : log-posterior values
    acceptance_rate : float
    """
    def log_posterior(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)
    
    theta = np.array(x0, dtype=float)
    dim = len(theta)
    
    chain = np.zeros((n_samples, dim))
    log_posts = np.zeros(n_samples)
    chain[0] = theta
    log_posts[0] = log_posterior(theta)
    n_accept = 0
    
    for t in range(1, n_samples):
        # Propose
        theta_prop = theta + proposal_std * np.random.randn(dim)
        log_post_prop = log_posterior(theta_prop)
        
        # Accept/reject
        log_alpha = log_post_prop - log_posts[t-1]
        
        if np.log(np.random.rand()) < log_alpha:
            theta = theta_prop
            log_posts[t] = log_post_prop
            n_accept += 1
        else:
            log_posts[t] = log_posts[t-1]
        
        chain[t] = theta
    
    return chain[burn_in:], log_posts[burn_in:], n_accept / (n_samples - 1)


def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x,
                       x0, y0, n_samples=10000, burn_in=2000):
    """
    Gibbs sampling for 2D distributions.
    
    Alternately sample:
        x ~ P(x|y)
        y ~ P(y|x)
    
    Parameters
    ----------
    conditional_x_given_y : callable(y) — returns sample x ~ P(x|y)
    conditional_y_given_x : callable(x) — returns sample y ~ P(y|x)
    
    Returns
    -------
    samples : (n, 2) array of (x, y) samples
    """
    samples = np.zeros((n_samples, 2))
    x, y = x0, y0
    
    for t in range(n_samples):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        samples[t] = [x, y]
    
    return samples[burn_in:]


def hmc(log_prob, grad_log_prob, x0, step_size=0.01, n_leapfrog=20,
         n_samples=5000, burn_in=1000):
    """
    Hamiltonian Monte Carlo (HMC).
    
    Augment parameter space with momentum p:
        H(x, p) = -log π(x) + p^T p / 2
    
    Use leapfrog integration to propose distant states.
    
    Advantages over random-walk MH:
    - Uses gradient → moves efficiently through parameter space
    - Low correlation between samples
    - High acceptance rate even in high dimensions
    """
    x = np.array(x0, dtype=float)
    dim = len(x)
    
    samples = np.zeros((n_samples, dim))
    samples[0] = x
    n_accept = 0
    current_log_prob = log_prob(x)
    
    for t in range(1, n_samples):
        # Sample momentum
        p = np.random.randn(dim)
        current_p = p.copy()
        current_x = x.copy()
        
        # Leapfrog integration
        p = p + 0.5 * step_size * grad_log_prob(x)  # Half step momentum
        
        for step in range(n_leapfrog - 1):
            x = x + step_size * p
            p = p + step_size * grad_log_prob(x)
        
        x = x + step_size * p
        p = p + 0.5 * step_size * grad_log_prob(x)  # Half step
        p = -p  # Negate for reversibility
        
        # Metropolis correction
        proposed_log_prob = log_prob(x)
        current_K = 0.5 * np.dot(current_p, current_p)
        proposed_K = 0.5 * np.dot(p, p)
        
        log_alpha = (proposed_log_prob - current_log_prob + current_K - proposed_K)
        
        if np.log(np.random.rand()) < log_alpha:
            current_log_prob = proposed_log_prob
            n_accept += 1
        else:
            x = current_x
        
        samples[t] = x
    
    return samples[burn_in:], n_accept / (n_samples - 1)


# ============================================================
# Diagnostics
# ============================================================
def autocorrelation(samples, max_lag=100):
    """Compute autocorrelation function of a 1D chain."""
    n = len(samples)
    mean = np.mean(samples)
    var = np.var(samples)
    acf = np.zeros(min(max_lag, n))
    
    for lag in range(len(acf)):
        acf[lag] = np.mean((samples[:n-lag] - mean) * (samples[lag:] - mean)) / var
    
    return acf


def effective_sample_size(samples):
    """
    Estimate effective sample size (ESS).
    
    ESS = N / (1 + 2 Σ ρ(k))
    
    where ρ(k) is the autocorrelation at lag k.
    """
    n = len(samples)
    acf = autocorrelation(samples, max_lag=min(500, n//2))
    
    # Sum until autocorrelation goes negative or sum < 0
    tau = 1.0
    for k in range(1, len(acf)):
        if acf[k] < 0:
            break
        tau += 2 * acf[k]
    
    return n / tau


def gelman_rubin(chains):
    """
    Gelman-Rubin R̂ diagnostic for convergence.
    
    Compare between-chain and within-chain variance.
    R̂ ≈ 1 indicates convergence. R̂ > 1.1 suggests issues.
    
    Parameters
    ----------
    chains : list of arrays, each shape (n_samples,)
    """
    m = len(chains)
    n = min(len(c) for c in chains)
    chains = [c[:n] for c in chains]
    
    chain_means = [np.mean(c) for c in chains]
    chain_vars = [np.var(c, ddof=1) for c in chains]
    
    overall_mean = np.mean(chain_means)
    B = n * np.var(chain_means, ddof=1)  # Between-chain variance
    W = np.mean(chain_vars)              # Within-chain variance
    
    var_hat = (1 - 1/n) * W + (1/n) * B
    R_hat = np.sqrt(var_hat / W)
    
    return R_hat


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MCMC FOR BAYESIAN INFERENCE DEMO")
    print("=" * 60)

    # --- Example 1: Linear regression Bayesian inference ---
    print("\n--- Example 1: Bayesian Linear Regression ---")
    print("Model: y = a·x + b + ε,  ε ~ N(0, σ²)")
    print("Priors: a ~ N(0, 10), b ~ N(0, 10), σ ~ HalfNormal(5)")
    
    np.random.seed(42)
    # True parameters
    a_true, b_true, sigma_true = 2.5, -1.0, 0.8
    n_data = 30
    x_data = np.random.uniform(0, 5, n_data)
    y_data = a_true * x_data + b_true + sigma_true * np.random.randn(n_data)
    
    def log_likelihood_lr(theta):
        a, b, log_sigma = theta
        sigma = np.exp(log_sigma)
        residuals = y_data - (a * x_data + b)
        return -0.5 * n_data * np.log(2*np.pi*sigma**2) - 0.5 * np.sum(residuals**2) / sigma**2
    
    def log_prior_lr(theta):
        a, b, log_sigma = theta
        # a ~ N(0, 10), b ~ N(0, 10), sigma ~ HalfNormal(5) → log_sigma has Jacobian
        lp = -0.5 * (a**2/100 + b**2/100)  # Gaussian priors
        sigma = np.exp(log_sigma)
        lp += -0.5 * sigma**2 / 25 + log_sigma  # HalfNormal + Jacobian
        return lp
    
    theta0 = np.array([1.0, 0.0, np.log(1.0)])
    samples_lr, log_posts, acc_rate = mcmc_bayesian(
        log_likelihood_lr, log_prior_lr, theta0,
        proposal_std=0.05, n_samples=50000, burn_in=10000
    )
    
    print(f"Acceptance rate: {acc_rate:.3f}")
    a_samples = samples_lr[:, 0]
    b_samples = samples_lr[:, 1]
    sigma_samples = np.exp(samples_lr[:, 2])
    
    print(f"a: {np.mean(a_samples):.3f} ± {np.std(a_samples):.3f} (true: {a_true})")
    print(f"b: {np.mean(b_samples):.3f} ± {np.std(b_samples):.3f} (true: {b_true})")
    print(f"σ: {np.mean(sigma_samples):.3f} ± {np.std(sigma_samples):.3f} (true: {sigma_true})")

    # --- Example 2: Gibbs sampling from bivariate normal ---
    print("\n--- Example 2: Gibbs Sampling — Bivariate Normal ---")
    rho = 0.8
    
    cond_x = lambda y: np.random.normal(rho * y, np.sqrt(1 - rho**2))
    cond_y = lambda x: np.random.normal(rho * x, np.sqrt(1 - rho**2))
    
    gibbs_samples = gibbs_sampling_2d(cond_x, cond_y, 0, 0, n_samples=10000, burn_in=1000)
    print(f"Sample correlation: {np.corrcoef(gibbs_samples[:, 0], gibbs_samples[:, 1])[0,1]:.3f} "
          f"(true: {rho})")
    print(f"Marginal x: mean={np.mean(gibbs_samples[:, 0]):.3f}, "
          f"std={np.std(gibbs_samples[:, 0]):.3f}")

    # --- Example 3: HMC ---
    print("\n--- Example 3: Hamiltonian Monte Carlo ---")
    
    # 10D correlated Gaussian
    dim = 10
    np.random.seed(123)
    # Generate random covariance
    A = np.random.randn(dim, dim) * 0.3
    Sigma = A @ A.T + np.eye(dim)
    Sigma_inv = np.linalg.inv(Sigma)
    
    log_prob_gauss = lambda x: -0.5 * x @ Sigma_inv @ x
    grad_log_prob_gauss = lambda x: -Sigma_inv @ x
    
    hmc_samples, hmc_acc = hmc(log_prob_gauss, grad_log_prob_gauss,
                                np.zeros(dim), step_size=0.15,
                                n_leapfrog=15, n_samples=5000, burn_in=500)
    
    print(f"HMC acceptance rate: {hmc_acc:.3f}")
    cov_estimated = np.cov(hmc_samples.T)
    print(f"Covariance error (Frobenius): {np.linalg.norm(cov_estimated - Sigma):.3f}")

    # --- Diagnostics ---
    print("\n--- Diagnostics ---")
    ess_a = effective_sample_size(a_samples)
    print(f"ESS for parameter 'a': {ess_a:.0f} / {len(a_samples)}")
    
    # Gelman-Rubin (run multiple chains)
    chains_a = []
    for _ in range(4):
        theta0_rand = theta0 + 0.5 * np.random.randn(3)
        s, _, _ = mcmc_bayesian(log_likelihood_lr, log_prior_lr, theta0_rand,
                                 proposal_std=0.05, n_samples=20000, burn_in=5000)
        chains_a.append(s[:, 0])
    
    R_hat = gelman_rubin(chains_a)
    print(f"Gelman-Rubin R̂ for 'a': {R_hat:.4f} (target: ≈ 1.0)")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Posterior of 'a'
        axes[0, 0].hist(a_samples, bins=50, density=True, alpha=0.7, color='steelblue')
        axes[0, 0].axvline(a_true, color='r', linewidth=2, label=f'True a = {a_true}')
        axes[0, 0].set_xlabel('a')
        axes[0, 0].set_title('Posterior: slope a')
        axes[0, 0].legend()
        
        # Trace plot
        axes[0, 1].plot(a_samples[:2000], 'b-', alpha=0.5, linewidth=0.5)
        axes[0, 1].axhline(a_true, color='r', linewidth=1)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('a')
        axes[0, 1].set_title('Trace Plot: a')
        
        # Autocorrelation
        acf = autocorrelation(a_samples, max_lag=100)
        axes[0, 2].bar(range(len(acf)), acf, color='steelblue', alpha=0.7)
        axes[0, 2].set_xlabel('Lag')
        axes[0, 2].set_ylabel('ACF')
        axes[0, 2].set_title(f'Autocorrelation (ESS={ess_a:.0f})')
        
        # Gibbs 2D
        axes[1, 0].scatter(gibbs_samples[:, 0], gibbs_samples[:, 1],
                          s=1, alpha=0.3, c='steelblue')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        axes[1, 0].set_title(f'Gibbs: Bivariate Normal (ρ={rho})')
        axes[1, 0].set_aspect('equal')
        
        # Data + posterior predictive
        axes[1, 1].scatter(x_data, y_data, c='k', s=20, label='Data')
        x_pred = np.linspace(0, 5, 100)
        for i in range(100):
            idx = np.random.randint(len(a_samples))
            y_pred = a_samples[idx] * x_pred + b_samples[idx]
            axes[1, 1].plot(x_pred, y_pred, 'b-', alpha=0.05)
        axes[1, 1].plot(x_pred, a_true*x_pred + b_true, 'r-', linewidth=2, label='True')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        axes[1, 1].set_title('Posterior Predictive')
        axes[1, 1].legend()
        
        # HMC vs MH comparison on 2D marginal
        axes[1, 2].scatter(hmc_samples[:, 0], hmc_samples[:, 1], s=2, alpha=0.3, label='HMC')
        axes[1, 2].set_xlabel('θ₁')
        axes[1, 2].set_ylabel('θ₂')
        axes[1, 2].set_title('HMC Samples (10D Gaussian, 2D marginal)')
        axes[1, 2].set_aspect('equal')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("mcmc_bayesian.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
