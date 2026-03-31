"""
Truncation & Round-off Error Analysis
========================================
Understanding and quantifying numerical errors.

**Truncation Error:**
Arises from approximating continuous objects (derivatives, integrals)
with discrete formulas. Controlled by step size h.

Example: forward difference  f'(x) ≈ (f(x+h) - f(x))/h  →  error O(h)
         central difference  f'(x) ≈ (f(x+h) - f(x-h))/(2h)  →  error O(h²)

**Round-off Error:**
Arises from finite precision of floating-point numbers.
IEEE 754 double: 64 bits → ~15-16 significant digits.
Machine epsilon ε_mach ≈ 2.2×10⁻¹⁶.

Total error = truncation + round-off:
- Truncation ∝ h^p  (decreases with smaller h)
- Round-off ∝ ε/h   (increases with smaller h)
- Optimal h: balance the two → h_opt ≈ ε^{1/(p+1)}

**Catastrophic Cancellation:**
Subtracting nearly equal numbers amplifies relative error.
Example: (1 + 10⁻¹⁵) - 1 → loses 15 digits of precision.

**Condition Number:**
Measures sensitivity of output to input perturbations.
κ(A) = ||A|| · ||A⁻¹|| for matrices.
Large κ → ill-conditioned → small input changes cause large output changes.
"""

import numpy as np


def forward_difference(f, x, h):
    """f'(x) ≈ (f(x+h) - f(x)) / h — O(h) truncation error."""
    return (f(x + h) - f(x)) / h


def central_difference(f, x, h):
    """f'(x) ≈ (f(x+h) - f(x-h)) / (2h) — O(h²) truncation error."""
    return (f(x + h) - f(x - h)) / (2 * h)


def second_derivative_cd(f, x, h):
    """f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h² — O(h²)."""
    return (f(x + h) - 2*f(x) + f(x - h)) / h**2


def richardson_extrapolation(f, x, h, p=2):
    """
    Richardson extrapolation: combine two estimates to eliminate
    leading error term.
    
    D(h) ≈ f'(x) + c·h^p + ...
    
    D_rich = (2^p · D(h/2) - D(h)) / (2^p - 1)
    
    This gives O(h^{p+2}) accuracy from O(h^p) estimates.
    """
    D_h = central_difference(f, x, h)
    D_h2 = central_difference(f, x, h/2)
    
    return (2**p * D_h2 - D_h) / (2**p - 1)


def analyze_differentiation_error(f, f_prime_exact, x0):
    """
    Study how truncation and round-off errors trade off
    as h varies from 1 to 10⁻¹⁶.
    
    Returns h values, forward diff errors, central diff errors.
    """
    h_values = np.logspace(0, -16, 100)
    exact = f_prime_exact(x0)
    
    err_fwd = []
    err_cen = []
    
    for h in h_values:
        d_fwd = forward_difference(f, x0, h)
        d_cen = central_difference(f, x0, h)
        
        err_fwd.append(abs(d_fwd - exact))
        err_cen.append(abs(d_cen - exact))
    
    return h_values, np.array(err_fwd), np.array(err_cen)


def demonstrate_catastrophic_cancellation():
    """
    Show how subtracting nearly-equal numbers causes precision loss.
    """
    print("  Catastrophic cancellation examples:")
    
    # Example 1: (1 + δ) - 1
    for k in range(1, 17):
        delta = 10**(-k)
        result = (1.0 + delta) - 1.0
        relative_error = abs(result - delta) / delta if delta != 0 else 0
        print(f"    δ=1e-{k:2d}: (1+δ)-1 = {result:.16e}, rel.err = {relative_error:.2e}")
    
    # Example 2: Quadratic formula
    print("\n  Quadratic formula: x² - 10⁶x + 1 = 0")
    a, b, c = 1, -1e6, 1
    
    # Standard formula (one root loses precision)
    x1_std = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    x2_std = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    
    # Stable formula using identity x₁·x₂ = c/a
    x1_stable = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    x2_stable = c / (a * x1_stable)
    
    print(f"    Standard:  x₁ = {x1_std:.15e}")
    print(f"               x₂ = {x2_std:.15e}")
    print(f"    Stable:    x₁ = {x1_stable:.15e}")
    print(f"               x₂ = {x2_stable:.15e}")
    print(f"    x₁·x₂ should = {c/a}")
    print(f"    Standard: {x1_std*x2_std:.15e}")
    print(f"    Stable:   {x1_stable*x2_stable:.15e}")


def condition_number_analysis():
    """
    Demonstrate how condition number affects solution accuracy.
    """
    print("  Condition number effect on Ax = b:")
    
    for kappa in [1, 10, 100, 1e4, 1e8, 1e12]:
        n = 10
        # Create matrix with specified condition number
        U, _ = np.linalg.qr(np.random.randn(n, n))
        s = np.logspace(0, np.log10(kappa), n)
        A = U @ np.diag(s) @ U.T
        
        x_true = np.ones(n)
        b = A @ x_true
        
        # Solve with perturbation
        b_pert = b + 1e-10 * np.random.randn(n)
        x_pert = np.linalg.solve(A, b_pert)
        
        rel_err = np.linalg.norm(x_pert - x_true) / np.linalg.norm(x_true)
        print(f"    κ = {kappa:.0e}: ||Δx||/||x|| = {rel_err:.2e}")


def floating_point_properties():
    """Display key floating-point properties."""
    import sys
    
    eps = np.finfo(float).eps
    tiny = np.finfo(float).tiny
    huge = np.finfo(float).max
    
    print(f"  Machine epsilon (ε): {eps}")
    print(f"  Smallest normal:     {tiny}")
    print(f"  Largest value:       {huge}")
    print(f"  Precision (digits):  {int(-np.log10(eps))}")
    
    # Demonstrate ε
    print(f"\n  1.0 + ε/2 == 1.0? {1.0 + eps/2 == 1.0}")
    print(f"  1.0 + ε   == 1.0? {1.0 + eps == 1.0}")
    
    # Demonstrate associativity failure
    a = 1e20
    b = -1e20
    c = 1.0
    print(f"\n  Associativity failure:")
    print(f"  (a + b) + c = {(a + b) + c}")
    print(f"  a + (b + c) = {a + (b + c)}")


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TRUNCATION & ROUND-OFF ERROR ANALYSIS")
    print("=" * 60)

    # TO TEST: Compare float precision behavior by changing test magnitudes (a, b, c) and epsilon-scale checks.
    # Observe loss of associativity and thresholds where arithmetic starts collapsing to identical values.
    # --- Floating-point properties ---
    print("\n--- IEEE 754 Double Precision ---")
    floating_point_properties()

    # TO TEST: Adjust delta range in cancellation loop and quadratic coefficients (a,b,c) near ill-conditioned roots.
    # Observe relative-error blowup and the stability gain from the alternative root formula.
    # --- Catastrophic cancellation ---
    print("\n--- Catastrophic Cancellation ---")
    demonstrate_catastrophic_cancellation()

    # TO TEST: Swap f/fp pairs (e.g., exp, cos), change x0, and extend h sweep bounds to probe truncation vs round-off crossover.
    # Observe U-shaped error curves and how optimal h shifts with formula order.
    # --- Differentiation error analysis ---
    print("\n--- Differentiation Error vs Step Size ---")
    f = np.sin
    fp = np.cos
    x0 = 1.0
    
    h_vals, err_fwd, err_cen = analyze_differentiation_error(f, fp, x0)
    
    # Find optimal h
    i_opt_fwd = np.argmin(err_fwd)
    i_opt_cen = np.argmin(err_cen)
    print(f"Forward diff: optimal h ≈ {h_vals[i_opt_fwd]:.2e}, "
          f"min error = {err_fwd[i_opt_fwd]:.2e}")
    print(f"Central diff: optimal h ≈ {h_vals[i_opt_cen]:.2e}, "
          f"min error = {err_cen[i_opt_cen]:.2e}")
    print(f"Theoretical optimal h (fwd): ε^(1/2) ≈ {np.sqrt(np.finfo(float).eps):.2e}")
    print(f"Theoretical optimal h (cen): ε^(1/3) ≈ {np.finfo(float).eps**(1/3):.2e}")

    # Richardson extrapolation
    h_test = 0.01
    d_cen = central_difference(f, x0, h_test)
    d_rich = richardson_extrapolation(f, x0, h_test)
    print(f"\nRichardson extrapolation (h={h_test}):")
    print(f"  Central diff error:  {abs(d_cen - fp(x0)):.2e}")
    print(f"  Richardson error:    {abs(d_rich - fp(x0)):.2e}")

    # TO TEST: Vary perturbation scale, matrix size n, and kappa range in condition_number_analysis.
    # Observe approximate proportionality between solution sensitivity and condition number.
    # --- Condition number ---
    print("\n--- Condition Number Analysis ---")
    np.random.seed(42)
    condition_number_analysis()

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Error vs h
        axes[0, 0].loglog(h_vals, err_fwd, 'b-', label='Forward O(h)')
        axes[0, 0].loglog(h_vals, err_cen, 'r-', label='Central O(h²)')
        # Theoretical lines
        eps = np.finfo(float).eps
        axes[0, 0].loglog(h_vals, h_vals, 'b--', alpha=0.3, label='h')
        axes[0, 0].loglog(h_vals, h_vals**2, 'r--', alpha=0.3, label='h²')
        axes[0, 0].loglog(h_vals, eps/h_vals, 'k--', alpha=0.3, label='ε/h')
        axes[0, 0].set_xlabel('Step size h')
        axes[0, 0].set_ylabel('|Error|')
        axes[0, 0].set_title('Truncation + Round-off Error Trade-off')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(1e-16, 10)
        
        # Catastrophic cancellation
        deltas = [10**(-k) for k in range(1, 17)]
        results = [(1.0 + d) - 1.0 for d in deltas]
        rel_errors = [abs(r - d)/d for r, d in zip(results, deltas)]
        
        axes[0, 1].semilogy(range(1, 17), rel_errors, 'ro-', markersize=5)
        axes[0, 1].set_xlabel('-log₁₀(δ)')
        axes[0, 1].set_ylabel('Relative error')
        axes[0, 1].set_title('Cancellation Error: (1+δ) - 1')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Condition number effect
        kappas = [1, 10, 100, 1e3, 1e4, 1e6, 1e8, 1e10, 1e12]
        rel_errs = []
        np.random.seed(42)
        for kappa in kappas:
            n = 10
            U, _ = np.linalg.qr(np.random.randn(n, n))
            s = np.logspace(0, np.log10(kappa), n)
            A = U @ np.diag(s) @ U.T
            x_true = np.ones(n)
            b = A @ x_true
            b_pert = b + 1e-10 * np.random.randn(n)
            x_pert = np.linalg.solve(A, b_pert)
            rel_errs.append(np.linalg.norm(x_pert - x_true) / np.linalg.norm(x_true))
        
        axes[1, 0].loglog(kappas, rel_errs, 'bs-', markersize=6)
        axes[1, 0].loglog(kappas, [k*1e-10 for k in kappas], 'r--', 
                          alpha=0.5, label='κ·||Δb||/||b||')
        axes[1, 0].set_xlabel('Condition Number κ(A)')
        axes[1, 0].set_ylabel('Relative Error ||Δx||/||x||')
        axes[1, 0].set_title('Condition Number Effect')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Richardson extrapolation convergence
        hs = np.logspace(-1, -6, 50)
        err_cen_conv = [abs(central_difference(f, x0, h) - fp(x0)) for h in hs]
        err_rich_conv = [abs(richardson_extrapolation(f, x0, h) - fp(x0)) for h in hs]
        
        axes[1, 1].loglog(hs, err_cen_conv, 'b-', label='Central diff O(h²)')
        axes[1, 1].loglog(hs, err_rich_conv, 'r-', label='Richardson O(h⁴)')
        axes[1, 1].loglog(hs, hs**2, 'b--', alpha=0.3)
        axes[1, 1].loglog(hs, hs**4, 'r--', alpha=0.3)
        axes[1, 1].set_xlabel('h')
        axes[1, 1].set_ylabel('|Error|')
        axes[1, 1].set_title('Richardson Extrapolation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("truncation_roundoff.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
