"""
Padé Approximants
==================
Rational function approximation: express f(x) ≈ P_M(x) / Q_N(x)
where P and Q are polynomials of degree M and N.

**Why Padé instead of Taylor?**
Taylor series often have limited radius of convergence.
Padé approximants can:
  - Capture poles (Taylor cannot!)
  - Converge outside the radius of convergence
  - Resum divergent perturbation series
  - Give much better accuracy with same number of coefficients

**Construction:**
Given Taylor coefficients c₀, c₁, ...  of f(x) = Σ cₖ xᵏ,
find [M/N] Padé such that:

    P_M(x) / Q_N(x) = c₀ + c₁x + ... + c_{M+N} x^{M+N} + O(x^{M+N+1})

This gives a linear system for the Q coefficients.

**Applications in Physics:**
- Resummation of perturbation theory (QFT, statistical mechanics)
- Analytic continuation of series
- Equation of state from virial coefficients
- Critical exponents from high-temperature expansions
- Padé-based ODE solvers (better stability than Taylor)

**Matrix representation:**
The [M/N] Padé approximant can be computed from a Toeplitz system.

Where to start:
━━━━━━━━━━━━━━
Try approximating e^x and tan(x) — compare Taylor vs Padé.
See how Padé captures the pole of 1/(1-x) beyond |x|<1.
Prerequisite: least_squares_fitting.py, lagrange_interpolation.py
"""

import math
import numpy as np
from scipy.integrate import quad


def pade_approximant(coeffs, M, N):
    """
    Compute [M/N] Padé approximant from Taylor coefficients.
    
    Given f(x) ≈ Σᵢ cᵢ xⁱ, find P_M(x)/Q_N(x) matching the
    first M+N+1 Taylor coefficients.
    
    Parameters
    ----------
    coeffs : array-like
        Taylor coefficients [c₀, c₁, c₂, ...], need at least M+N+1
    M : int
        Degree of numerator
    N : int
        Degree of denominator
    
    Returns
    -------
    p : np.array
        Numerator coefficients [p₀, p₁, ..., p_M]
    q : np.array  
        Denominator coefficients [1, q₁, ..., q_N] (q₀ = 1)
    """
    c = np.array(coeffs, dtype=float)
    
    if len(c) < M + N + 1:
        raise ValueError(f"Need at least {M+N+1} Taylor coefficients, got {len(c)}")
    
    if N == 0:
        # Pure polynomial, no rational part
        return c[:M+1].copy(), np.array([1.0])
    
    # Solve for denominator coefficients q₁, ..., q_N
    # System: Σⱼ q_j · c_{i-j} = 0  for i = M+1, ..., M+N
    # where q₀ = 1
    
    A = np.zeros((N, N))
    rhs = np.zeros(N)
    
    for i in range(N):
        row = M + 1 + i  # equation index
        for j in range(N):
            col_idx = row - (j + 1)
            if 0 <= col_idx < len(c):
                A[i, j] = c[col_idx]
        rhs[i] = -c[row] if row < len(c) else 0.0
    
    # Solve for q₁, ..., q_N
    try:
        q_coeffs = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        q_coeffs = np.linalg.lstsq(A, rhs, rcond=None)[0]
    
    q = np.zeros(N + 1)
    q[0] = 1.0
    q[1:] = q_coeffs
    
    # Compute numerator: p_k = Σⱼ₌₀ᵏ q_j · c_{k-j}  for k = 0, ..., M
    p = np.zeros(M + 1)
    for k in range(M + 1):
        for j in range(min(k, N) + 1):
            p[k] += q[j] * c[k - j]
    
    return p, q


def eval_pade(p, q, x):
    """
    Evaluate Padé approximant P(x)/Q(x).
    
    Parameters
    ----------
    p : array
        Numerator coefficients [p₀, p₁, ..., p_M]
    q : array
        Denominator coefficients [1, q₁, ..., q_N]
    x : float or array
        Evaluation point(s)
    """
    x = np.asarray(x, dtype=float)
    
    # Horner's method for numerator
    num = np.zeros_like(x)
    for k in range(len(p) - 1, -1, -1):
        num = num * x + p[k]
    
    # Horner's method for denominator
    den = np.zeros_like(x)
    for k in range(len(q) - 1, -1, -1):
        den = den * x + q[k]
    
    return num / den


def eval_taylor(coeffs, x, n_terms=None):
    """Evaluate Taylor polynomial Σ cₖ xᵏ."""
    if n_terms is not None:
        coeffs = coeffs[:n_terms]
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    for k in range(len(coeffs) - 1, -1, -1):
        result = result * x + coeffs[k]
    return result


def taylor_coefficients(func_name, n_terms):
    """
    Get Taylor coefficients for common functions.
    
    Parameters
    ----------
    func_name : str
        'exp', 'sin', 'cos', 'log1p', '1/(1-x)', 'tan'
    n_terms : int
        Number of terms
    """
    c = np.zeros(n_terms)
    
    if func_name == 'exp':
        # e^x = Σ xⁿ/n!
        factorial = 1.0
        for k in range(n_terms):
            c[k] = 1.0 / factorial
            factorial *= (k + 1)
    
    elif func_name == 'sin':
        factorial = 1.0
        for k in range(n_terms):
            if k % 2 == 1:
                c[k] = (-1)**((k-1)//2) / factorial
            factorial *= (k + 1)
    
    elif func_name == 'cos':
        factorial = 1.0
        for k in range(n_terms):
            if k % 2 == 0:
                c[k] = (-1)**(k//2) / factorial
            factorial *= (k + 1)
    
    elif func_name == 'log1p':
        # ln(1+x) = x - x²/2 + x³/3 - ...
        for k in range(1, n_terms):
            c[k] = (-1)**(k+1) / k
    
    elif func_name == '1/(1-x)':
        # 1/(1-x) = 1 + x + x² + ...
        c[:] = 1.0
    
    elif func_name == 'tan':
        # Compute via recurrence
        # tan(x) Taylor coefficients up to n terms
        from math import factorial as fact
        # Use numerical differentiation approach
        c[0] = 0.0
        if n_terms > 1: c[1] = 1.0
        if n_terms > 3: c[3] = 1.0/3
        if n_terms > 5: c[5] = 2.0/15
        if n_terms > 7: c[7] = 17.0/315
        if n_terms > 9: c[9] = 62.0/2835
        if n_terms > 11: c[11] = 1382.0/155925
    
    return c


def pade_table(coeffs, max_order=4):
    """
    Compute the Padé table up to given order.
    
    The [M/N] entry uses M+N+1 Taylor coefficients.
    Diagonal entries [N/N] are often the best.
    
    Returns dict of {(M,N): (p, q)}.
    """
    table = {}
    for M in range(max_order + 1):
        for N in range(max_order + 1):
            if M + N + 1 <= len(coeffs):
                p, q = pade_approximant(coeffs, M, N)
                table[(M, N)] = (p, q)
    return table


def pade_poles_zeros(p, q):
    """
    Find poles and zeros of a Padé approximant.
    
    Zeros: roots of P(x) = 0
    Poles: roots of Q(x) = 0
    """
    zeros = np.roots(p[::-1])
    poles = np.roots(q[::-1])
    return zeros, poles


def continued_fraction_pade(coeffs, x):
    """
    Evaluate Padé via continued fraction representation.
    
    More numerically stable for high-order approximants.
    Uses the qd (quotient-difference) algorithm.
    """
    n = len(coeffs)
    c = np.array(coeffs, dtype=float)
    
    if n == 0:
        return 0.0
    if n == 1:
        return c[0] * np.ones_like(np.asarray(x, dtype=float))
    
    x = np.asarray(x, dtype=float)
    
    # Simple approach: compute [n//2 / n//2] or [(n-1)//2 / n//2]
    M = (n - 1) // 2
    N = n // 2
    p, q = pade_approximant(c, M, N)
    return eval_pade(p, q, x)


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PADÉ APPROXIMANTS DEMO")
    print("=" * 60)

    # --- 1. Exponential function ---
    # TO TEST: Modify n (Taylor term count), Padé order [M/N], and x_test range; observe where Padé outperforms truncated Taylor, especially at larger x.
    print("\n--- exp(x): Taylor vs Padé ---")
    n = 7  # Use 7 Taylor coefficients
    c_exp = taylor_coefficients('exp', n)
    
    x_test = np.array([1.0, 2.0, 5.0, 10.0])
    exact = np.exp(x_test)
    
    print(f"{'x':>5} {'Exact':>12} {'Taylor[6]':>12} {'Pade[3/3]':>12} "
          f"{'Taylor err':>12} {'Pade err':>12}")
    
    p33, q33 = pade_approximant(c_exp, 3, 3)
    
    for x in x_test:
        ex = np.exp(x)
        tay = eval_taylor(c_exp, x)
        pad = eval_pade(p33, q33, x)
        print(f"{x:5.1f} {ex:12.4f} {tay:12.4f} {pad:12.4f} "
              f"{abs(tay-ex):12.4e} {abs(pad-ex):12.4e}")

    # --- 2. 1/(1-x): Beyond radius of convergence ---
    # TO TEST: Vary x_test2 values across |x|<1 and |x|>=1 and try different [M/N] orders; observe Taylor divergence and Padé behavior near the pole x=1.
    print("\n--- 1/(1-x): Padé extends beyond |x| < 1 ---")
    c_geom = taylor_coefficients('1/(1-x)', 11)
    
    # [5/5] Padé for 1/(1-x) should give exact result!
    p55, q55 = pade_approximant(c_geom, 5, 5)
    
    x_test2 = np.array([0.5, 0.9, 1.5, 2.0, 5.0])
    print(f"{'x':>5} {'Exact':>10} {'Taylor[9]':>10} {'Pade[5/5]':>12} "
          f"{'Taylor div?':>12}")
    
    for x in x_test2:
        ex = 1.0 / (1.0 - x) if x != 1.0 else float('inf')
        tay = eval_taylor(c_geom, x, 10)
        pad = eval_pade(p55, q55, x)
        divg = "DIVERGES" if abs(x) >= 1 else ""
        print(f"{x:5.1f} {ex:10.4f} {tay:10.4f} {pad:12.4f} {divg:>12}")

    # --- 3. Padé finds poles ---
    # TO TEST: Change approximation order (e.g., [3/3], [4/4], [5/5]) and tolerance for real poles; observe estimated pole locations and numerical stability of root finding.
    print("\n--- Pole detection in Padé ---")
    c_geom = taylor_coefficients('1/(1-x)', 9)
    p44, q44 = pade_approximant(c_geom, 4, 4)
    zeros, poles = pade_poles_zeros(p44, q44)
    print(f"  [4/4] Padé of 1/(1-x):")
    
    real_poles = poles[np.abs(poles.imag) < 1e-10].real
    if len(real_poles) > 0:
        print(f"  Real poles at: {np.sort(real_poles)}")
        print(f"  Exact pole at: x = 1.0")

    # --- 4. tan(x): multi-pole function ---
    print("\n--- tan(x): multiple poles ---")
    c_tan = taylor_coefficients('tan', 12)
    p55_tan, q55_tan = pade_approximant(c_tan, 5, 6)
    
    x_range = np.linspace(-1.4, 1.4, 50)
    exact_tan = np.tan(x_range)
    pade_tan = eval_pade(p55_tan, q55_tan, x_range)
    taylor_tan = eval_taylor(c_tan, x_range, 11)
    
    # Error at x = 1.4 (near π/2 ≈ 1.571)
    x_near_pole = 1.4
    print(f"  At x = {x_near_pole}:")
    print(f"    Exact:     {np.tan(x_near_pole):.6f}")
    print(f"    Taylor[10]: {eval_taylor(c_tan, x_near_pole, 11):.6f}")
    print(f"    Padé[5/6]: {eval_pade(p55_tan, q55_tan, x_near_pole):.6f}")

    # --- 5. Padé table comparison ---
    print("\n--- Padé Table for exp(x) ---")
    c_exp_10 = taylor_coefficients('exp', 10)
    table = pade_table(c_exp_10, max_order=4)
    
    x_eval = 3.0
    exact_val = np.exp(x_eval)
    print(f"\n  Errors at x = {x_eval} (exact = {exact_val:.6f}):")
    print(f"  {'[M/N]':>8} {'Value':>12} {'Error':>12}")
    
    for M in range(5):
        for N in range(5):
            if (M, N) in table:
                p_mn, q_mn = table[(M, N)]
                val = eval_pade(p_mn, q_mn, x_eval)
                err = abs(val - exact_val)
                print(f"  [{M}/{N}]{'':<4} {val:12.6f} {err:12.6e}")

    # --- 6. Resummation example ---
    print("\n--- Resummation of divergent series ---")
    # The series 1 - 1! x + 2! x² - 3! x³ + ...
    # diverges for all x ≠ 0, but corresponds to ∫₀^∞ e^{-t}/(1+xt) dt
    # (Borel-summable)
    
    n_terms = 10
    c_div = np.array([(-1)**k * math.factorial(k) for k in range(n_terms)],
                     dtype=float)
    
    x_eval = 0.5
    exact_int, _ = quad(lambda t: np.exp(-t)/(1+x_eval*t), 0, np.inf)

    print(f"  f(x) ~ Σ (-1)^n n! x^n (divergent for x>0)")
    print(f"  Exact integral at x={x_eval}: {exact_int:.8f}")

    # Padé resummation
    for M, N in [(2,2), (3,3), (4,4), (4,5)]:
        if M + N + 1 <= n_terms:
            p_mn, q_mn = pade_approximant(c_div, M, N)
            val = eval_pade(p_mn, q_mn, x_eval)
            print(f"  Padé [{M}/{N}]: {val:.8f}  (error = {abs(val - exact_int):.2e})")
    
    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # exp(x)
        ax = axes[0, 0]
        x = np.linspace(-1, 5, 200)
        ax.plot(x, np.exp(x), 'k-', linewidth=2, label='exp(x)')
        ax.plot(x, eval_taylor(c_exp, x), 'b--', label='Taylor[6]')
        ax.plot(x, eval_pade(p33, q33, x), 'r-', label='Padé[3/3]')
        ax.set_ylim(-5, 50)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('exp(x): Taylor vs Padé')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 1/(1-x) — beyond radius
        ax = axes[0, 1]
        x = np.linspace(-0.5, 3.0, 200)
        exact_fn = 1.0 / (1.0 - x)
        ax.plot(x, exact_fn, 'k-', linewidth=2, label='1/(1-x)')
        c10 = taylor_coefficients('1/(1-x)', 10)
        ax.plot(x, eval_taylor(c10, x, 10), 'b--', label='Taylor[9]', alpha=0.7)
        p_44, q_44 = pade_approximant(c10, 4, 4)
        ax.plot(x, eval_pade(p_44, q_44, x), 'r-', label='Padé[4/4]')
        ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5, label='pole at x=1')
        ax.set_ylim(-10, 20)
        ax.set_xlabel('x')
        ax.set_title('1/(1-x): Padé captures the pole')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # tan(x)
        ax = axes[0, 2]
        x = np.linspace(-1.5, 1.5, 300)
        ax.plot(x, np.tan(x), 'k-', linewidth=2, label='tan(x)')
        c_t = taylor_coefficients('tan', 12)
        ax.plot(x, eval_taylor(c_t, x, 11), 'b--', label='Taylor[10]', alpha=0.7)
        p_56, q_56 = pade_approximant(c_t, 5, 6)
        ax.plot(x, eval_pade(p_56, q_56, x), 'r-', label='Padé[5/6]')
        ax.set_ylim(-10, 10)
        ax.set_xlabel('x')
        ax.set_title('tan(x): poles at ±π/2')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Error comparison for exp(x)
        ax = axes[1, 0]
        x = np.linspace(0.1, 5, 100)
        for M, N in [(1,1), (2,2), (3,3), (4,4)]:
            p_mn, q_mn = pade_approximant(c_exp_10, M, N)
            err = np.abs(eval_pade(p_mn, q_mn, x) - np.exp(x)) / np.exp(x)
            ax.semilogy(x, err + 1e-16, label=f'[{M}/{N}]')
        for k in [2, 4, 6, 8]:
            err_t = np.abs(eval_taylor(c_exp_10, x, k+1) - np.exp(x)) / np.exp(x)
            ax.semilogy(x, err_t + 1e-16, '--', alpha=0.5, label=f'Taylor[{k}]')
        ax.set_xlabel('x')
        ax.set_ylabel('Relative Error')
        ax.set_title('Padé vs Taylor Error for exp(x)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Padé table heatmap
        ax = axes[1, 1]
        errors = np.full((5, 5), np.nan)
        for M in range(5):
            for N in range(5):
                if (M, N) in table:
                    p_mn, q_mn = table[(M, N)]
                    val = eval_pade(p_mn, q_mn, 3.0)
                    errors[N, M] = np.log10(abs(val - np.exp(3.0)) + 1e-16)
        im = ax.imshow(errors, cmap='RdYlGn_r', aspect='equal')
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xlabel('M (numerator degree)')
        ax.set_ylabel('N (denominator degree)')
        ax.set_title('Padé Table: log₁₀(error) at x=3')
        plt.colorbar(im, ax=ax)
        
        # Divergent series resummation
        ax = axes[1, 2]
        x_range = np.linspace(0.01, 2.0, 100)
        exact_vals = np.array([quad(lambda t: np.exp(-t)/(1+xi*t), 0, np.inf)[0]
                               for xi in x_range])
        ax.plot(x_range, exact_vals, 'k-', linewidth=2, label='Exact')
        for M, N in [(2,2), (3,3), (4,4)]:
            p_mn, q_mn = pade_approximant(c_div, M, N)
            ax.plot(x_range, eval_pade(p_mn, q_mn, x_range), '--',
                   label=f'Padé [{M}/{N}]')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Resummation of Divergent Series')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("pade_approximants.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
