"""
Fixed-Point Iteration
======================
Rewrite f(x) = 0 as x = g(x), then iterate: x_{n+1} = g(x_n).

If |g'(x*)| < 1 at the fixed point x*, the iteration converges.
If |g'(x*)| > 1, it diverges.

Convergence rate:
- Linear: error_{n+1} ≈ g'(x*) · error_n
- The smaller |g'(x*)|, the faster the convergence.
- If g'(x*) = 0, convergence is quadratic (Newton's method is a special case!).

Many methods are special cases of fixed-point iteration:
- Newton: g(x) = x - f(x)/f'(x)
- Jacobi: each step is a fixed-point update
- Picard iteration for ODEs

Physics applications:
- Self-consistent field methods (Hartree-Fock).
- Iterative convergence of coupled equations.
"""

import numpy as np


def fixed_point_iteration(g, x0, tol=1e-12, max_iter=1000):
    """
    Fixed-point iteration: x_{n+1} = g(x_n).
    
    Parameters
    ----------
    g : callable – iteration function
    x0 : float – initial guess
    tol : float – tolerance |x_{n+1} - x_n|
    max_iter : int
    
    Returns
    -------
    x : float – fixed point
    info : dict
    """
    x = x0
    history = [x]
    
    for k in range(max_iter):
        try:
            x_new = g(x)
            
            # Check for divergence (value too large)
            if abs(x_new) > 1e10:
                return x, {
                    'iterations': k + 1,
                    'history': history,
                    'converged': False,
                    'error': 'Divergence detected: |x| > 1e10'
                }
            
            history.append(x_new)
            
            if abs(x_new - x) < tol:
                return x_new, {
                    'iterations': k + 1,
                    'history': history,
                    'converged': True
                }
            
            x = x_new
        except (OverflowError, ValueError) as e:
            return x, {
                'iterations': k + 1,
                'history': history,
                'converged': False,
                'error': f'Numerical error: {type(e).__name__}'
            }
    
    return x, {'iterations': max_iter, 'history': history, 'converged': False}


def aitken_acceleration(g, x0, tol=1e-12, max_iter=1000):
    """
    Aitken's Δ² acceleration (Steffensen's method).
    
    Accelerates a linearly convergent sequence to quadratic convergence!
    
    Given three iterates x_n, x_{n+1}, x_{n+2}, the accelerated value is:
        x̃ = x_n - (x_{n+1} - x_n)² / (x_{n+2} - 2x_{n+1} + x_n)
    """
    x = x0
    history = [x]
    
    for k in range(max_iter):
        x1 = g(x)
        x2 = g(x1)
        
        denom = x2 - 2*x1 + x
        if abs(denom) < 1e-16:
            return x2, {'iterations': k + 1, 'history': history, 'converged': True}
        
        # Aitken's formula
        x_new = x - (x1 - x)**2 / denom
        history.append(x_new)
        
        if abs(x_new - x) < tol:
            return x_new, {'iterations': k + 1, 'history': history, 'converged': True}
        
        x = x_new
    
    return x, {'iterations': max_iter, 'history': history, 'converged': False}


def check_convergence_condition(g, x_star, h=1e-6):
    """
    Check the Lipschitz/contraction condition |g'(x*)| < 1.
    
    Returns the estimated |g'(x*)| and whether convergence is expected.
    """
    gp = abs((g(x_star + h) - g(x_star - h)) / (2 * h))
    return gp, gp < 1


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FIXED-POINT ITERATION DEMO")
    print("=" * 60)

    # --- Example 1: cos(x) = x (fixed point of cosine) ---
    print("\n--- x = cos(x) (the Dottie number) ---")
    g1 = lambda x: np.cos(x)
    
    root1, info1 = fixed_point_iteration(g1, 0.5)
    gp, conv = check_convergence_condition(g1, root1)
    print(f"Fixed point: {root1:.15f}")
    print(f"|g'(x*)|: {gp:.6f} ({'converges' if conv else 'diverges'})")
    print(f"Iterations: {info1['iterations']}")

    # --- Example 2: Different formulations of same equation ---
    print("\n--- x³ = x + 2, rearranged different ways ---")
    # f(x) = x³ - x - 2 = 0
    # Formulation A: x = (x + 2)^{1/3}
    gA = lambda x: (x + 2)**(1/3)
    rA, iA = fixed_point_iteration(gA, 1.5)
    gpA, cA = check_convergence_condition(gA, rA)
    print(f"  g(x) = (x+2)^(1/3): root = {rA:.12f}, iters = {iA['iterations']}, "
          f"|g'| = {gpA:.4f}")

    # Formulation B: x = x³ - 2 (diverges!)
    gB = lambda x: x**3 - 2
    rB, iB = fixed_point_iteration(gB, 1.5, max_iter=20)
    if iB['converged']:
        gpB, cB = check_convergence_condition(gB, rB)
        print(f"  g(x) = x³ - 2:      root = {rB:.12f}, iters = {iB['iterations']}, "
              f"|g'| = {gpB:.4f}")
    else:
        error_msg = iB.get('error', 'Did not converge')
        print(f"  g(x) = x³ - 2:      {error_msg}, iters = {iB['iterations']}")

    # --- Aitken acceleration ---
    print("\n--- Aitken Acceleration ---")
    r_plain, i_plain = fixed_point_iteration(g1, 0.5)
    r_aitken, i_aitken = aitken_acceleration(g1, 0.5)
    print(f"  Plain:  {i_plain['iterations']} iterations")
    print(f"  Aitken: {i_aitken['iterations']} iterations (accelerated!)")

    # --- Self-consistent equation (physics-like) ---
    print("\n--- Self-consistent: x = 1 + 0.5·sin(x) ---")
    g_sc = lambda x: 1 + 0.5 * np.sin(x)
    r_sc, i_sc = fixed_point_iteration(g_sc, 1.0)
    gp_sc, _ = check_convergence_condition(g_sc, r_sc)
    print(f"  Fixed point: {r_sc:.12f}")
    print(f"  |g'(x*)|: {gp_sc:.6f}")
    print(f"  Iterations: {i_sc['iterations']}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Cobweb diagram for x = cos(x)
        x_line = np.linspace(0, 1.5, 200)
        axes[0].plot(x_line, np.cos(x_line), 'b-', lw=2, label='g(x) = cos(x)')
        axes[0].plot(x_line, x_line, 'k--', lw=1, label='y = x')
        
        # Draw cobweb
        x = 0.5
        for _ in range(15):
            x_new = np.cos(x)
            axes[0].plot([x, x], [x, x_new], 'r-', alpha=0.5, linewidth=0.8)
            axes[0].plot([x, x_new], [x_new, x_new], 'r-', alpha=0.5, linewidth=0.8)
            x = x_new
        
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('g(x)')
        axes[0].set_title('Cobweb Diagram: x = cos(x)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Convergence comparison: plain vs Aitken
        errors_plain = [abs(x - root1) for x in i_plain['history'] if abs(x - root1) > 1e-16]
        errors_aitken = [abs(x - root1) for x in i_aitken['history'] if abs(x - root1) > 1e-16]
        
        axes[1].semilogy(range(len(errors_plain)), errors_plain, 'bo-', label='Plain', markersize=4)
        axes[1].semilogy(range(len(errors_aitken)), errors_aitken, 'rs-', label='Aitken', markersize=4)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('|Error|')
        axes[1].set_title('Aitken Acceleration Effect')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Convergence/divergence regions
        x_range = np.linspace(-2, 3, 300)
        axes[2].plot(x_range, np.abs(np.gradient(gA(x_range), x_range)), 'b-', label="|g'_A(x)|")
        axes[2].axhline(y=1, color='r', linestyle='--', label='|g\'| = 1 boundary')
        axes[2].fill_between(x_range, 0, 1, alpha=0.1, color='green', label='Convergence region')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel("|g'(x)|")
        axes[2].set_title('Convergence Condition')
        axes[2].set_ylim(0, 3)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("fixed_point_iteration.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
