"""
Bisection Method
=================
The simplest and most robust root-finding algorithm.

Given f continuous on [a, b] with f(a)·f(b) < 0 (sign change),
there must be a root in [a, b] (Intermediate Value Theorem).

Algorithm: Repeatedly halve the interval, keeping the half where
the sign change occurs.

Properties:
- ALWAYS converges (guaranteed by IVT).
- Linear convergence: gains ~1 bit per iteration.
- After k iterations: error ≤ (b-a)/2^k.
- Number of iterations for tolerance ε: k = ⌈log₂((b-a)/ε)⌉.
- No derivative information needed.
- Slow compared to Newton, but bulletproof.
"""

import numpy as np


def bisection(f, a, b, tol=1e-12, max_iter=100):
    """
    Bisection method for finding a root of f(x) = 0 in [a, b].
    
    Parameters
    ----------
    f : callable – function
    a, b : float – bracket endpoints (must have f(a)·f(b) < 0)
    tol : float – tolerance on interval width
    max_iter : int – maximum iterations
    
    Returns
    -------
    root : float – approximate root
    info : dict – convergence information
    """
    fa, fb = f(a), f(b)
    
    if fa * fb > 0:
        raise ValueError(f"f(a)={fa:.4e} and f(b)={fb:.4e} have the same sign. "
                         "No guaranteed root in [a, b].")
    
    if abs(fa) < 1e-16:
        return a, {'iterations': 0, 'history': [(a, 0)]}
    if abs(fb) < 1e-16:
        return b, {'iterations': 0, 'history': [(b, 0)]}
    
    history = []
    
    for k in range(max_iter):
        c = (a + b) / 2          # Midpoint
        fc = f(c)
        interval = b - a
        history.append((c, interval))
        
        if abs(fc) < 1e-16 or interval < tol:
            return c, {'iterations': k + 1, 'history': history}
        
        # Choose the half with the sign change
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    c = (a + b) / 2
    return c, {'iterations': max_iter, 'history': history}


def bisection_count(a, b, tol):
    """Predict the number of iterations needed for given tolerance."""
    return int(np.ceil(np.log2((b - a) / tol)))


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BISECTION METHOD DEMO")
    print("=" * 60)

    # --- Example 1: Simple root ---
    # TO TEST: Change bracket [1, 2] to [0, 2] or [1, 3], and observe that convergence is guaranteed only when f(a)*f(b) < 0.
    # IMPORTANT: Bisection requires f(a) and f(b) to have OPPOSITE signs
    # Try: f(1) = 1 - 1 - 2 = -2 (negative), f(2) = 8 - 2 - 2 = 4 (positive) ✓
    # To experiment: Try changing [1, 2] to other intervals like [0, 2] or [1, 3]
    # as long as they bracket a root (opposite signs at endpoints)
    print("\n--- f(x) = x³ - x - 2, root near x ≈ 1.5214 ---")
    f = lambda x: x**3 - x - 2
    
    root, info = bisection(f, 1, 2)
    print(f"Root: {root:.15f}")
    print(f"f(root): {f(root):.2e}")
    print(f"Iterations: {info['iterations']}")
    print(f"Predicted iterations: {bisection_count(1, 2, 1e-12)}")

    # --- Example 2: Transcendental equation ---
    # TO TEST: Replace [0, pi/2] with [0, 1] or [1, 3], and observe success/failure based on endpoint sign change.
    # IMPORTANT: The interval [0, π/2] is REQUIRED (not [1, 2]!)
    # Check: f(0) = cos(0) - 0 = 1 (positive), f(π/2) ≈ -1.57 (negative) ✓
    # Common mistake: [1, 2] gives f(1) ≈ -0.46 (negative), f(2) ≈ -2.42 (negative) ✗
    # To experiment: Try [0, 1] or [1, 3] and check the signs first!
    print("\n--- f(x) = cos(x) - x (fixed point of cos) ---")
    f2 = lambda x: np.cos(x) - x
    
    root2, info2 = bisection(f2, 0, np.pi/2)
    print(f"Root: {root2:.15f}")
    print(f"Verification: cos(root) = {np.cos(root2):.15f}")
    print(f"Iterations: {info2['iterations']}")

    # --- Example 3: Multiple roots ---
    # TO TEST: Modify interval list [(2,4), (5,7), (-1,1)] to ranges like [(-3,-2), (3,5)] and observe which k*pi root each bracket converges to.
    # sin(x) has roots at 0, π, 2π, 3π, ...
    # Each interval below brackets a different root (crossing zero)
    # To experiment: Try [-π, π], [3, 5], or any [a,b] where sin(a)·sin(b) < 0
    print("\n--- f(x) = sin(x), finding different roots ---")
    f3 = lambda x: np.sin(x)
    
    for a, b in [(2, 4), (5, 7), (-1, 1)]:
        r, info = bisection(f3, a, b)
        print(f"  [{a}, {b}]: root = {r:.12f} (nearest kπ = {round(r/np.pi):.0f}π)")

    # --- Convergence visualization ---
    print("\n--- Convergence history ---")
    print(f"{'Iter':>4s} | {'Midpoint':>16s} | {'Interval':>12s}")
    print("-" * 40)
    _, info = bisection(f, 1, 2, tol=1e-15)
    for k, (c, w) in enumerate(info['history'][:15]):
        print(f"{k+1:4d} | {c:16.12f} | {w:12.4e}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Function and root
        x = np.linspace(0, 3, 300)
        axes[0].plot(x, f(x), 'b-', lw=2)
        axes[0].axhline(y=0, color='k', linewidth=0.5)
        axes[0].plot(root, 0, 'ro', markersize=10, label=f'Root = {root:.6f}')
        
        # Show bisection steps
        _, info_vis = bisection(f, 1, 2, tol=1e-3)
        for k, (c, w) in enumerate(info_vis['history']):
            axes[0].axvline(x=c, color='gray', linewidth=0.5, alpha=0.5)
        
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('f(x)')
        axes[0].set_title('f(x) = x³ - x - 2')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Convergence rate
        _, info_conv = bisection(f, 1, 2, tol=1e-15)
        errors = [abs(c - root) for c, _ in info_conv['history']]
        widths = [w for _, w in info_conv['history']]
        
        axes[1].semilogy(range(1, len(errors)+1), errors, 'bo-', label='|error|')
        axes[1].semilogy(range(1, len(widths)+1), widths, 'rs-', label='interval width')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Value')
        axes[1].set_title('Bisection Convergence (Linear)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("bisection_method.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
