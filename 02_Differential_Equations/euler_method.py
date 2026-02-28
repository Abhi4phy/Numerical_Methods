"""
Euler Method (Forward Euler)
=============================
The simplest ODE solver. Given dy/dt = f(t, y), advance by:
    y_{n+1} = y_n + h * f(t_n, y_n)

Properties:
- First-order accurate: local error O(h²), global error O(h).
- Explicit method — no equations to solve at each step.
- Conditionally stable: step size h must be small enough.
- Not recommended for production, but essential to understand.

Physics applications:
- Projectile motion, simple harmonic oscillator (for illustration).
- Understanding numerical stability and order of accuracy.
"""

import numpy as np


def euler_method(f, t_span, y0, h):
    """
    Forward Euler method for dy/dt = f(t, y).
    
    Parameters
    ----------
    f : callable
        Right-hand side function f(t, y). y can be a scalar or array.
    t_span : tuple (t0, tf)
        Start and end times.
    y0 : float or ndarray
        Initial condition.
    h : float
        Step size.
    
    Returns
    -------
    t : ndarray – time points
    y : ndarray – solution at each time point
    """
    t0, tf = t_span
    t = np.arange(t0, tf + h/2, h)
    n_steps = len(t)

    y0 = np.atleast_1d(np.array(y0, dtype=float))
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0

    for i in range(n_steps - 1):
        y[i + 1] = y[i] + h * np.atleast_1d(f(t[i], y[i]))

    return t, y.squeeze()


def backward_euler(f, df_dy, t_span, y0, h, newton_tol=1e-10, newton_max=50):
    """
    Backward (Implicit) Euler method.
    
    y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
    
    This is an implicit equation solved with Newton's method at each step.
    Advantage: unconditionally stable for stiff problems.
    
    Parameters
    ----------
    df_dy : callable
        Jacobian ∂f/∂y (t, y). For scalar ODE, this is just df/dy.
    """
    t0, tf = t_span
    t = np.arange(t0, tf + h/2, h)
    n_steps = len(t)

    y0 = np.atleast_1d(np.array(y0, dtype=float))
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0
    dim = len(y0)

    for i in range(n_steps - 1):
        # Newton iteration to solve: g(y_{n+1}) = y_{n+1} - y_n - h*f(t_{n+1}, y_{n+1}) = 0
        y_guess = y[i] + h * np.atleast_1d(f(t[i], y[i]))  # Forward Euler as initial guess
        
        for _ in range(newton_max):
            g = y_guess - y[i] - h * np.atleast_1d(f(t[i+1], y_guess))
            J = np.eye(dim) - h * np.atleast_2d(df_dy(t[i+1], y_guess))
            delta = np.linalg.solve(J, -g)
            y_guess = y_guess + delta
            if np.linalg.norm(delta) < newton_tol:
                break

        y[i + 1] = y_guess

    return t, y.squeeze()


# ============================================================
# Demo: Simple Harmonic Oscillator & Exponential Decay
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EULER METHOD DEMO")
    print("=" * 60)

    # --- Example 1: Exponential decay dy/dt = -y ---
    print("\n--- Exponential Decay: dy/dt = -y, y(0) = 1 ---")
    f_exp = lambda t, y: -y
    y_exact_exp = lambda t: np.exp(-t)

    for h in [0.5, 0.1, 0.01]:
        t, y = euler_method(f_exp, (0, 5), 1.0, h)
        error = np.max(np.abs(y - y_exact_exp(t)))
        print(f"  h = {h:5.3f}  |  Max error = {error:.6e}")

    # --- Example 2: Simple Harmonic Oscillator ---
    # x'' + ω²x = 0  →  system: y = [x, v], dy/dt = [v, -ω²x]
    print("\n--- Simple Harmonic Oscillator: x'' + ω²x = 0 ---")
    omega = 2 * np.pi  # period = 1

    def f_sho(t, y):
        return np.array([y[1], -omega**2 * y[0]])

    y0_sho = [1.0, 0.0]  # x(0)=1, v(0)=0

    # Euler: energy should grow (known instability)
    t, y = euler_method(f_sho, (0, 5), y0_sho, h=0.001)
    energy = 0.5 * y[:, 1]**2 + 0.5 * omega**2 * y[:, 0]**2
    print(f"  Forward Euler: E(0)={energy[0]:.4f}, E(5)={energy[-1]:.4f} "
          f"(energy {'grows' if energy[-1] > energy[0] else 'decays'})")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Exponential decay
        for h in [0.5, 0.2, 0.05]:
            t, y = euler_method(f_exp, (0, 5), 1.0, h)
            axes[0].plot(t, y, 'o-', markersize=2, label=f'Euler h={h}')
        t_exact = np.linspace(0, 5, 200)
        axes[0].plot(t_exact, y_exact_exp(t_exact), 'k-', lw=2, label='Exact')
        axes[0].set_xlabel('t')
        axes[0].set_ylabel('y')
        axes[0].set_title("Exponential Decay")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # SHO phase space
        t, y = euler_method(f_sho, (0, 5), y0_sho, h=0.001)
        axes[1].plot(y[:, 0], y[:, 1], 'b-', alpha=0.7, label='Euler')
        theta = np.linspace(0, 2*np.pi, 200)
        axes[1].plot(np.cos(theta), -omega*np.sin(theta), 'k--', label='Exact')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('v')
        axes[1].set_title("Phase Space (SHO)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect('equal')

        plt.tight_layout()
        plt.savefig("euler_method.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
