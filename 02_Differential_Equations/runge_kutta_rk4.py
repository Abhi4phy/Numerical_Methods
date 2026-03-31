"""
Runge-Kutta Methods (RK4)
===========================
Fourth-order Runge-Kutta method — the workhorse of ODE solving.

    k1 = f(t_n, y_n)
    k2 = f(t_n + h/2, y_n + h/2 * k1)
    k3 = f(t_n + h/2, y_n + h/2 * k2)
    k4 = f(t_n + h, y_n + h * k3)
    y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)

Properties:
- Fourth-order accurate: local error O(h⁵), global error O(h⁴).
- 4 function evaluations per step.
- Excellent balance of accuracy and computational cost.
- Self-starting (unlike multi-step methods).

Physics applications:
- Orbital mechanics, molecular dynamics, circuit simulation.
- Any ODE system where moderate accuracy is needed.
"""

import numpy as np


def rk4(f, t_span, y0, h):
    """
    Classical 4th-order Runge-Kutta method.
    
    Parameters
    ----------
    f : callable
        f(t, y) returning dy/dt.
    t_span : tuple (t0, tf)
    y0 : float or ndarray – initial condition
    h : float – step size
    
    Returns
    -------
    t : ndarray – time points
    y : ndarray – solution
    """
    t0, tf = t_span
    t = np.arange(t0, tf + h/2, h)
    n_steps = len(t)

    y0 = np.atleast_1d(np.array(y0, dtype=float))
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0

    for i in range(n_steps - 1):
        k1 = np.atleast_1d(f(t[i], y[i]))
        k2 = np.atleast_1d(f(t[i] + h/2, y[i] + h/2 * k1))
        k3 = np.atleast_1d(f(t[i] + h/2, y[i] + h/2 * k2))
        k4 = np.atleast_1d(f(t[i] + h,   y[i] + h * k3))
        y[i + 1] = y[i] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return t, y.squeeze()


def rk2_midpoint(f, t_span, y0, h):
    """
    2nd-order Runge-Kutta (Midpoint method) for comparison.
    
    k1 = f(t_n, y_n)
    k2 = f(t_n + h/2, y_n + h/2 * k1)
    y_{n+1} = y_n + h * k2
    """
    t0, tf = t_span
    t = np.arange(t0, tf + h/2, h)
    n_steps = len(t)

    y0 = np.atleast_1d(np.array(y0, dtype=float))
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0

    for i in range(n_steps - 1):
        k1 = np.atleast_1d(f(t[i], y[i]))
        k2 = np.atleast_1d(f(t[i] + h/2, y[i] + h/2 * k1))
        y[i + 1] = y[i] + h * k2

    return t, y.squeeze()


# ============================================================
# Demo: Convergence Study & Orbital Mechanics
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RUNGE-KUTTA (RK4) DEMO")
    print("=" * 60)

    # --- Convergence study with known solution ---
    # TO TEST: Verify convergence rates for Euler (order 1), RK2 (order 2), RK4 (order 4).
    # Parameters: f(t,y)=-y (exponential decay), h=[0.5, 0.2, 0.1, 0.05, 0.01], time horizon [0,5].
    # Initial values: y(0)=1 (exact solution y(t)=exp(-t)).
    # Observe: Error ratios on subsequent halvings: Euler ~2x, RK2 ~4x, RK4 ~16x (2^order).
    # Try: Change f to -2y or use time horizon [0,10] to see convergence order more clearly.
    print("\n--- Convergence study: dy/dt = -y, y(0) = 1 ---")
    f = lambda t, y: -y
    y_exact = lambda t: np.exp(-t)

    print(f"{'h':>10s} | {'Euler':>12s} | {'RK2':>12s} | {'RK4':>12s}")
    print("-" * 55)

    from euler_method import euler_method

    for h in [0.5, 0.2, 0.1, 0.05, 0.01]:
        t1, y1 = euler_method(f, (0, 5), 1.0, h)
        t2, y2 = rk2_midpoint(f, (0, 5), 1.0, h)
        t4, y4 = rk4(f, (0, 5), 1.0, h)
        e1 = np.max(np.abs(y1 - y_exact(t1)))
        e2 = np.max(np.abs(y2 - y_exact(t2)))
        e4 = np.max(np.abs(y4 - y_exact(t4)))
        print(f"{h:10.3f} | {e1:12.6e} | {e2:12.6e} | {e4:12.6e}")

    # --- Kepler orbit (2-body problem) ---
    # TO TEST: Vary eccentricity e, number of orbits (5*T can be 10*T), or step size h.
    # Parameters: e (0.6=ellipse, 0=circle, 0.9=very eccentric), h (step size), integration time (5*T).
    # Initial values: y0_kepler from elliptical orbit formulas (perihelion + circular velocity).
    # Observe: Energy should stay nearly constant over 5 orbits. Watch for drift growing in time.
    # Try: increase h (0.02, 0.05) to see energy drift worsen; or h=0.001 to see improved accuracy.
    print("\n--- Kepler Orbit (e=0.6 elliptical orbit) ---")
    # Equations: d²r/dt² = -r/|r|³ (gravitational units)
    # State: y = [x, y, vx, vy]
    e = 0.6  # eccentricity

    def kepler(t, y):
        x, y_pos, vx, vy = y
        r3 = (x**2 + y_pos**2) ** 1.5
        return np.array([vx, vy, -x / r3, -y_pos / r3])

    # Initial conditions for elliptical orbit
    y0_kepler = [1 - e, 0, 0, np.sqrt((1 + e) / (1 - e))]
    T = 2 * np.pi  # Orbital period

    t, y = rk4(kepler, (0, 5 * T), y0_kepler, h=0.01)

    # Check energy conservation
    E = 0.5 * (y[:, 2]**2 + y[:, 3]**2) - 1.0 / np.sqrt(y[:, 0]**2 + y[:, 1]**2)
    print(f"Initial energy: {E[0]:.10f}")
    print(f"Final energy:   {E[-1]:.10f}")
    print(f"Energy drift:   {abs(E[-1] - E[0]):.2e}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Orbit
        axes[0].plot(y[:, 0], y[:, 1], 'b-', alpha=0.7)
        axes[0].plot(0, 0, 'yo', markersize=10, label='Central body')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_title(f'Kepler Orbit (e={e})')
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Energy
        axes[1].plot(t / T, (E - E[0]) / abs(E[0]), 'r-')
        axes[1].set_xlabel('Orbits')
        axes[1].set_ylabel('ΔE/E₀')
        axes[1].set_title('Energy Conservation (RK4)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("runge_kutta_rk4.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
