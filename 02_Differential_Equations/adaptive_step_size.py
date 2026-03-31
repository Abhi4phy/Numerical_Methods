"""
Adaptive Step-Size Methods (RK45 / Dormand-Prince)
====================================================
Automatically adjusts the step size to maintain a desired accuracy.

Idea: Compute two estimates of different orders (e.g., 4th and 5th).
The difference gives an error estimate. If error is too large, shrink h;
if error is small, grow h.

Embedded Runge-Kutta methods (like RK45) compute both orders using the
SAME function evaluations — very efficient.

Key properties:
- Concentrates computational effort where the solution changes rapidly.
- Avoids wasting effort in smooth regions.
- "FSAL" (First Same As Last) property saves function evaluations.

This is what scipy.integrate.solve_ivp uses by default.
"""

import numpy as np


def rkf45(f, t_span, y0, tol=1e-6, h_init=0.1, h_min=1e-12, h_max=1.0):
    """
    Runge-Kutta-Fehlberg 4(5) with adaptive step-size control.
    
    Uses a 4th-order solution for stepping and a 5th-order solution
    for error estimation.
    
    Parameters
    ----------
    f : callable – f(t, y)
    t_span : tuple (t0, tf)
    y0 : float or ndarray
    tol : float – desired local truncation error tolerance
    h_init : float – initial step size
    h_min, h_max : float – step size bounds
    
    Returns
    -------
    t : ndarray – adaptive time points (non-uniform spacing)
    y : ndarray – solution values
    """
    # RKF45 Butcher tableau coefficients
    a2, a3, a4, a5, a6 = 1/4, 3/8, 12/13, 1, 1/2

    b21 = 1/4
    b31, b32 = 3/32, 9/32
    b41, b42, b43 = 1932/2197, -7200/2197, 7296/2197
    b51, b52, b53, b54 = 439/216, -8, 3680/513, -845/4104
    b61, b62, b63, b64, b65 = -8/27, 2, -3544/2565, 1859/4104, -11/40

    # 4th-order weights
    c1, c3, c4, c5 = 25/216, 1408/2565, 2197/4104, -1/5
    # 5th-order weights
    d1, d3, d4, d5, d6 = 16/135, 6656/12825, 28561/56430, -9/50, 2/55

    t0, tf = t_span
    y0 = np.atleast_1d(np.array(y0, dtype=float))

    t_list = [t0]
    y_list = [y0.copy()]
    h = h_init
    t = t0
    y = y0.copy()

    n_accept = 0
    n_reject = 0

    while t < tf:
        if t + h > tf:
            h = tf - t

        # Six stages
        k1 = h * np.atleast_1d(f(t, y))
        k2 = h * np.atleast_1d(f(t + a2*h, y + b21*k1))
        k3 = h * np.atleast_1d(f(t + a3*h, y + b31*k1 + b32*k2))
        k4 = h * np.atleast_1d(f(t + a4*h, y + b41*k1 + b42*k2 + b43*k3))
        k5 = h * np.atleast_1d(f(t + a5*h, y + b51*k1 + b52*k2 + b53*k3 + b54*k4))
        k6 = h * np.atleast_1d(f(t + a6*h, y + b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))

        # 4th-order solution
        y4 = y + c1*k1 + c3*k3 + c4*k4 + c5*k5
        # 5th-order solution
        y5 = y + d1*k1 + d3*k3 + d4*k4 + d5*k5 + d6*k6

        # Error estimate
        error = np.linalg.norm(y5 - y4)
        error = max(error, 1e-20)  # Avoid division by zero

        # Step size adjustment (safety factor = 0.84)
        if error <= tol:
            # Accept step
            t = t + h
            y = y4  # Use 4th-order solution (local extrapolation uses y5)
            t_list.append(t)
            y_list.append(y.copy())
            n_accept += 1

        else:
            n_reject += 1

        # Optimal step size
        h_new = 0.84 * h * (tol / error) ** 0.25
        h = np.clip(h_new, h_min, h_max)

    t_arr = np.array(t_list)
    y_arr = np.array(y_list).squeeze()

    print(f"RKF45: {n_accept} accepted, {n_reject} rejected steps, "
          f"{len(t_list)} total points")
    return t_arr, y_arr


# ============================================================
# Demo: Stiff problem & comparison with fixed-step
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ADAPTIVE STEP-SIZE (RKF45) DEMO")
    print("=" * 60)

    # --- Example 1: Smooth problem with varying timescales ---
    # TO TEST: Modify stiffness parameter mu (1, 3, 10) and tolerance tol (1e-6 to 1e-10).
    # Parameters: mu (=relaxation parameter), y0=[2,0] (initial state), tol (error tolerance), time span [0,30].
    # Initial values: mu=3, tol=1e-8 (moderate stiffness and precision).
    # Observe: Number of adaptive points vs fixed-step points. Adaptive should be 10-50% of fixed!
    # Try: Increase mu to 10 for more regions with small time scales (RKF45 will use smaller steps).
    print("\n--- van der Pol oscillator (μ=3) ---")
    # x'' - μ(1-x²)x' + x = 0
    mu = 3.0

    def van_der_pol(t, y):
        return np.array([y[1], mu * (1 - y[0]**2) * y[1] - y[0]])

    y0 = [2.0, 0.0]
    t_adapt, y_adapt = rkf45(van_der_pol, (0, 30), y0, tol=1e-8)

    # Fixed-step RK4 for comparison
    from runge_kutta_rk4 import rk4
    t_fixed, y_fixed = rk4(van_der_pol, (0, 30), y0, h=0.01)

    print(f"\nAdaptive: {len(t_adapt)} points")
    print(f"Fixed h=0.01: {len(t_fixed)} points")
    print(f"Adaptive uses {len(t_adapt)/len(t_fixed)*100:.1f}% of fixed-step points")

    # --- Example 2: Exponential decay (exact solution known) ---
    # TO TEST: Try different decay rates (-5y, -20y, -100y) and tolerances (1e-8, 1e-12).
    # Parameters: decay rate (coefficient of y), y0=1.0 (initial), tol=1e-10, time horizon [0,2].
    # Initial values: dy/dt=-10y gives smooth solution (RKF45 takes large steps). Try -100y for stiff problem.
    # Observe: Max error vs exact solution exp(-10*t). Should be < tol. Compare with RK4 fixed-step.
    # Note: Very stiff problems (coefficients >>10) may need specially-designed solvers (DOPRI8, Rosenbrock).
    print("\n--- Exponential decay: dy/dt = -10y, y(0) = 1 ---")
    f_exp = lambda t, y: -10 * y
    t_a, y_a = rkf45(f_exp, (0, 2), 1.0, tol=1e-10)
    y_exact = np.exp(-10 * t_a)
    print(f"Max error: {np.max(np.abs(y_a - y_exact)):.2e}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Van der Pol solution
        axes[0, 0].plot(t_adapt, y_adapt[:, 0], 'b-', linewidth=1)
        axes[0, 0].set_xlabel('t')
        axes[0, 0].set_ylabel('x')
        axes[0, 0].set_title('Van der Pol Oscillator (Adaptive)')
        axes[0, 0].grid(True, alpha=0.3)

        # Step sizes
        dt = np.diff(t_adapt)
        axes[0, 1].semilogy(t_adapt[:-1], dt, 'r-', linewidth=1)
        axes[0, 1].set_xlabel('t')
        axes[0, 1].set_ylabel('Step size h')
        axes[0, 1].set_title('Adaptive Step Sizes')
        axes[0, 1].grid(True, alpha=0.3)

        # Phase portrait
        axes[1, 0].plot(y_adapt[:, 0], y_adapt[:, 1], 'b-', alpha=0.7)
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('dx/dt')
        axes[1, 0].set_title('Phase Portrait')
        axes[1, 0].grid(True, alpha=0.3)

        # Point distribution
        axes[1, 1].plot(t_adapt, np.zeros_like(t_adapt), '|', markersize=10, alpha=0.5)
        axes[1, 1].set_xlabel('t')
        axes[1, 1].set_title('Distribution of Adaptive Points')
        axes[1, 1].set_yticks([])
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("adaptive_step_size.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
