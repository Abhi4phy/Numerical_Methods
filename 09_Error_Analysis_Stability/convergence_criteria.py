"""
Convergence Criteria & Order of Accuracy
==========================================
Methods for measuring, verifying, and ensuring convergence of
numerical methods.

**Convergence:**
A numerical method converges if the solution approaches the true
solution as the discretization parameter (h, Δt, N) → 0.

**Order of accuracy p:**
    ||u_h - u_exact|| = C · h^p + O(h^{p+1})

Verified by grid refinement study: halve h, check if error decreases
by factor 2^p.

**Convergence rate estimation:**
    p ≈ log(e_h / e_{h/2}) / log(2)

**Iterative method convergence:**
- Linear convergence: ||e_{k+1}|| ≤ ρ ||e_k||  (0 < ρ < 1)
- Quadratic: ||e_{k+1}|| ≤ C ||e_k||²  (Newton's method)
- Superlinear: ||e_{k+1}||/||e_k|| → 0

**Stopping criteria for iterative methods:**
1. Absolute residual: ||r_k|| < ε_a
2. Relative residual: ||r_k||/||r_0|| < ε_r
3. Solution change: ||x_k - x_{k-1}||/||x_k|| < ε
4. Maximum iterations exceeded

Physics: ensuring simulation accuracy, selecting grid resolution,
validating numerical implementations.
"""

import numpy as np


def grid_refinement_study(solver, exact_solution, grid_sizes, norm='max'):
    """
    Perform a grid refinement study to estimate convergence order.
    
    Parameters
    ----------
    solver : callable(N) — returns (x, u_numerical) for grid size N
    exact_solution : callable(x) — true solution
    grid_sizes : list of N values
    norm : 'max' (L∞), 'L2', or 'L1'
    
    Returns
    -------
    errors : list of errors
    orders : list of estimated convergence orders
    hs : list of grid spacings
    """
    errors = []
    hs = []
    
    for N in grid_sizes:
        x, u_num = solver(N)
        u_exact = exact_solution(x)
        
        h = x[1] - x[0] if len(x) > 1 else 1.0/N
        hs.append(h)
        
        diff = u_num - u_exact
        
        if norm == 'max':
            errors.append(np.max(np.abs(diff)))
        elif norm == 'L2':
            errors.append(np.sqrt(h * np.sum(diff**2)))
        elif norm == 'L1':
            errors.append(h * np.sum(np.abs(diff)))
    
    # Estimate convergence order
    orders = []
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i-1] > 0:
            p = np.log(errors[i-1] / errors[i]) / np.log(hs[i-1] / hs[i])
            orders.append(p)
        else:
            orders.append(float('nan'))
    
    return errors, orders, hs


def convergence_rate_iterative(errors):
    """
    Estimate convergence rate and type of an iterative method.
    
    Returns
    -------
    rates : list of ||e_{k+1}||/||e_k||
    order : estimated convergence order (1=linear, 2=quadratic)
    """
    errors = np.array(errors)
    errors = errors[errors > 0]  # Remove zeros
    
    if len(errors) < 3:
        return [], None
    
    # Linear convergence rate
    rates = errors[1:] / errors[:-1]
    
    # Estimate order: log(e_{k+1}/e_k) / log(e_k/e_{k-1})
    orders = []
    for k in range(1, len(errors) - 1):
        if errors[k] > 0 and errors[k-1] > 0:
            # ||e_{k+1}|| ≈ C ||e_k||^p
            # log(e_{k+1}) ≈ log(C) + p log(e_k)
            log_ratio_1 = np.log(errors[k+1] / errors[k])
            log_ratio_0 = np.log(errors[k] / errors[k-1])
            if abs(log_ratio_0) > 1e-15:
                p = log_ratio_1 / log_ratio_0
                orders.append(p)
    
    avg_order = np.median(orders) if orders else None
    
    return rates.tolist(), avg_order


def aitken_delta_squared(sequence):
    """
    Aitken's Δ² method to accelerate a linearly convergent sequence.
    
    s'_n = s_n - (s_{n+1} - s_n)² / (s_{n+2} - 2s_{n+1} + s_n)
    
    Can also estimate the convergence rate.
    """
    s = np.array(sequence)
    n = len(s)
    if n < 3:
        return s
    
    accelerated = np.zeros(n - 2)
    for i in range(n - 2):
        denom = s[i+2] - 2*s[i+1] + s[i]
        if abs(denom) > 1e-30:
            accelerated[i] = s[i] - (s[i+1] - s[i])**2 / denom
        else:
            accelerated[i] = s[i]
    
    return accelerated


def verify_ode_solver_order(ode_solver, f, y0, t_span, exact_sol):
    """
    Verify convergence order of an ODE solver by grid refinement.
    
    Parameters
    ----------
    ode_solver : callable(f, y0, t_span, N) — returns (t, y)
    f : callable(t, y) — RHS
    y0 : initial condition
    t_span : (t0, tf)
    exact_sol : callable(t)
    
    Returns
    -------
    dt_values, errors, orders
    """
    Ns = [50, 100, 200, 400, 800, 1600]
    dt_values = []
    errors = []
    
    for N in Ns:
        t, y = ode_solver(f, y0, t_span, N)
        dt = (t_span[1] - t_span[0]) / N
        dt_values.append(dt)
        
        y_exact = exact_sol(t[-1])
        errors.append(abs(y[-1] - y_exact))
    
    orders = []
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i-1] > 0:
            p = np.log(errors[i-1]/errors[i]) / np.log(dt_values[i-1]/dt_values[i])
            orders.append(p)
        else:
            orders.append(float('nan'))
    
    return dt_values, errors, orders


# ============================================================
# Example ODE Solvers (for convergence testing)
# ============================================================
def euler_solve(f, y0, t_span, N):
    """Forward Euler — order 1."""
    t0, tf = t_span
    dt = (tf - t0) / N
    t = np.linspace(t0, tf, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    for i in range(N):
        y[i+1] = y[i] + dt * f(t[i], y[i])
    return t, y


def rk2_solve(f, y0, t_span, N):
    """RK2 (midpoint) — order 2."""
    t0, tf = t_span
    dt = (tf - t0) / N
    t = np.linspace(t0, tf, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    for i in range(N):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + 0.5*dt, y[i] + 0.5*dt*k1)
        y[i+1] = y[i] + dt * k2
    return t, y


def rk4_solve(f, y0, t_span, N):
    """Classical RK4 — order 4."""
    t0, tf = t_span
    dt = (tf - t0) / N
    t = np.linspace(t0, tf, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0
    for i in range(N):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + 0.5*dt, y[i] + 0.5*dt*k1)
        k3 = f(t[i] + 0.5*dt, y[i] + 0.5*dt*k2)
        k4 = f(t[i] + dt, y[i] + dt*k3)
        y[i+1] = y[i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, y


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CONVERGENCE CRITERIA & ORDER OF ACCURACY")
    print("=" * 60)

    # TO TEST: Change t_span, initial value y0, or the ODE RHS (f_ode) and verify observed order trends (Euler~1, RK2~2, RK4~4).
    # Observe how dt refinement changes terminal error and the estimated order column.
    # --- ODE Solver Convergence ---
    print("\n--- ODE Convergence: y' = -y, y(0) = 1, exact = e^{-t} ---")
    
    f_ode = lambda t, y: -y
    exact_ode = lambda t: np.exp(-t)
    t_span = (0, 1)
    
    for name, solver in [("Euler (O(h))", euler_solve),
                          ("RK2 (O(h²))", rk2_solve),
                          ("RK4 (O(h⁴))", rk4_solve)]:
        dt_vals, errs, orders = verify_ode_solver_order(
            solver, f_ode, 1.0, t_span, exact_ode
        )
        print(f"\n  {name}:")
        print(f"  {'dt':>12s} {'Error':>12s} {'Order':>8s}")
        for i, (dt, err) in enumerate(zip(dt_vals, errs)):
            order_str = f"{orders[i-1]:.2f}" if i > 0 else "  —"
            print(f"  {dt:12.6f} {err:12.2e} {order_str:>8s}")

    # TO TEST: Vary Ns, norm in grid_refinement_study (max/L2/L1), or source/BC setup in poisson_solver.
    # Observe whether halving h yields the expected ~4x error reduction for this 2nd-order stencil.
    # --- PDE Grid Refinement ---
    print("\n--- PDE Grid Refinement: -u'' = π²sin(πx) ---")
    
    def poisson_solver(N):
        """2nd-order FD Poisson solver."""
        h = 1.0 / N
        x = np.linspace(0, 1, N + 1)
        f = np.pi**2 * np.sin(np.pi * x)
        
        # Tridiagonal system
        A = np.zeros((N-1, N-1))
        for i in range(N-1):
            A[i, i] = 2.0
            if i > 0: A[i, i-1] = -1.0
            if i < N-2: A[i, i+1] = -1.0
        A /= h**2
        
        u_int = np.linalg.solve(A, f[1:N])
        u = np.zeros(N + 1)
        u[1:N] = u_int
        return x, u
    
    exact_poisson = lambda x: np.sin(np.pi * x)
    Ns = [10, 20, 40, 80, 160, 320]
    
    errors, orders, hs = grid_refinement_study(poisson_solver, exact_poisson, Ns)
    
    print(f"\n  {'N':>6s} {'h':>12s} {'Error':>12s} {'Order':>8s}")
    for i, (N, h, err) in enumerate(zip(Ns, hs, errors)):
        order_str = f"{orders[i-1]:.2f}" if i > 0 else "  —"
        print(f"  {N:6d} {h:12.6f} {err:12.2e} {order_str:>8s}")

    # --- Iterative convergence: Newton vs bisection ---
    print("\n--- Iterative Convergence: f(x) = x² - 2, root = √2 ---")
    
    # Newton's method
    x_newton = 2.0
    errors_newton = []
    for k in range(10):
        errors_newton.append(abs(x_newton - np.sqrt(2)))
        x_newton = x_newton - (x_newton**2 - 2) / (2*x_newton)
    
    rates_n, order_n = convergence_rate_iterative(errors_newton)
    print(f"\n  Newton's method (quadratic convergence):")
    print(f"  Estimated order: {order_n:.2f}")
    for k, err in enumerate(errors_newton):
        print(f"    k={k}: error = {err:.2e}")
    
    # Bisection
    a, b = 1.0, 2.0
    errors_bisect = []
    for k in range(50):
        mid = (a + b) / 2
        errors_bisect.append(abs(mid - np.sqrt(2)))
        if mid**2 < 2:
            a = mid
        else:
            b = mid
    
    rates_b, order_b = convergence_rate_iterative(errors_bisect)
    print(f"\n  Bisection (linear convergence):")
    print(f"  Estimated order: {order_b:.2f}")
    print(f"  Rate ρ ≈ {np.mean(rates_b):.4f} (expect 0.5)")

    # --- Aitken acceleration ---
    print("\n--- Aitken Δ² Acceleration ---")
    # Linearly convergent sequence: x_{n+1} = cos(x_n), converges to 0.7391
    seq = [0.5]
    for k in range(30):
        seq.append(np.cos(seq[-1]))
    
    accel = aitken_delta_squared(seq)
    true_val = 0.7390851332  # Fixed point of cos
    
    print(f"  Original sequence error after 30 iters: {abs(seq[-1] - true_val):.2e}")
    print(f"  Aitken accelerated error:               {abs(accel[-1] - true_val):.2e}")
    print(f"  Speedup: {abs(seq[-1]-true_val)/abs(accel[-1]-true_val):.0f}x")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ODE convergence
        for name, solver, marker, color in [
            ("Euler", euler_solve, 'o', 'blue'),
            ("RK2", rk2_solve, 's', 'green'),
            ("RK4", rk4_solve, '^', 'red')
        ]:
            dt_vals, errs, _ = verify_ode_solver_order(
                solver, f_ode, 1.0, t_span, exact_ode
            )
            axes[0, 0].loglog(dt_vals, errs, f'{marker}-', color=color,
                            label=name, markersize=6)
        
        # Reference slopes
        dt_ref = np.array(dt_vals)
        axes[0, 0].loglog(dt_ref, dt_ref, 'k--', alpha=0.3, label='O(h)')
        axes[0, 0].loglog(dt_ref, dt_ref**2, 'k-.', alpha=0.3, label='O(h²)')
        axes[0, 0].loglog(dt_ref, dt_ref**4, 'k:', alpha=0.3, label='O(h⁴)')
        axes[0, 0].set_xlabel('Δt')
        axes[0, 0].set_ylabel('Error at t=1')
        axes[0, 0].set_title('ODE Solver Convergence')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        # PDE convergence
        axes[0, 1].loglog(hs, errors, 'bo-', markersize=6)
        axes[0, 1].loglog(hs, [h**2 for h in hs], 'r--', alpha=0.5, label='O(h²)')
        axes[0, 1].set_xlabel('Grid spacing h')
        axes[0, 1].set_ylabel('Max error')
        axes[0, 1].set_title('FD Poisson: Grid Refinement')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Newton vs Bisection convergence
        axes[1, 0].semilogy(range(len(errors_newton)), errors_newton, 'r-o',
                           label='Newton (quadratic)', markersize=5)
        axes[1, 0].semilogy(range(len(errors_bisect)), errors_bisect, 'b-',
                           label='Bisection (linear)', alpha=0.7)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('|x_k - √2|')
        axes[1, 0].set_title('Iterative Convergence Rates')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Aitken acceleration
        axes[1, 1].semilogy(range(len(seq)), [abs(s - true_val) for s in seq],
                           'b-', label='Original')
        axes[1, 1].semilogy(range(len(accel)), [abs(a - true_val) for a in accel],
                           'r-', label='Aitken Δ²')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('|x_k - x*|')
        axes[1, 1].set_title('Aitken Δ² Acceleration')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("convergence_criteria.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
