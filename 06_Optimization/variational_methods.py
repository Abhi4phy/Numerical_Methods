"""
Variational Methods
====================
Find functions y(x) that extremize (minimize/maximize) functionals:

    J[y] = ∫ₐᵇ F(x, y, y') dx

The necessary condition is the **Euler-Lagrange equation**:

    ∂F/∂y - d/dx (∂F/∂y') = 0

This is a 2nd-order ODE for y(x) with boundary conditions y(a)=A, y(b)=B.

Numerical approach:
1. Discretize y(x) on a grid x_0, x_1, ..., x_N.
2. Approximate the functional J as a discrete sum.
3. Minimize J w.r.t. the interior values y_1, ..., y_{N-1}.

Classic problems:
- **Brachistochrone**: curve of fastest descent under gravity.
- **Geodesics**: shortest path on a surface.
- **Catenary**: shape of a hanging chain.
- **Minimal surface**: soap film shape.

Physics: Lagrangian mechanics, quantum mechanics (variational principle),
         general relativity (geodesics), field theory.
"""

import numpy as np


def euler_lagrange_solve(F, dF_dy, dF_dyp, x_span, y_bc, N=100):
    """
    Solve the Euler-Lagrange equation numerically using finite differences.
    
    ∂F/∂y - d/dx(∂F/∂y') = 0
    
    This is discretized as a nonlinear system and solved with Newton's method.
    
    Parameters
    ----------
    F : callable(x, y, yp) — integrand
    dF_dy : callable(x, y, yp) — ∂F/∂y
    dF_dyp : callable(x, y, yp) — ∂F/∂y'
    x_span : (a, b) — domain
    y_bc : (ya, yb) — boundary conditions
    N : int — number of grid points
    
    Returns
    -------
    x : grid points
    y : solution
    """
    a, b = x_span
    ya, yb = y_bc
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    
    # Initial guess: linear interpolation
    y = ya + (yb - ya) * (x - a) / (b - a)
    
    # Newton iteration on the discretized E-L equation
    for newton_iter in range(100):
        # Compute residuals for interior points
        residual = np.zeros(N - 1)
        
        for i in range(1, N):
            yp_plus = (y[i+1] - y[i]) / h
            yp_minus = (y[i] - y[i-1]) / h
            
            # d/dx(∂F/∂y') ≈ [∂F/∂y'(x_{i+1/2}) - ∂F/∂y'(x_{i-1/2})] / h
            x_mid_r = 0.5 * (x[i] + x[i+1])
            y_mid_r = 0.5 * (y[i] + y[i+1])
            x_mid_l = 0.5 * (x[i-1] + x[i])
            y_mid_l = 0.5 * (y[i-1] + y[i])
            
            dFdyp_r = dF_dyp(x_mid_r, y_mid_r, yp_plus)
            dFdyp_l = dF_dyp(x_mid_l, y_mid_l, yp_minus)
            
            residual[i-1] = dF_dy(x[i], y[i], yp_minus) - (dFdyp_r - dFdyp_l) / h
        
        if np.linalg.norm(residual) < 1e-10:
            print(f"E-L solver converged in {newton_iter} Newton iterations")
            break
        
        # Jacobian (tridiagonal)
        J = np.zeros((N-1, N-1))
        eps = 1e-7
        for j in range(N-1):
            y_pert = y.copy()
            y_pert[j+1] += eps
            
            res_pert = np.zeros(N-1)
            for i in range(1, N):
                yp_plus = (y_pert[i+1] - y_pert[i]) / h
                yp_minus = (y_pert[i] - y_pert[i-1]) / h
                x_mid_r = 0.5 * (x[i] + x[i+1])
                y_mid_r = 0.5 * (y_pert[i] + y_pert[i+1])
                x_mid_l = 0.5 * (x[i-1] + x[i])
                y_mid_l = 0.5 * (y_pert[i-1] + y_pert[i])
                dFdyp_r = dF_dyp(x_mid_r, y_mid_r, yp_plus)
                dFdyp_l = dF_dyp(x_mid_l, y_mid_l, yp_minus)
                res_pert[i-1] = dF_dy(x[i], y_pert[i], yp_minus) - (dFdyp_r - dFdyp_l) / h
            
            J[:, j] = (res_pert - residual) / eps
        
        # Newton step
        dy = np.linalg.solve(J, -residual)
        y[1:N] += dy
    
    return x, y


def functional_value(F, x, y):
    """
    Compute J[y] = ∫ F(x, y, y') dx using trapezoidal quadrature.
    """
    N = len(x) - 1
    J = 0.0
    for i in range(N):
        h = x[i+1] - x[i]
        yp = (y[i+1] - y[i]) / h
        x_mid = 0.5 * (x[i] + x[i+1])
        y_mid = 0.5 * (y[i] + y[i+1])
        J += F(x_mid, y_mid, yp) * h
    return J


def direct_minimization(F, x_span, y_bc, N=50, max_iter=5000, alpha=0.01):
    """
    Directly minimize the functional J[y] = ∫F dx by gradient descent
    on the discrete values y_1, ..., y_{N-1}.
    
    Gradient w.r.t. y_i is computed numerically.
    """
    a, b = x_span
    ya, yb = y_bc
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    
    # Initial guess
    y = ya + (yb - ya) * (x - a) / (b - a)
    
    def compute_J(y_int):
        y_full = np.concatenate([[ya], y_int, [yb]])
        return functional_value(F, x, y_full)
    
    y_int = y[1:N].copy()
    
    for k in range(max_iter):
        J_val = compute_J(y_int)
        
        # Numerical gradient
        grad = np.zeros(N - 1)
        eps = 1e-7
        for i in range(N - 1):
            y_pert = y_int.copy()
            y_pert[i] += eps
            grad[i] = (compute_J(y_pert) - J_val) / eps
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        y_int -= alpha * grad
    
    y_full = np.concatenate([[ya], y_int, [yb]])
    J_final = compute_J(y_int)
    return x, y_full, J_final


def brachistochrone_analytical(x1, y1, N=200):
    """
    Analytical brachistochrone curve (parametric cycloid).
    
    The curve that minimizes travel time under gravity from
    (0,0) to (x1, y1) with y positive downward.
    
    Parametric: x(θ) = R(θ - sin θ), y(θ) = R(1 - cos θ)
    """
    from scipy.optimize import brentq
    
    def endpoint_eq(theta_f):
        R = y1 / (1 - np.cos(theta_f))
        return R * (theta_f - np.sin(theta_f)) - x1
    
    # Find theta_f
    theta_f = brentq(endpoint_eq, 0.01, 2*np.pi - 0.01)
    R = y1 / (1 - np.cos(theta_f))
    
    theta = np.linspace(0, theta_f, N)
    x = R * (theta - np.sin(theta))
    y = R * (1 - np.cos(theta))
    
    return x, y, R, theta_f


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("VARIATIONAL METHODS DEMO")
    print("=" * 60)

    # --- Problem 1: Shortest path (geodesic in flat space) ---
    # TO TEST: Change boundary conditions from (0,1)->(0,2) or grid size N=50->100, and observe max|y-x| behavior and arc-length error.
    print("\n--- Problem 1: Shortest path from (0,0) to (1,1) ---")
    print("F = √(1 + y'²),  Euler-Lagrange → y'' = 0 → straight line")
    
    F_length = lambda x, y, yp: np.sqrt(1 + yp**2)
    dF_dy = lambda x, y, yp: 0.0
    dF_dyp = lambda x, y, yp: yp / np.sqrt(1 + yp**2)
    
    x_geo, y_geo = euler_lagrange_solve(F_length, dF_dy, dF_dyp,
                                         (0, 1), (0, 1), N=50)
    
    J_line = functional_value(F_length, x_geo, y_geo)
    J_exact = np.sqrt(2)  # Length of straight line from (0,0) to (1,1)
    print(f"Computed arc length: {J_line:.8f}")
    print(f"Exact (√2):         {J_exact:.8f}")
    print(f"Error:              {abs(J_line - J_exact):.2e}")
    print(f"Solution is linear: max|y - x| = {np.max(np.abs(y_geo - x_geo)):.2e}")

    # --- Problem 2: Catenary ---
    # TO TEST: Modify domain/boundaries (for example x in [-2,2] with y(-2)=y(2)=cosh(2)) and observe changes in max error versus cosh(x).
    print("\n--- Problem 2: Catenary (hanging chain) ---")
    print("Minimize potential energy: J[y] = ∫ y √(1+y'²) dx")
    print("y(−1)=cosh(1), y(1)=cosh(1)")
    
    ya = np.cosh(1)
    F_cat = lambda x, y, yp: y * np.sqrt(1 + yp**2)
    dF_dy_cat = lambda x, y, yp: np.sqrt(1 + yp**2)
    dF_dyp_cat = lambda x, y, yp: y * yp / np.sqrt(1 + yp**2)
    
    x_cat, y_cat = euler_lagrange_solve(F_cat, dF_dy_cat, dF_dyp_cat,
                                         (-1, 1), (ya, ya), N=80)
    
    y_exact = np.cosh(x_cat)
    max_err = np.max(np.abs(y_cat - y_exact))
    print(f"Max error vs cosh(x): {max_err:.6e}")

    # --- Problem 3: Brachistochrone ---
    # TO TEST: Change endpoint (x1,y1), alpha, or max_iter in direct_minimization, and observe travel-time J_b plus agreement with cycloid comparison.
    print("\n--- Problem 3: Brachistochrone (fastest descent) ---")
    print("Minimize travel time: J[y] = ∫ √((1+y'²)/(2gy)) dx")
    print("From (0,0) to (1, 0.5), y downward positive")
    
    g = 9.81
    x1, y1 = 1.0, 0.5
    
    # Use direct minimization (E-L is tricky here due to singularity at y=0)
    F_brach = lambda x, y, yp: np.sqrt((1 + yp**2) / (2 * g * max(y, 1e-12)))
    
    x_b, y_b, J_b = direct_minimization(F_brach, (0, x1), (0.001, y1), N=40,
                                          max_iter=2000, alpha=0.0001)
    print(f"Computed travel time: {J_b:.6f} s")
    
    # Analytical comparison
    try:
        x_an, y_an, R, theta_f = brachistochrone_analytical(x1, y1)
        J_an = functional_value(F_brach, x_an, y_an)
        print(f"Analytical time:     {J_an:.6f} s")
        print(f"Cycloid R = {R:.4f}, θ_f = {theta_f:.4f}")
    except Exception as e:
        print(f"(Analytical comparison skipped: {e})")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Shortest path
        axes[0, 0].plot(x_geo, y_geo, 'b-o', markersize=2, label='Computed')
        axes[0, 0].plot([0, 1], [0, 1], 'r--', label='y = x (exact)')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].set_title('Shortest Path (Geodesic)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Catenary
        axes[0, 1].plot(x_cat, y_cat, 'b-', linewidth=2, label='Computed')
        axes[0, 1].plot(x_cat, y_exact, 'r--', linewidth=1.5, label='cosh(x) exact')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        axes[0, 1].set_title(f'Catenary (max error = {max_err:.2e})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Brachistochrone
        axes[1, 0].plot(x_b, y_b, 'b-', linewidth=2, label='Numerical')
        try:
            axes[1, 0].plot(x_an, y_an, 'r--', linewidth=1.5, label='Cycloid (analytical)')
        except NameError:
            pass
        # Also plot straight line comparison
        axes[1, 0].plot([0, x1], [0, y1], 'g:', label='Straight line')
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y (downward)')
        axes[1, 0].set_title('Brachistochrone')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Catenary error
        axes[1, 1].semilogy(x_cat, np.abs(y_cat - y_exact) + 1e-16, 'b-')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('|y_num - cosh(x)|')
        axes[1, 1].set_title('Catenary: Pointwise Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("variational_methods.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
