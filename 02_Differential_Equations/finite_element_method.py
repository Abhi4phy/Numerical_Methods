"""
Finite Element Method (FEM) — 1D
=================================
Solves boundary value problems using a weak (variational) formulation.

Core idea:
1. Multiply the PDE by a test function and integrate (weak form).
2. Approximate the solution as a sum of basis functions (hat functions).
3. Assemble a global stiffness matrix from element contributions.
4. Solve the resulting linear system.

Advantages over FDM:
- Handles irregular geometries naturally.
- Mathematically rigorous error bounds.
- Easily handles variable coefficients and mixed BCs.

Demo problem: -d/dx[p(x) du/dx] + q(x)u = f(x) on [a,b]
with Dirichlet BCs (can extend to Neumann).
"""

import numpy as np


def fem_1d(p_func, q_func, f_func, x_range, n_elements,
           bc_left=0.0, bc_right=0.0):
    """
    1D Finite Element Method using linear (hat) basis functions.
    
    Solves: -d/dx[p(x) du/dx] + q(x) u = f(x)
    with u(a) = bc_left, u(b) = bc_right.
    
    Parameters
    ----------
    p_func : callable – coefficient p(x) (e.g., thermal conductivity)
    q_func : callable – coefficient q(x) (e.g., reaction term)
    f_func : callable – source term f(x)
    x_range : tuple (a, b)
    n_elements : int – number of elements
    
    Returns
    -------
    x : ndarray – node positions
    u : ndarray – solution at nodes
    """
    a, b = x_range
    n_nodes = n_elements + 1
    x = np.linspace(a, b, n_nodes)
    h = np.diff(x)  # Element sizes

    # Global stiffness matrix K and load vector F
    K = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)

    # Gauss quadrature points for integration (2-point, exact for cubics)
    xi_gq = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])  # Reference element [-1, 1]
    w_gq = np.array([1.0, 1.0])

    for e in range(n_elements):
        # Element nodes
        x1, x2 = x[e], x[e + 1]
        he = h[e]

        # Local stiffness and load
        K_local = np.zeros((2, 2))
        F_local = np.zeros(2)

        for gp in range(2):
            # Map from reference [-1,1] to physical [x1, x2]
            xi = xi_gq[gp]
            x_phys = 0.5 * (x1 + x2) + 0.5 * he * xi

            # Shape functions on reference element
            N1 = 0.5 * (1 - xi)
            N2 = 0.5 * (1 + xi)
            # Derivatives (dN/dx = dN/dξ * dξ/dx = dN/dξ * 2/h)
            dN1 = -1.0 / he
            dN2 = 1.0 / he

            # Evaluate coefficients at quadrature point
            p_val = p_func(x_phys)
            q_val = q_func(x_phys)
            f_val = f_func(x_phys)

            # Jacobian (dx/dξ = h/2)
            jac = he / 2.0

            # Element stiffness: ∫ [p dNi/dx dNj/dx + q Ni Nj] dx
            N = np.array([N1, N2])
            dN = np.array([dN1, dN2])

            K_local += (p_val * np.outer(dN, dN) + q_val * np.outer(N, N)) * jac * w_gq[gp]
            F_local += f_val * N * jac * w_gq[gp]

        # Assemble into global system
        dofs = [e, e + 1]
        for i_local in range(2):
            F[dofs[i_local]] += F_local[i_local]
            for j_local in range(2):
                K[dofs[i_local], dofs[j_local]] += K_local[i_local, j_local]

    # Apply Dirichlet BCs
    # Method: Set equation rows for boundary nodes to identity
    K[0, :] = 0;    K[0, 0] = 1;     F[0] = bc_left
    K[-1, :] = 0;   K[-1, -1] = 1;   F[-1] = bc_right

    # Solve
    u = np.linalg.solve(K, F)

    return x, u


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FINITE ELEMENT METHOD (1D) DEMO")
    print("=" * 60)

    # --- Problem 1: Simple Poisson ---
    # -u'' = π²sin(πx), u(0)=0, u(1)=0
    # Exact: u(x) = sin(πx)
    print("\n--- Problem 1: -u'' = π²sin(πx) ---")
    p = lambda x: 1.0
    q = lambda x: 0.0
    f = lambda x: np.pi**2 * np.sin(np.pi * x)
    u_exact = lambda x: np.sin(np.pi * x)

    for n_el in [5, 10, 20, 50]:
        x, u = fem_1d(p, q, f, (0, 1), n_el)
        error = np.max(np.abs(u - u_exact(x)))
        print(f"  n_elements = {n_el:3d}  |  Max error = {error:.6e}")

    # --- Problem 2: Variable coefficient ---
    # -d/dx[(1+x) du/dx] = 1, u(0)=0, u(1)=0
    print("\n--- Problem 2: -d/dx[(1+x) du/dx] = 1 ---")
    p2 = lambda x: 1.0 + x
    q2 = lambda x: 0.0
    f2 = lambda x: 1.0
    # Exact: u = -x + ln(1+x)/ln(2)  ...approximately
    x2, u2 = fem_1d(p2, q2, f2, (0, 1), 50)
    print(f"  Solution at midpoint: u(0.5) = {u2[25]:.6f}")

    # --- Problem 3: Reaction-diffusion ---
    # -u'' + u = x, u(0)=0, u(1)=0
    print("\n--- Problem 3: -u'' + u = x (reaction-diffusion) ---")
    p3 = lambda x: 1.0
    q3 = lambda x: 1.0
    f3 = lambda x: x
    x3, u3 = fem_1d(p3, q3, f3, (0, 1), 50)
    print(f"  Solution at midpoint: u(0.5) = {u3[25]:.6f}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Problem 1
        x, u = fem_1d(p, q, f, (0, 1), 10)
        axes[0].plot(x, u, 'bo-', markersize=6, label='FEM (10 elements)')
        x_fine = np.linspace(0, 1, 200)
        axes[0].plot(x_fine, u_exact(x_fine), 'r-', label='Exact')
        axes[0].set_title("Poisson: -u'' = π²sin(πx)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Problem 2
        axes[1].plot(x2, u2, 'b-o', markersize=3)
        axes[1].set_title("-d/dx[(1+x) du/dx] = 1")
        axes[1].set_xlabel('x')
        axes[1].grid(True, alpha=0.3)

        # Convergence study
        errors = []
        ns = [5, 10, 20, 50, 100, 200]
        for n_el in ns:
            x, u = fem_1d(p, q, f, (0, 1), n_el)
            errors.append(np.max(np.abs(u - u_exact(x))))
        axes[2].loglog(ns, errors, 'bo-', label='FEM error')
        axes[2].loglog(ns, [errors[0] * (ns[0]/n)**2 for n in ns], 'r--', label='O(h²)')
        axes[2].set_xlabel('Number of elements')
        axes[2].set_ylabel('Max error')
        axes[2].set_title('Convergence Rate')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("finite_element_method.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
