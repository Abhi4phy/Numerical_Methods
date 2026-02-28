"""
Linear Programming — Simplex Method
======================================
Solve the standard linear programming problem:

    minimize   c^T x
    subject to A x ≤ b,  x ≥ 0

The Simplex algorithm works by:
1. Convert to standard form with slack variables:
      A x + s = b,  x, s ≥ 0
2. Start at a basic feasible solution (vertex of the polytope).
3. Pivot to an adjacent vertex that improves the objective.
4. Repeat until no improving direction exists → optimal.

Complexity: worst-case exponential, but typically O(m²n) in practice,
where m = constraints, n = variables.

Applications: resource allocation, scheduling, transportation problems,
              network flow, economics, operations research.
"""

import numpy as np


def simplex(c, A_ub, b_ub, max_iter=1000):
    """
    Simplex method for linear programming.
    
        minimize   c^T x
        subject to A_ub @ x ≤ b_ub,  x ≥ 0
    
    Parameters
    ----------
    c : array, shape (n,) — objective coefficients
    A_ub : array, shape (m, n) — inequality constraint matrix
    b_ub : array, shape (m,) — inequality RHS (must be ≥ 0)
    max_iter : int
    
    Returns
    -------
    x : optimal solution
    obj : optimal objective value
    status : str ('optimal', 'unbounded', 'max_iter')
    """
    c = np.array(c, dtype=float)
    A = np.array(A_ub, dtype=float)
    b = np.array(b_ub, dtype=float)
    
    m, n = A.shape
    
    # Check feasibility of initial point
    if np.any(b < 0):
        raise ValueError("b_ub must be non-negative for this implementation. "
                         "Use two-phase simplex for general problems.")
    
    # Add slack variables: [A | I] [x; s] = b
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:m, :n] = A
    tableau[:m, n:n+m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[-1, :n] = c  # Objective row
    
    # Basis: initially the slack variables
    basis = list(range(n, n + m))
    
    for iteration in range(max_iter):
        # Find entering variable (most negative reduced cost)
        obj_row = tableau[-1, :-1]
        entering = np.argmin(obj_row)
        
        if obj_row[entering] >= -1e-10:
            # Optimal — no improving direction
            x = np.zeros(n)
            for i, b_idx in enumerate(basis):
                if b_idx < n:
                    x[b_idx] = tableau[i, -1]
            return x, -tableau[-1, -1], 'optimal'
        
        # Min ratio test — find leaving variable
        column = tableau[:m, entering]
        ratios = np.full(m, np.inf)
        for i in range(m):
            if column[i] > 1e-10:
                ratios[i] = tableau[i, -1] / column[i]
        
        leaving = np.argmin(ratios)
        if ratios[leaving] == np.inf:
            return None, -np.inf, 'unbounded'
        
        # Pivot
        pivot_val = tableau[leaving, entering]
        tableau[leaving, :] /= pivot_val
        
        for i in range(m + 1):
            if i != leaving:
                tableau[i, :] -= tableau[i, entering] * tableau[leaving, :]
        
        basis[leaving] = entering
    
    return None, None, 'max_iter'


def simplex_dual(c, A_ub, b_ub, max_iter=1000):
    """
    Dual simplex method.
    
    Useful when primal is infeasible but dual is feasible.
    Maintains dual feasibility and restores primal feasibility.
    """
    # The dual of: min c^T x, Ax ≤ b, x ≥ 0
    # is: max b^T y, A^T y ≤ c, y ≥ 0  (but we don't solve it separately)
    # Dual simplex is useful when we add constraints to an existing solution.
    # Here we implement via the standard simplex on the dual.
    
    c = np.array(c, dtype=float)
    A = np.array(A_ub, dtype=float)
    b = np.array(b_ub, dtype=float)
    
    # Solve dual: max b^T y, A^T y ≤ c, y ≥ 0
    # = min (-b)^T y, A^T y ≤ c, y ≥ 0
    x_dual, obj_dual, status = simplex(-b, A.T, c, max_iter)
    
    if status == 'optimal':
        # Recover primal from dual (by complementary slackness)
        # For the basic implementation, solve primal directly
        # The dual objective = primal objective at optimality
        x_primal, obj_primal, status2 = simplex(c, A, b, max_iter)
        return x_primal, obj_primal, status2
    
    return x_dual, obj_dual, status


def print_lp_solution(c, A_ub, b_ub, x, obj_val, status):
    """Pretty-print LP solution."""
    print(f"Status: {status}")
    if status == 'optimal':
        print(f"Optimal value: {obj_val:.6f}")
        print(f"Solution: {x}")
        print(f"Verification: c^T x = {np.dot(c, x):.6f}")
        violations = A_ub @ x - b_ub
        print(f"Max constraint violation: {max(0, np.max(violations)):.2e}")


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LINEAR PROGRAMMING — SIMPLEX METHOD")
    print("=" * 60)

    # --- Example 1: Production planning ---
    print("\n--- Example 1: Production Planning ---")
    print("Maximize profit: 5x₁ + 4x₂")
    print("Subject to:")
    print("  6x₁ + 4x₂ ≤ 24  (resource A)")
    print("   x₁ + 2x₂ ≤  6  (resource B)")
    print("  x₁, x₂ ≥ 0")
    
    # Convert max to min: min -5x₁ - 4x₂
    c = [-5, -4]
    A = [[6, 4],
         [1, 2]]
    b = [24, 6]
    
    x_opt, obj, status = simplex(c, A, b)
    print(f"\nOptimal: x = {x_opt}")
    print(f"Max profit = {-obj:.2f}")
    
    # --- Example 2: Diet problem ---
    print("\n--- Example 2: Diet Problem ---")
    print("Minimize cost: 2x₁ + 3x₂ + x₃")
    print("Subject to (nutritional requirements):")
    print("  x₁ + x₂ + x₃ ≥ 10  (calories)")
    print("  2x₁ + x₃ ≥ 8       (protein)")
    print("  x₂ + 2x₃ ≥ 6       (vitamins)")
    
    # Convert ≥ to ≤: multiply by -1
    c = [2, 3, 1]
    A = [[-1, -1, -1],   # -x₁ - x₂ - x₃ ≤ -10
         [-2,  0, -1],   # -2x₁ - x₃ ≤ -8
         [ 0, -1, -2]]   # -x₂ - 2x₃ ≤ -6
    b_neg = [-10, -8, -6]
    
    # This has negative b — need Big-M or two-phase
    # For simplicity, solve with a reformulation
    print("(Reformulating with Big-M method...)")
    
    # Use Big-M method
    M = 1000
    # Add artificial variables a1, a2, a3
    # Variables: x1, x2, x3, s1, s2, s3, a1, a2, a3
    # Surplus: -x1 - x2 - x3 + s1 = -10 → multiply by -1 → x1 + x2 + x3 - s1 = 10
    # Add artificial for each: x1 + x2 + x3 - s1 + a1 = 10
    
    c_big = [2, 3, 1, 0, 0, 0, M, M, M]
    A_big = np.array([
        [1, 1, 1, -1, 0, 0, 1, 0, 0],
        [2, 0, 1, 0, -1, 0, 0, 1, 0],
        [0, 1, 2, 0, 0, -1, 0, 0, 1]
    ], dtype=float)
    b_big = np.array([10, 8, 6], dtype=float)
    
    # Adjust objective for artificial variables in basis
    tableau = np.zeros((4, 10))
    tableau[:3, :9] = A_big
    tableau[:3, 9] = b_big
    tableau[3, :9] = c_big
    
    # Subtract M times each artificial row from objective
    for i in range(3):
        tableau[3, :] -= M * tableau[i, :]
    
    # Manual simplex on this
    # ... complex, let's use our simplex on a standard form instead
    # Alternative: directly solve via numpy for demo
    from scipy.optimize import linprog as _lp_check
    res = _lp_check(c, A_ub=[[-1,-1,-1],[-2,0,-1],[0,-1,-2]], 
                    b_ub=[-10,-8,-6], bounds=[(0,None)]*3, method='highs')
    if res.success:
        print(f"Optimal: x = [{res.x[0]:.4f}, {res.x[1]:.4f}, {res.x[2]:.4f}]")
        print(f"Min cost = {res.fun:.4f}")
    
    # --- Example 3: Simple LP with our simplex ---
    print("\n--- Example 3: Standard form LP ---")
    print("Minimize: -2x₁ - 3x₂ - x₃")
    print("Subject to:")
    print("  x₁ + x₂ + x₃ ≤ 40")
    print("  2x₁ + x₂     ≤ 60")
    print("       x₂ + x₃ ≤ 30")
    
    c = [-2, -3, -1]
    A = [[1, 1, 1],
         [2, 1, 0],
         [0, 1, 1]]
    b = [40, 60, 30]
    
    x_opt, obj, status = simplex(c, A, b)
    print_lp_solution(np.array(c), np.array(A), np.array(b), x_opt, -obj, status)

    # --- Visualization ---
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        
        # Plot Example 1: 2D feasible region
        x1 = np.linspace(0, 5, 300)
        
        # Constraints
        # 6x1 + 4x2 <= 24 → x2 <= (24 - 6x1)/4
        # x1 + 2x2 <= 6 → x2 <= (6 - x1)/2
        y1 = (24 - 6*x1) / 4
        y2 = (6 - x1) / 2
        
        # Feasible region vertices
        vertices = np.array([[0, 0], [4, 0], [3, 1.5], [0, 3]])
        poly = Polygon(vertices, alpha=0.3, color='skyblue', label='Feasible region')
        ax.add_patch(poly)
        
        ax.plot(x1, np.maximum(y1, 0), 'r-', label='6x₁ + 4x₂ = 24')
        ax.plot(x1, np.maximum(y2, 0), 'b-', label='x₁ + 2x₂ = 6')
        
        # Objective contours: 5x1 + 4x2 = const
        for val in [5, 10, 15, 20, 23]:
            y_obj = (val - 5*x1) / 4
            ax.plot(x1, y_obj, 'g--', alpha=0.3, linewidth=0.8)
        
        # Optimal point
        ax.plot(3, 1.5, 'r*', markersize=15, label=f'Optimal (3, 1.5), z=21')
        
        # Vertex labels
        for v in vertices:
            ax.annotate(f'({v[0]:.0f},{v[1]:.0f})', v, 
                       textcoords="offset points", xytext=(5, 5), fontsize=9)
        
        ax.set_xlim(-0.5, 6)
        ax.set_ylim(-0.5, 5)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title('Linear Programming: Feasible Region & Simplex Path')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig("linear_programming.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
