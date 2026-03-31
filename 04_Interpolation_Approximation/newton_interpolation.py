"""
Newton's Divided Difference Interpolation
==========================================
Another form of polynomial interpolation, using divided differences.

Formula:
    P(x) = f[x₀] + f[x₀,x₁](x-x₀) + f[x₀,x₁,x₂](x-x₀)(x-x₁) + ...

Divided differences:
    f[xᵢ] = f(xᵢ)
    f[xᵢ, xᵢ₊₁] = (f[xᵢ₊₁] - f[xᵢ]) / (xᵢ₊₁ - xᵢ)
    f[xᵢ, ..., xⱼ] = (f[xᵢ₊₁, ..., xⱼ] - f[xᵢ, ..., xⱼ₋₁]) / (xⱼ - xᵢ)

Advantages over Lagrange form:
- Adding a new point only requires computing ONE new divided difference.
- Error estimation is natural: the next term gives the error.
- More efficient for incremental interpolation.
"""

import numpy as np


def divided_differences(x_nodes, y_nodes):
    """
    Compute the divided difference table.
    
    Returns the top row of the table (coefficients of Newton form).
    
    Parameters
    ----------
    x_nodes : ndarray – interpolation nodes
    y_nodes : ndarray – function values
    
    Returns
    -------
    coeffs : ndarray – divided difference coefficients [f[x₀], f[x₀,x₁], ...]
    table : ndarray – full divided difference table (for inspection)
    """
    n = len(x_nodes)
    # Initialize table with y values
    table = np.zeros((n, n))
    table[:, 0] = y_nodes
    
    # Fill in divided differences column by column
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i+1, j-1] - table[i, j-1]) / (x_nodes[i+j] - x_nodes[i])
    
    # Coefficients are the first row
    coeffs = table[0, :]
    return coeffs, table


def newton_interpolation(x_nodes, coeffs, x):
    """
    Evaluate Newton's interpolating polynomial at x using Horner's method.
    
    P(x) = c₀ + c₁(x-x₀) + c₂(x-x₀)(x-x₁) + ...
    
    Horner form (efficient, O(n)):
    P(x) = c₀ + (x-x₀)[c₁ + (x-x₁)[c₂ + (x-x₂)[...]]]
    """
    n = len(coeffs)
    # Horner's method: evaluate from inside out
    result = coeffs[n - 1]
    for i in range(n - 2, -1, -1):
        result = result * (x - x_nodes[i]) + coeffs[i]
    
    return result


def newton_add_point(x_nodes, coeffs, x_new, y_new):
    """
    Add a new data point to an existing Newton interpolation.
    
    This is the key advantage: O(n) to add a point.
    """
    x_nodes_new = np.append(x_nodes, x_new)
    
    # Compute the new divided difference
    # f[x₀, ..., xₙ₊₁] using the existing polynomial
    # The new coefficient is (y_new - P(x_new)) / Π(x_new - xᵢ)
    P_at_new = newton_interpolation(x_nodes, coeffs, x_new)
    omega = np.prod(x_new - x_nodes)
    new_coeff = (y_new - P_at_new) / omega
    
    coeffs_new = np.append(coeffs, new_coeff)
    return x_nodes_new, coeffs_new


def forward_differences(y_nodes):
    """
    Forward difference table for equally spaced data.
    
    Δ⁰f_i = f_i
    Δ¹f_i = f_{i+1} - f_i
    Δ²f_i = Δf_{i+1} - Δf_i
    ...
    
    Used in Newton's forward difference interpolation formula.
    """
    n = len(y_nodes)
    table = np.zeros((n, n))
    table[:, 0] = y_nodes
    
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = table[i+1, j-1] - table[i, j-1]
    
    return table


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("NEWTON INTERPOLATION DEMO")
    print("=" * 60)

    # --- Basic example ---
    # TO TEST: Modify x_nodes spacing, interpolation degree (number of nodes), and x_test; observe changes in interpolation error and divided-difference coefficients.
    print("\n--- Interpolate exp(x) at 5 points ---")
    x_nodes = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    y_nodes = np.exp(x_nodes)
    
    coeffs, table = divided_differences(x_nodes, y_nodes)
    
    print("Divided difference table:")
    for i in range(len(x_nodes)):
        row = " ".join(f"{table[i,j]:10.6f}" for j in range(len(x_nodes) - i))
        print(f"  {row}")
    
    print(f"\nCoefficients: {np.round(coeffs, 6)}")
    
    # Evaluate
    x_test = 0.75
    P = newton_interpolation(x_nodes, coeffs, x_test)
    exact = np.exp(x_test)
    print(f"\nP({x_test}) = {P:.10f}")
    print(f"exp({x_test}) = {exact:.10f}")
    print(f"Error = {abs(P - exact):.2e}")

    # --- Adding a point incrementally ---
    # TO TEST: Change the added point (x_new, y_new) to values inside and outside the original node range; observe whether the updated polynomial improves local or global accuracy.
    print("\n--- Adding point x=2.5 incrementally ---")
    x_new, coeffs_new = newton_add_point(x_nodes, coeffs, 2.5, np.exp(2.5))
    P_new = newton_interpolation(x_new, coeffs_new, x_test)
    print(f"Before: P({x_test}) error = {abs(P - exact):.2e}")
    print(f"After:  P({x_test}) error = {abs(P_new - exact):.2e}")

    # --- Forward differences for equally spaced data ---
    # TO TEST: Modify x_deg to keep or break equal spacing and compare forward-difference magnitudes; observe how table structure reflects smoothness and spacing assumptions.
    print("\n--- Forward Difference Table (sin at 0°, 30°, 60°, 90°, 120°) ---")
    x_deg = np.array([0, 30, 60, 90, 120])
    x_rad = np.radians(x_deg)
    y_sin = np.sin(x_rad)
    
    fd_table = forward_differences(y_sin)
    print("  x(°)  |  f(x)  |  Δf  |  Δ²f  |  Δ³f  |  Δ⁴f")
    for i in range(len(x_deg)):
        row = f"  {x_deg[i]:4d}  | "
        row += " | ".join(f"{fd_table[i,j]:+.4f}" for j in range(len(x_deg) - i))
        print(row)

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        x_fine = np.linspace(0, 2, 200)
        y_interp = [newton_interpolation(x_nodes, coeffs, xi) for xi in x_fine]
        
        axes[0].plot(x_fine, np.exp(x_fine), 'k-', lw=2, label='exp(x)')
        axes[0].plot(x_fine, y_interp, 'b--', label=f'Newton P_{len(x_nodes)-1}(x)')
        axes[0].plot(x_nodes, y_nodes, 'ro', markersize=8, label='Data points')
        axes[0].set_title('Newton Interpolation of exp(x)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Error
        error_curve = [abs(newton_interpolation(x_nodes, coeffs, xi) - np.exp(xi)) 
                       for xi in x_fine]
        axes[1].semilogy(x_fine, error_curve, 'r-')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('|Error|')
        axes[1].set_title('Interpolation Error')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("newton_interpolation.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
