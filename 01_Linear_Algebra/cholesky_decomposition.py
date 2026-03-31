"""
Cholesky Decomposition
=======================
For a symmetric positive-definite (SPD) matrix A, factorizes A = L L^T,
where L is a lower triangular matrix with positive diagonal entries.

Why it matters:
- Twice as fast as LU decomposition for SPD matrices.
- Numerically very stable — no pivoting needed.
- Guarantees A is positive-definite (if decomposition succeeds).

Physics applications:
- Covariance matrices in statistics and data analysis.
- Mass/stiffness matrices in finite element analysis.
- Correlation functions in many-body physics.
"""

import numpy as np


def cholesky_decomposition(A):
    """
    Compute the Cholesky decomposition A = L L^T.
    
    Algorithm (column-by-column):
        L[j,j] = sqrt(A[j,j] - sum(L[j,k]^2 for k<j))
        L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k] for k<j)) / L[j,j]   for i > j
    
    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric positive-definite matrix.
    
    Returns
    -------
    L : ndarray, shape (n, n)
        Lower triangular Cholesky factor.
    
    Raises
    ------
    ValueError : If A is not positive-definite.
    """
    n = A.shape[0]
    L = np.zeros((n, n))

    for j in range(n):
        # Diagonal element
        val = A[j, j] - np.sum(L[j, :j] ** 2)
        if val <= 0:
            raise ValueError(
                f"Matrix is not positive-definite (negative value {val:.2e} "
                f"at position ({j},{j}))."
            )
        L[j, j] = np.sqrt(val)

        # Off-diagonal elements below diagonal
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]

    return L


def solve_cholesky(A, b):
    """
    Solve Ax = b using Cholesky decomposition.
    
    Steps:
        1. A = L L^T
        2. Solve Ly = b   (forward substitution)
        3. Solve L^T x = y (back substitution)
    """
    L = cholesky_decomposition(A)
    n = len(b)

    # Forward substitution: Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]

    # Back substitution: L^T x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - L.T[i, i+1:] @ x[i+1:]) / L[i, i]

    return x


def is_positive_definite(A):
    """Check if a symmetric matrix is positive-definite using Cholesky."""
    try:
        cholesky_decomposition(A)
        return True
    except ValueError:
        return False


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CHOLESKY DECOMPOSITION DEMO")
    print("=" * 60)

    # --- Example 1: Cholesky decomposition ---
    # Factors a symmetric positive-definite matrix into LL^T form
    # TO TEST: Change matrix size (4x4 to 6x6, 10x10), modify B to create different SPD matrices,
    # or adjust diagonal scaling (0.1 to 1.0) to see numerical stability effects
    # Create a symmetric positive-definite matrix
    # A = B^T B guarantees SPD
    np.random.seed(42)
    B = np.random.randn(4, 4)
    A = B.T @ B + 0.1 * np.eye(4)  # add small diagonal for numerical safety

    print("\nSymmetric positive-definite matrix A:")
    print(np.round(A, 4))
    print("\nIs symmetric?", np.allclose(A, A.T))

    # Cholesky decomposition
    L = cholesky_decomposition(A)
    print("\nCholesky factor L:")
    print(np.round(L, 4))

    print("\nVerification: L @ L^T:")
    print(np.round(L @ L.T, 4))
    print("Reconstruction error:", np.linalg.norm(A - L @ L.T))

    # Solve a system
    b = np.array([1, 2, 3, 4], dtype=float)
    x = solve_cholesky(A, b)
    print("\nSolving Ax = b:")
    print("b =", b)
    print("x =", np.round(x, 6))
    print("Residual ||Ax-b||:", np.linalg.norm(A @ x - b))

    # Compare with NumPy
    x_np = np.linalg.solve(A, b)
    print("NumPy solution:", np.round(x_np, 6))
    print("Difference:", np.linalg.norm(x - x_np))

    # --- Example 2: Solve a system ---
    # Uses Cholesky factorization to efficiently solve Ax = b systems
    # TO TEST: Change b vector to different values (e.g., [1,1,1,1], [0,0,1,0], or random),
    # modify matrix A by changing random seed or increasing diagonal scaling
    # Try different system sizes by changing the matrix dimension
    # Create a different system for testing
    b2 = np.array([1, 1, 1, 1], dtype=float)
    x2 = solve_cholesky(A, b2)

    # --- Example 2 Output ---
    print("\nExample 2 - Solving Ax = b2:")
    print("b2 =", b2)
    print("x2 =", np.round(x2, 6))
    print("Residual ||Ax2-b2||:", np.linalg.norm(A @ x2 - b2))

    # --- Example 3: Testing non-positive-definite matrix ---
    # Demonstrates failure case when matrix lacks positive-definiteness
    # TO TEST: Try different indefinite matrices like [[2, -3], [-3, 2]] or [[0, 1], [1, 0]],
    # modify matrix C to make it positive-definite, or scale eigenvalues to see boundary cases
    print("\n--- Testing non-positive-definite matrix ---")
    C = np.array([[1, 2], [2, 1]])  # eigenvalues: 3 and -1
    print("Matrix C:", C)
    print("Is positive-definite?", is_positive_definite(C))
