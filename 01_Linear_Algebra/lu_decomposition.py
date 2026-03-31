"""
LU Decomposition
=================
Factorizes a matrix A into a product of a Lower triangular matrix (L)
and an Upper triangular matrix (U), such that A = L @ U.

Why it matters:
- Solving Ax = b becomes two easy triangular solves: Ly = b, then Ux = y.
- Very efficient when solving multiple systems with the same A but different b.
- O(n^3/3) operations — same as Gaussian elimination but reusable.

Physics applications:
- Circuit analysis, structural mechanics, fluid dynamics discretizations.
"""

import numpy as np


def lu_decomposition(A):
    """
    Perform LU decomposition without pivoting.
    
    Parameters
    ----------
    A : ndarray, shape (n, n)
        Square matrix to decompose.
    
    Returns
    -------
    L : ndarray, shape (n, n)
        Lower triangular matrix with 1s on diagonal.
    U : ndarray, shape (n, n)
        Upper triangular matrix.
    """
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()

    for k in range(n - 1):
        if abs(U[k, k]) < 1e-12:
            raise ValueError(f"Zero pivot encountered at position ({k},{k}). "
                             "Use LU with partial pivoting instead.")
        for i in range(k + 1, n):
            # Multiplier: how much of row k to subtract from row i
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return L, U


def lu_decomposition_partial_pivoting(A):
    """
    LU decomposition with partial pivoting: PA = LU.
    
    Partial pivoting swaps rows to place the largest element on the diagonal,
    improving numerical stability.
    
    Returns
    -------
    P : ndarray – Permutation matrix
    L : ndarray – Lower triangular
    U : ndarray – Upper triangular
    """
    n = A.shape[0]
    U = A.astype(float).copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        # Find the row with the largest absolute value in column k
        max_row = np.argmax(np.abs(U[k:, k])) + k

        # Swap rows in U, L, and P
        if max_row != k:
            U[[k, max_row]] = U[[max_row, k]]
            P[[k, max_row]] = P[[max_row, k]]
            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U


def forward_substitution(L, b):
    """Solve Ly = b where L is lower triangular."""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def back_substitution(U, y):
    """Solve Ux = y where U is upper triangular."""
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x


def solve_lu(A, b):
    """Solve Ax = b using LU decomposition."""
    L, U = lu_decomposition(A)
    y = forward_substitution(L, b)
    x = back_substitution(U, y)
    return x


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LU DECOMPOSITION DEMO")
    print("=" * 60)

    # Example: solve a 4x4 system
    # TO TEST: Try different matrix sizes with np.random.rand(n,n) or use ill-conditioned matrices
    # Example: A = np.random.rand(6, 6); b = np.random.rand(6)
    A = np.array([
        [2,  1,  1,  0],
        [4,  3,  3,  1],
        [8,  7,  9,  5],
        [6,  7,  9,  8]
    ], dtype=float)

    b = np.array([1, 1, 1, 1], dtype=float)

    print("\nMatrix A:")
    print(A)
    print("\nVector b:", b)

    # Without pivoting
    # TO TEST: See which decomposition is more stable for your matrix
    L, U = lu_decomposition(A)  # No pivoting - can be numerically unstable
    print("\n--- LU (no pivoting) ---")
    print("L:\n", L)
    print("U:\n", U)
    print("L @ U:\n", L @ U)
    print("Reconstruction error: ", np.linalg.norm(A - L @ U))

    x = solve_lu(A, b)
    print("\nSolution x:", x)
    print("Verification A @ x:", A @ x)
    print("Residual ||Ax - b||:", np.linalg.norm(A @ x - b))

    # With pivoting
    P, L2, U2 = lu_decomposition_partial_pivoting(A)
    print("\n--- LU (partial pivoting) ---")
    print("P @ A == L @ U ?  Error:", np.linalg.norm(P @ A - L2 @ U2))

    # Compare with NumPy
    print("\n--- NumPy verification ---")
    x_np = np.linalg.solve(A, b)
    print("NumPy solution:", x_np)
    print("Our solution:  ", x)
    print("Difference:    ", np.linalg.norm(x - x_np))
