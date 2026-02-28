"""
QR Decomposition
=================
Factorizes a matrix A into Q (orthogonal) and R (upper triangular): A = QR.

Methods implemented:
1. Gram-Schmidt process (classical and modified)
2. Householder reflections (more numerically stable)

Why it matters:
- Foundation of the QR algorithm for eigenvalue computation.
- Solves least-squares problems: min ||Ax - b||.
- More numerically stable than LU for certain problems.

Physics applications:
- Normal mode analysis, quantum mechanics (orthogonalization of states).
"""

import numpy as np


def qr_gram_schmidt(A):
    """
    QR decomposition via Modified Gram-Schmidt (MGS).
    
    MGS is preferred over Classical Gram-Schmidt because it is
    more numerically stable — it re-orthogonalizes against the
    updated vectors rather than the original ones.
    
    Parameters
    ----------
    A : ndarray, shape (m, n), m >= n
    
    Returns
    -------
    Q : ndarray, shape (m, n) – orthonormal columns
    R : ndarray, shape (n, n) – upper triangular
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.astype(float).copy()

    for j in range(n):
        # Compute the norm of the j-th column
        R[j, j] = np.linalg.norm(V[:, j])
        if R[j, j] < 1e-14:
            raise ValueError("Matrix has linearly dependent columns.")
        
        # Normalize to get the j-th orthonormal vector
        Q[:, j] = V[:, j] / R[j, j]
        
        # Subtract projection of remaining columns onto Q[:, j]
        for k in range(j + 1, n):
            R[j, k] = Q[:, j] @ V[:, k]
            V[:, k] -= R[j, k] * Q[:, j]

    return Q, R


def qr_householder(A):
    """
    QR decomposition via Householder reflections.
    
    A Householder reflection H = I - 2vv^T/v^Tv zeros out all elements
    below the diagonal in one column. This is the standard method used
    in production-level numerical libraries.
    
    Returns
    -------
    Q : ndarray, shape (m, m) – full orthogonal matrix
    R : ndarray, shape (m, n) – upper triangular
    """
    m, n = A.shape
    R = A.astype(float).copy()
    Q = np.eye(m)

    for k in range(min(m - 1, n)):
        # Extract the column below the diagonal
        x = R[k:, k].copy()
        
        # Construct the Householder vector
        # We choose the sign to avoid cancellation
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x) * (-1 if x[0] < 0 else 1)
        v = x + e1
        v = v / np.linalg.norm(v)

        # Apply reflection to R: R[k:, k:] = R[k:, k:] - 2 v (v^T R[k:, k:])
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        
        # Accumulate Q: Q[:, k:] = Q[:, k:] - 2 (Q[:, k:] v) v^T
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)

    return Q, R


def solve_least_squares(A, b):
    """
    Solve the least-squares problem min ||Ax - b||_2 using QR.
    
    If A is m×n with m > n (overdetermined system), QR gives
    the least-squares solution: R x = Q^T b (using the thin QR).
    """
    Q, R = qr_gram_schmidt(A)
    n = A.shape[1]
    # Q^T b
    Qtb = Q.T @ b
    # Back-substitution on the n×n upper part of R
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Qtb[i] - R[i, i+1:n] @ x[i+1:n]) / R[i, i]
    return x


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("QR DECOMPOSITION DEMO")
    print("=" * 60)

    A = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ], dtype=float)

    print("\nMatrix A:")
    print(A)

    # Gram-Schmidt
    Q1, R1 = qr_gram_schmidt(A)
    print("\n--- Modified Gram-Schmidt ---")
    print("Q:\n", Q1)
    print("R:\n", R1)
    print("Q^T Q (should be I):\n", np.round(Q1.T @ Q1, 10))
    print("Reconstruction error:", np.linalg.norm(A - Q1 @ R1))

    # Householder
    Q2, R2 = qr_householder(A)
    print("\n--- Householder ---")
    print("Q:\n", np.round(Q2, 6))
    print("R:\n", np.round(R2, 6))
    print("Q^T Q (should be I):\n", np.round(Q2.T @ Q2, 10))
    print("Reconstruction error:", np.linalg.norm(A - Q2 @ R2))

    # Least-squares example
    print("\n--- Least-Squares Problem ---")
    # Fit y = a + b*x to noisy data
    np.random.seed(42)
    x_data = np.linspace(0, 5, 20)
    y_data = 2.0 + 3.0 * x_data + np.random.randn(20) * 0.5

    # Design matrix [1, x]
    A_ls = np.column_stack([np.ones_like(x_data), x_data])
    coeffs = solve_least_squares(A_ls, y_data)
    print(f"Fitted: y = {coeffs[0]:.4f} + {coeffs[1]:.4f} * x")
    print(f"True:   y = 2.0000 + 3.0000 * x")
