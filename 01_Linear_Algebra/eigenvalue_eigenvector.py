"""
Eigenvalue and Eigenvector Problems
====================================
Find λ and v such that A v = λ v.

Methods implemented:
1. Power Iteration – finds the dominant (largest |λ|) eigenvalue.
2. Inverse Iteration – finds the eigenvalue closest to a given shift.
3. QR Algorithm – finds ALL eigenvalues iteratively.

Why it matters:
- Eigenvalues describe natural frequencies, stability, principal components.
- Fundamental in quantum mechanics (Hamiltonian eigenproblems).
- Used in PCA, vibration analysis, stability of dynamical systems.
"""

import numpy as np


def power_iteration(A, num_iters=1000, tol=1e-10):
    """
    Power Iteration: Find the dominant eigenvalue and eigenvector.
    
    Idea: Repeatedly multiply a random vector by A. The component along
    the dominant eigenvector grows fastest, so the vector converges to it.
    
    Convergence rate: |λ₂/λ₁| — fast when the dominant eigenvalue is
    well-separated from the second largest.
    
    Parameters
    ----------
    A : ndarray, shape (n, n)
    num_iters : int
    tol : float – convergence tolerance
    
    Returns
    -------
    eigenvalue : float
    eigenvector : ndarray
    """
    n = A.shape[0]
    # Start with a random vector
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    eigenvalue_old = 0.0
    for i in range(num_iters):
        # Multiply by A
        w = A @ v
        # Rayleigh quotient gives the eigenvalue estimate
        eigenvalue = v @ w
        # Normalize
        v = w / np.linalg.norm(w)

        if abs(eigenvalue - eigenvalue_old) < tol:
            print(f"  Power iteration converged in {i+1} iterations.")
            break
        eigenvalue_old = eigenvalue

    return eigenvalue, v


def inverse_iteration(A, sigma=0.0, num_iters=1000, tol=1e-10):
    """
    Inverse Iteration: Find the eigenvalue closest to sigma.
    
    Idea: Apply power iteration to (A - σI)^{-1}. The dominant
    eigenvalue of this matrix corresponds to the eigenvalue of A
    closest to σ.
    
    Parameters
    ----------
    sigma : float – shift (target eigenvalue region)
    """
    n = A.shape[0]
    A_shifted = A - sigma * np.eye(n)
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    eigenvalue_old = 0.0
    for i in range(num_iters):
        # Solve (A - σI) w = v  instead of inverting explicitly
        w = np.linalg.solve(A_shifted, v)
        # Rayleigh quotient for the original matrix
        eigenvalue = v @ (A @ v)
        # Normalize
        v = w / np.linalg.norm(w)

        if abs(eigenvalue - eigenvalue_old) < tol:
            print(f"  Inverse iteration converged in {i+1} iterations.")
            break
        eigenvalue_old = eigenvalue

    return eigenvalue, v


def qr_algorithm(A, num_iters=200, tol=1e-10):
    """
    QR Algorithm: Find ALL eigenvalues of A.
    
    Idea: Repeatedly compute A_k = Q_k R_k, then form A_{k+1} = R_k Q_k.
    The sequence A_k converges to an upper triangular (Schur) form,
    and the diagonal entries are the eigenvalues.
    
    This is the basic unshifted version. Production code uses
    Hessenberg reduction + implicit shifts for O(n³) total cost.
    
    Returns
    -------
    eigenvalues : ndarray – eigenvalues (diagonal of converged matrix)
    """
    n = A.shape[0]
    Ak = A.astype(float).copy()

    for i in range(num_iters):
        # QR factorization
        Q, R = np.linalg.qr(Ak)
        Ak_new = R @ Q

        # Check convergence: off-diagonal elements should vanish
        off_diag = np.sum(np.abs(np.tril(Ak_new, -1)))
        if off_diag < tol:
            print(f"  QR algorithm converged in {i+1} iterations.")
            break
        Ak = Ak_new

    eigenvalues = np.diag(Ak)
    return eigenvalues


def rayleigh_quotient(A, v):
    """
    Rayleigh quotient: R(v) = v^T A v / v^T v.
    
    Gives the best eigenvalue estimate for a given approximate eigenvector.
    The error in R(v) is O(||v - v_exact||²) — quadratically accurate!
    """
    return (v @ A @ v) / (v @ v)


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EIGENVALUE / EIGENVECTOR DEMO")
    print("=" * 60)

    # Symmetric matrix (guarantees real eigenvalues)
    A = np.array([
        [4, 1, 2],
        [1, 3, 0],
        [2, 0, 5]
    ], dtype=float)

    print("\nMatrix A:")
    print(A)

    # NumPy reference
    evals_np, evecs_np = np.linalg.eigh(A)
    print("\n--- NumPy reference ---")
    print("Eigenvalues:", evals_np)

    # Power Iteration
    # TO TEST: Try different initial guesses or different matrices  
    print("\n--- Power Iteration (dominant eigenvalue) ---")
    np.random.seed(42)
    lam, v = power_iteration(A)
    print(f"Dominant eigenvalue: {lam:.8f}")
    print(f"Eigenvector: {v}")
    print(f"Verification ||Av - λv||: {np.linalg.norm(A @ v - lam * v):.2e}")

    # Inverse Iteration
    print("\n--- Inverse Iteration (smallest eigenvalue, σ=0) ---")
    np.random.seed(42)
    lam_min, v_min = inverse_iteration(A, sigma=0.0)
    print(f"Smallest eigenvalue: {lam_min:.8f}")
    print(f"Eigenvector: {v_min}")

    # Find eigenvalue near σ=3.5
    print("\n--- Inverse Iteration (near σ=3.5) ---")
    np.random.seed(42)
    lam_near, v_near = inverse_iteration(A, sigma=3.5)
    print(f"Eigenvalue near 3.5: {lam_near:.8f}")

    # QR Algorithm
    print("\n--- QR Algorithm (all eigenvalues) ---")
    all_evals = qr_algorithm(A)
    all_evals_sorted = np.sort(all_evals)
    print(f"Eigenvalues: {all_evals_sorted}")
    print(f"NumPy:       {evals_np}")
    print(f"Difference:  {np.linalg.norm(all_evals_sorted - evals_np):.2e}")
