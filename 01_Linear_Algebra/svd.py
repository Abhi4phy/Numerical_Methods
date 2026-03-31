"""
Singular Value Decomposition (SVD)
=====================================
Factor any m×n matrix A into:

    A = U Σ V^T

where:
- U (m×m) — left singular vectors (orthonormal), columns form basis for column space
- Σ (m×n) — diagonal matrix of singular values σ₁ ≥ σ₂ ≥ ... ≥ 0
- V (n×n) — right singular vectors (orthonormal), columns form basis for row space

Why SVD is fundamental:
━━━━━━━━━━━━━━━━━━━━━━
1. Works for ANY matrix (no symmetry/square requirement)
2. Reveals rank, range, null space, condition number
3. Best low-rank approximation (Eckart–Young theorem)
4. Numerically stable pseudoinverse (Moore-Penrose)
5. Principal Component Analysis (PCA) = SVD of centered data

Key relationships:
- Singular values: σᵢ = √(eigenvalue of A^T A)
- Condition number: κ(A) = σ_max / σ_min
- Rank: number of non-zero singular values
- Pseudoinverse: A⁺ = V Σ⁺ U^T

Algorithms:
- Golub-Kahan bidiagonalization + QR iteration (standard)
- One-sided Jacobi SVD (for parallel/GPU)
- Randomized SVD (for very large matrices)

Physics: data reduction, noise filtering, quantum state tomography,
         signal processing, image compression, PCA of simulation data.

Where to start:
━━━━━━━━━━━━━━
Read the docstring → run the demo → try svd_simple() on a small matrix
→ use truncated_svd() for data compression → apply to PCA.
"""

import numpy as np


def svd_power_iteration(A, k=None, tol=1e-10, max_iter=1000):
    """
    Compute SVD via successive rank-1 approximations.
    
    For each singular triplet (σ, u, v):
    1. Power iteration on A^T A to find v (right singular vector)
    2. u = Av / ||Av||
    3. σ = ||Av||
    4. Deflate: A ← A - σ u v^T
    
    Parameters
    ----------
    A : ndarray (m, n)
    k : int — number of singular values (default: min(m,n))
    
    Returns
    -------
    U, S, Vt — such that A ≈ U @ diag(S) @ Vt
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    if k is None:
        k = min(m, n)
    
    U = np.zeros((m, k))
    S = np.zeros(k)
    V = np.zeros((n, k))
    
    A_work = A.copy()
    
    for i in range(k):
        # Power iteration for dominant singular vector
        v = np.random.randn(n)
        v /= np.linalg.norm(v)
        
        for _ in range(max_iter):
            # One step of power iteration on A^T A
            v_new = A_work.T @ (A_work @ v)
            v_new_norm = np.linalg.norm(v_new)
            if v_new_norm < 1e-15:
                break
            v_new /= v_new_norm
            
            if np.abs(np.abs(np.dot(v_new, v)) - 1) < tol:
                v = v_new
                break
            v = v_new
        
        # Compute singular value and left singular vector
        Av = A_work @ v
        sigma = np.linalg.norm(Av)
        
        if sigma < 1e-14:
            break
        
        u = Av / sigma
        
        U[:, i] = u
        S[i] = sigma
        V[:, i] = v
        
        # Deflate
        A_work = A_work - sigma * np.outer(u, v)
    
    return U[:, :len(S)], S, V[:, :len(S)].T


def pseudoinverse(A, tol=1e-10):
    """
    Moore-Penrose pseudoinverse via SVD.
    
    A⁺ = V Σ⁺ U^T
    
    where Σ⁺ inverts non-zero singular values.
    Handles rank-deficient matrices gracefully.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Invert non-zero singular values
    s_inv = np.zeros_like(s)
    for i, si in enumerate(s):
        if si > tol * s[0]:  # Relative threshold
            s_inv[i] = 1.0 / si
    
    return (Vt.T * s_inv) @ U.T


def truncated_svd(A, k):
    """
    Rank-k truncated SVD — best rank-k approximation.
    
    By Eckart–Young theorem:
        A_k = U_k Σ_k V_k^T  minimizes ||A - A_k||_F
    
    Compression ratio: k(m+n+1) / (m*n)
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    
    # Error: ||A - A_k||_F = sqrt(σ_{k+1}² + ... + σ_r²)
    error = np.sqrt(np.sum(s[k:]**2))
    energy_captured = np.sum(s[:k]**2) / np.sum(s**2)
    
    return A_k, s[:k], error, energy_captured


def randomized_svd(A, k, p=10, n_iter=2):
    """
    Randomized SVD for very large matrices.
    
    Algorithm (Halko, Martinsson, Tropp 2011):
    1. Form random projection: Y = A @ Ω,  Ω ∈ R^{n×(k+p)}
    2. Orthonormalize: Q, _ = QR(Y)
    3. Form small matrix: B = Q^T @ A
    4. SVD of small matrix: Û, S, Vt = SVD(B)
    5. U = Q @ Û
    
    Power iteration improves accuracy for slowly decaying spectra.
    
    Complexity: O(mn(k+p)) instead of O(mn·min(m,n))
    """
    m, n = A.shape
    
    # Random projection
    Omega = np.random.randn(n, k + p)
    Y = A @ Omega
    
    # Power iteration for better accuracy
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)
    
    # Orthonormalize
    Q, _ = np.linalg.qr(Y)
    
    # Project to small matrix
    B = Q.T @ A
    
    # SVD of small matrix
    U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)
    
    U = Q @ U_hat
    
    return U[:, :k], S[:k], Vt[:k, :]


def pca(data, n_components=None):
    """
    Principal Component Analysis via SVD.
    
    Parameters
    ----------
    data : ndarray (n_samples, n_features)
    n_components : int — number of principal components
    
    Returns
    -------
    scores : projected data (n_samples, n_components)
    components : principal directions (n_components, n_features)
    explained_variance_ratio : fraction of variance per component
    """
    # Center data
    mean = np.mean(data, axis=0)
    X = data - mean
    
    # SVD of centered data
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Explained variance
    n = len(data)
    explained_variance = s**2 / (n - 1)
    total_var = np.sum(explained_variance)
    explained_ratio = explained_variance / total_var
    
    if n_components is None:
        n_components = min(X.shape)
    
    components = Vt[:n_components]
    scores = X @ components.T
    
    return scores, components, explained_ratio[:n_components]


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SINGULAR VALUE DECOMPOSITION (SVD) DEMO")
    print("=" * 60)

    np.random.seed(42)

    # --- Example 1: Basic SVD ---
    # Demonstrates SVD decomposition showing singular values and reconstruction
    # TO TEST: Try different matrix sizes (m,n) like (10,5), (20,10) or add more noise
    # Try comparing error with different k values: rank-1, rank-2, or full rank
    print("\n--- Basic SVD ---")
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]], dtype=float)
    
    U, S, Vt = svd_power_iteration(A, k=3)
    A_reconstructed = U @ np.diag(S) @ Vt
    print(f"Shape: {A.shape}")
    print(f"Singular values: {S}")
    print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}")
    print(f"Rank (≈): {np.sum(S > 1e-10)}")
    
    # Compare with numpy
    U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
    print(f"NumPy singular values: {S_np}")

    # --- Example 2: Pseudoinverse ---
    # Uses SVD to compute Moore-Penrose pseudoinverse for solving least-squares problems
    # TO TEST: Change the RHS vector b to different noise patterns or use random b
    # Try solving with fewer equations (remove some rows) or more equations (add rows)
    print("\n--- Pseudoinverse: Least-squares via SVD ---")
    # Overdetermined system: 4 equations, 2 unknowns
    A_ls = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=float)
    b = np.array([2.1, 3.9, 6.2, 7.8])
    
    A_pinv = pseudoinverse(A_ls)
    x_ls = A_pinv @ b
    print(f"Least-squares solution: {x_ls}")
    print(f"Residual: {np.linalg.norm(A_ls @ x_ls - b):.4f}")

    # --- Example 3: Truncated SVD ---
    # Shows low-rank approximation and compression ratio for a structured matrix
    # TO TEST: Change true_rank (5 to 3 or 10), increase/decrease noise (0.01 to 0.1),
    # or modify matrix size (m, n) to see how compression ratio scales
    # Compare energy captured at different k values
    print("\n--- Truncated SVD: Low-rank approximation ---")
    m, n = 100, 80
    # Create a matrix with known rank structure
    true_rank = 5
    A_lr = np.random.randn(m, true_rank) @ np.random.randn(true_rank, n)
    A_lr += 0.01 * np.random.randn(m, n)  # Small noise
    
    for k in [1, 3, 5, 10]:
        A_k, s_k, err, energy = truncated_svd(A_lr, k)
        ratio = k * (m + n + 1) / (m * n)
        print(f"  Rank-{k:2d}: error={err:.4f}, energy={energy:.4f}, "
              f"compression={ratio:.3f}")

    # --- Example 4: Randomized SVD ---
    # Demonstrates efficient SVD on large matrices using randomization
    # TO TEST: Increase matrix size (m, n) to 5000x2000, adjust power iteration (n_iter: 1,3,5),
    # or change k value (10 to 20 or 30) to see accuracy vs speed tradeoff
    # Compare timing differences between randomized and full SVD
    print("\n--- Randomized SVD (large matrix) ---")
    m, n = 1000, 500
    A_big = np.random.randn(m, 10) @ np.random.randn(10, n)
    
    import time
    t0 = time.perf_counter()
    U_r, S_r, Vt_r = randomized_svd(A_big, k=10)
    t_rand = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    U_f, S_f, Vt_f = np.linalg.svd(A_big, full_matrices=False)
    t_full = time.perf_counter() - t0
    
    print(f"Randomized SVD:  {t_rand*1000:.1f} ms, top sing vals = {S_r[:5].round(2)}")
    print(f"Full SVD:        {t_full*1000:.1f} ms, top sing vals = {S_f[:5].round(2)}")

    # --- Example 5: PCA ---
    # Principal Component Analysis using SVD on centered data
    # TO TEST: Change n_samples (500 to 100, 1000), modify data_3d coefficients (2*t_param),
    # adjust noise level (0.5 to 0.1 or 1.0), or try n_components 2 vs 3
    # Observe how explained variance ratio changes with different data sparsity
    print("\n--- PCA on synthetic data ---")
    n_samples = 500
    # 3D data that lies mostly on a 2D plane
    t_param = np.random.randn(n_samples)
    s_param = np.random.randn(n_samples)
    data_3d = np.column_stack([
        2*t_param + s_param,
        t_param - s_param + 0.5*np.random.randn(n_samples),
        3*t_param + 2*s_param + 0.3*np.random.randn(n_samples)
    ])
    
    scores, components, var_ratio = pca(data_3d, n_components=3)
    print(f"Explained variance ratio: {var_ratio}")
    print(f"First 2 components capture {var_ratio[:2].sum()*100:.1f}% of variance")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Singular value spectrum
        _, S_demo, _ = np.linalg.svd(A_lr, full_matrices=False)
        axes[0, 0].semilogy(range(1, len(S_demo)+1), S_demo, 'bo-', markersize=4)
        axes[0, 0].axvline(true_rank, color='r', linestyle='--', label=f'True rank = {true_rank}')
        axes[0, 0].set_xlabel('Index')
        axes[0, 0].set_ylabel('Singular value')
        axes[0, 0].set_title('Singular Value Spectrum')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Energy captured
        S_cumul = np.cumsum(S_demo**2) / np.sum(S_demo**2)
        axes[0, 1].plot(range(1, len(S_cumul)+1), S_cumul, 'rs-', markersize=4)
        axes[0, 1].axhline(0.99, color='g', linestyle='--', label='99% energy')
        axes[0, 1].set_xlabel('Number of components k')
        axes[0, 1].set_ylabel('Cumulative energy')
        axes[0, 1].set_title('Truncated SVD: Energy Captured')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # PCA 2D projection
        axes[1, 0].scatter(scores[:, 0], scores[:, 1], s=5, alpha=0.5, c='steelblue')
        axes[1, 0].set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)')
        axes[1, 0].set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)')
        axes[1, 0].set_title('PCA: 2D Projection')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Condition number visualization
        kappas = []
        ns = range(5, 55, 5)
        for ni in ns:
            H = np.zeros((ni, ni))
            for i in range(ni):
                for j in range(ni):
                    H[i, j] = 1.0 / (i + j + 1)  # Hilbert matrix
            _, sh, _ = np.linalg.svd(H)
            kappas.append(sh[0] / sh[-1])
        
        axes[1, 1].semilogy(ns, kappas, 'ro-', markersize=5)
        axes[1, 1].set_xlabel('Matrix size n')
        axes[1, 1].set_ylabel('Condition number κ')
        axes[1, 1].set_title('Hilbert Matrix: κ = σ_max/σ_min')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("svd.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
