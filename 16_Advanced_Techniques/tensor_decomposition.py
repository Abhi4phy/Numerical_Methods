"""
Tensor Decomposition
====================
Generalize matrix factorizations (SVD, eigendecomposition) to
higher-dimensional arrays (tensors). Essential for multi-dimensional
data in quantum mechanics, data science, and signal processing.

**What is a tensor?**
A tensor of order N is an N-dimensional array:
  - Order 0: scalar
  - Order 1: vector
  - Order 2: matrix
  - Order 3+: tensor (e.g., RGB image = 3D, video = 4D)

**Key Decompositions:**

1. CP (CANDECOMP/PARAFAC) Decomposition:
   T ≈ Σᵣ aᵣ ⊗ bᵣ ⊗ cᵣ  (sum of rank-1 tensors)
   Generalizes SVD's outer product form

2. Tucker Decomposition:
   T ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃
   Core tensor G contracted with factor matrices Uₙ
   Generalizes truncated SVD

3. Tensor Train (TT) / Matrix Product States:
   T(i₁,...,iₙ) = G₁(i₁) · G₂(i₂) · ... · Gₙ(iₙ)
   Each Gₖ is a small 3D tensor → breaks curse of dimensionality

**Mode-n Unfolding:**
Reshape tensor into a matrix along mode n.
Matricization is key to many tensor algorithms.

**Applications:**
- Quantum many-body physics (MPS, DMRG)
- Chemometrics (fluorescence spectroscopy)
- Signal processing (blind source separation)
- Machine learning (recommendation systems)
- Data compression

Where to start:
━━━━━━━━━━━━━━
1. Understand mode-n unfolding — it's the bridge between tensors and matrices
2. Implement Tucker via HOSVD — it reuses SVD you already know
3. Then CP via ALS — elegant iterative algorithm
4. TT decomposition for high-dimensional problems
Prerequisites: svd.py, eigenvalue_eigenvector.py
"""

import numpy as np


# ============================================================
# Tensor Operations
# ============================================================

def mode_n_unfold(T, n):
    """
    Mode-n unfolding (matricization) of tensor T.
    
    Rearranges T into a matrix where mode-n fibers become columns.
    
    Parameters
    ----------
    T : ndarray
        Tensor of any order
    n : int
        Mode along which to unfold (0-indexed)
    
    Returns
    -------
    T_n : ndarray (shape[n], prod(other dims))
    """
    shape = T.shape
    N = len(shape)
    
    # Move mode n to front, then reshape
    order = [n] + [i for i in range(N) if i != n]
    T_perm = np.transpose(T, order)
    
    return T_perm.reshape(shape[n], -1)


def mode_n_fold(T_n, n, shape):
    """
    Fold a mode-n unfolded matrix back into a tensor.
    
    Parameters
    ----------
    T_n : ndarray
        Mode-n unfolded matrix
    n : int
        Mode
    shape : tuple
        Original tensor shape
    """
    N = len(shape)
    order = [n] + [i for i in range(N) if i != n]
    
    # Reverse permutation
    new_shape = [shape[i] for i in order]
    T_perm = T_n.reshape(new_shape)
    
    # Inverse permutation
    inv_order = [0] * N
    for i, o in enumerate(order):
        inv_order[o] = i
    
    return np.transpose(T_perm, inv_order)


def mode_n_product(T, M, n):
    """
    Mode-n product: T ×ₙ M
    
    Multiply tensor T with matrix M along mode n.
    T has shape (I₁,...,Iₙ,...,Iₙ), M has shape (J, Iₙ)
    Result has shape (I₁,...,J,...,Iₙ)
    """
    T_n = mode_n_unfold(T, n)
    result_n = M @ T_n
    
    new_shape = list(T.shape)
    new_shape[n] = M.shape[0]
    
    return mode_n_fold(result_n, n, tuple(new_shape))


def tensor_norm(T):
    """Frobenius norm of a tensor."""
    return np.sqrt(np.sum(T**2))


def outer_product(*vectors):
    """N-way outer product of vectors."""
    result = vectors[0]
    for v in vectors[1:]:
        result = np.tensordot(result, v, axes=0)
    return result


# ============================================================
# CP Decomposition via Alternating Least Squares (ALS)
# ============================================================

def cp_als(T, R, max_iter=100, tol=1e-8, seed=42):
    """
    CP decomposition via Alternating Least Squares.
    
    T ≈ Σᵣ λᵣ · a₁ᵣ ⊗ a₂ᵣ ⊗ ... ⊗ aₙᵣ
    
    Parameters
    ----------
    T : ndarray
        Input tensor
    R : int
        Target rank
    max_iter : int
        Maximum ALS iterations
    tol : float
        Convergence tolerance
    
    Returns
    -------
    factors : list of ndarray
        Factor matrices [A₁, A₂, ..., Aₙ], each shape (Iₙ, R)
    weights : ndarray
        Weights λ of shape (R,)
    errors : list
        Reconstruction error at each iteration
    """
    rng = np.random.default_rng(seed)
    N = len(T.shape)
    
    # Initialize factor matrices randomly
    factors = [rng.standard_normal((T.shape[n], R)) for n in range(N)]
    
    # Normalize columns
    for n in range(N):
        norms = np.linalg.norm(factors[n], axis=0) + 1e-30
        factors[n] /= norms
    
    errors = []
    T_norm = tensor_norm(T)
    
    for iteration in range(max_iter):
        for n in range(N):
            # Compute the Khatri-Rao product of all factors except n
            # V = (Aₙ₋₁ ⊙ ... ⊙ A₁ ⊙ Aₙ₋₁ ⊙ ... ⊙ A₀)
            V = np.ones((1, R))
            gram = np.ones((R, R))
            
            for m in range(N):
                if m != n:
                    gram *= (factors[m].T @ factors[m])
            
            # Khatri-Rao product (column-wise Kronecker)
            V = khatri_rao_product([factors[m] for m in range(N) if m != n])
            
            # Update factor n: Aₙ = T_(n) · V · (gram)⁻¹
            T_n = mode_n_unfold(T, n)
            
            # Solve least squares
            try:
                factors[n] = T_n @ V @ np.linalg.inv(gram + 1e-12 * np.eye(R))
            except np.linalg.LinAlgError:
                factors[n] = T_n @ V @ np.linalg.pinv(gram)
        
        # Normalize and extract weights
        weights = np.zeros(R)
        for r in range(R):
            columns = [factors[n][:, r] for n in range(N)]
            weight = 1.0
            for col in columns:
                norm = np.linalg.norm(col)
                weight *= norm
            weights[r] = weight
            for n in range(N):
                norm = np.linalg.norm(factors[n][:, r]) + 1e-30
                factors[n][:, r] /= norm
        
        # Reconstruction error
        T_approx = cp_reconstruct(factors, weights)
        err = tensor_norm(T - T_approx) / T_norm
        errors.append(err)
        
        if err < tol:
            break
    
    return factors, weights, errors


def khatri_rao_product(matrices):
    """
    Khatri-Rao product (column-wise Kronecker product).
    
    For matrices of shapes (I₁,R), (I₂,R), ..., (Iₙ,R)
    Result has shape (I₁·I₂·...·Iₙ, R)
    """
    result = matrices[0]
    for M in matrices[1:]:
        R = result.shape[1]
        I1 = result.shape[0]
        I2 = M.shape[0]
        new_result = np.zeros((I1 * I2, R))
        for r in range(R):
            new_result[:, r] = np.kron(result[:, r], M[:, r])
        result = new_result
    return result


def cp_reconstruct(factors, weights):
    """Reconstruct tensor from CP decomposition."""
    R = len(weights)
    shape = tuple(f.shape[0] for f in factors)
    T = np.zeros(shape)
    
    for r in range(R):
        component = weights[r] * outer_product(*[f[:, r] for f in factors])
        T += component
    
    return T


# ============================================================
# Tucker Decomposition via HOSVD
# ============================================================

def hosvd(T, ranks=None):
    """
    Higher-Order SVD (Tucker decomposition).
    
    T ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃ ...
    
    Parameters
    ----------
    T : ndarray
        Input tensor
    ranks : tuple, optional
        Truncation ranks for each mode.
        If None, keep all (exact decomposition).
    
    Returns
    -------
    core : ndarray
        Core tensor G
    factors : list of ndarray
        Orthogonal factor matrices [U₁, U₂, ...]
    """
    N = len(T.shape)
    
    if ranks is None:
        ranks = T.shape
    
    factors = []
    
    for n in range(N):
        T_n = mode_n_unfold(T, n)
        U, S, Vt = np.linalg.svd(T_n, full_matrices=False)
        
        # Truncate
        r = min(ranks[n], U.shape[1])
        factors.append(U[:, :r])
    
    # Core tensor: G = T ×₁ U₁ᵀ ×₂ U₂ᵀ ×₃ U₃ᵀ ...
    core = T.copy()
    for n in range(N):
        core = mode_n_product(core, factors[n].T, n)
    
    return core, factors


def tucker_reconstruct(core, factors):
    """Reconstruct tensor from Tucker decomposition."""
    T = core.copy()
    for n in range(len(factors)):
        T = mode_n_product(T, factors[n], n)
    return T


# ============================================================
# Tensor Train Decomposition
# ============================================================

def tensor_train(T, max_rank=None, tol=1e-10):
    """
    Tensor Train (TT) decomposition via sequential SVD.
    
    T(i₁, i₂, ..., iₙ) = G₁[i₁] · G₂[i₂] · ... · Gₙ[iₙ]
    
    where Gₖ[iₖ] is an rₖ₋₁ × rₖ matrix.
    
    Parameters
    ----------
    T : ndarray
        Input tensor
    max_rank : int, optional
        Maximum TT rank
    tol : float
        Truncation tolerance (relative to Frobenius norm)
    
    Returns
    -------
    cores : list of ndarray
        TT cores, each of shape (rₖ₋₁, Iₖ, rₖ)
    """
    N = len(T.shape)
    shape = T.shape
    delta = tol * tensor_norm(T) / np.sqrt(N - 1)
    
    cores = []
    C = T.copy()
    r_prev = 1
    
    for k in range(N - 1):
        # Reshape to matrix
        Ik = shape[k]
        C = C.reshape(r_prev * Ik, -1)
        
        # SVD
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        
        # Truncate
        if max_rank is not None:
            r = min(max_rank, len(S))
        else:
            r = len(S)
        
        # Truncation by tolerance
        cumsum = np.cumsum(S[::-1]**2)[::-1]
        r_tol = np.searchsorted(-cumsum, -delta**2) + 1
        r = min(r, r_tol, len(S))
        
        U = U[:, :r]
        S = S[:r]
        Vt = Vt[:r, :]
        
        # Store core
        cores.append(U.reshape(r_prev, Ik, r))
        
        # Continue with remaining
        C = np.diag(S) @ Vt
        r_prev = r
    
    # Last core
    cores.append(C.reshape(r_prev, shape[-1], 1))
    
    return cores


def tt_reconstruct(cores):
    """Reconstruct tensor from TT decomposition."""
    N = len(cores)
    
    # Start with first core
    result = cores[0].reshape(cores[0].shape[1], cores[0].shape[2])
    
    for k in range(1, N):
        # result has shape (..., r_k)
        # cores[k] has shape (r_k, I_{k+1}, r_{k+1})
        Ik = cores[k].shape[1]
        rk = cores[k].shape[2]
        
        # Contract last index of result with first index of core
        result = np.tensordot(result, cores[k], axes=([-1], [0]))
        # Reshape to merge first dimensions
        new_shape = list(result.shape[:-2]) + [Ik * rk]
        result = result.reshape(*result.shape[:-2], Ik, rk)
    
    return result.squeeze()


def tt_ranks(cores):
    """Get TT ranks."""
    return [1] + [c.shape[2] for c in cores[:-1]] + [1]


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TENSOR DECOMPOSITION DEMO")
    print("=" * 60)
    
    rng = np.random.default_rng(42)
    
    # TO TEST: Try different tensor shapes and unfold/fold across all modes.
    # Observe fold-back error staying near numerical round-off.
    # --- 1. Mode-n unfolding ---
    print("\n--- Mode-n Unfolding ---")
    T = rng.standard_normal((3, 4, 5))
    print(f"  Tensor shape: {T.shape}")
    for n in range(3):
        T_n = mode_n_unfold(T, n)
        T_back = mode_n_fold(T_n, n, T.shape)
        err = np.max(np.abs(T - T_back))
        print(f"  Mode-{n} unfolding: {T_n.shape}, "
              f"fold-back error: {err:.2e}")
    
    # TO TEST: Vary target rank R_test, noise amplitude, and ALS iteration cap.
    # Observe reconstruction error versus rank and convergence speed/stability.
    # --- 2. CP Decomposition ---
    print("\n--- CP Decomposition (ALS) ---")
    
    # Create a rank-3 tensor
    I1, I2, I3 = 10, 8, 6
    R_true = 3
    A_true = rng.standard_normal((I1, R_true))
    B_true = rng.standard_normal((I2, R_true))
    C_true = rng.standard_normal((I3, R_true))
    w_true = np.array([3.0, 2.0, 1.0])
    
    T_rank3 = cp_reconstruct([A_true, B_true, C_true], w_true)
    # Add small noise
    T_noisy = T_rank3 + 0.01 * rng.standard_normal(T_rank3.shape)
    
    for R_test in [1, 2, 3, 5]:
        factors, weights, errors = cp_als(T_noisy, R_test, max_iter=200)
        print(f"  Rank-{R_test}: final error = {errors[-1]:.6f} "
              f"({len(errors)} iterations)")
    
    # TO TEST: Sweep Tucker ranks from full to aggressive compression.
    # Observe error-compression tradeoff and when approximation quality drops sharply.
    # --- 3. Tucker / HOSVD ---
    print("\n--- Tucker Decomposition (HOSVD) ---")
    
    T_tucker = rng.standard_normal((20, 15, 10))
    
    for ranks in [(20, 15, 10), (10, 8, 5), (5, 4, 3), (2, 2, 2)]:
        core, factors = hosvd(T_tucker, ranks)
        T_approx = tucker_reconstruct(core, factors)
        err = tensor_norm(T_tucker - T_approx) / tensor_norm(T_tucker)
        compression = np.prod(ranks) + sum(T_tucker.shape[n] * ranks[n] 
                                           for n in range(3))
        original = np.prod(T_tucker.shape)
        print(f"  Ranks {ranks}: error = {err:.6f}, "
              f"compression = {compression}/{original} "
              f"({compression/original*100:.1f}%)")
    
    # TO TEST: Change max_rank and tensor dimensionality/shape.
    # Observe TT-rank patterns, parameter count reduction, and reconstruction error.
    # --- 4. Tensor Train ---
    print("\n--- Tensor Train Decomposition ---")
    
    # 4D tensor
    shape_4d = (8, 6, 5, 7)
    T_4d = rng.standard_normal(shape_4d)
    
    for max_r in [None, 10, 5, 3, 2]:
        cores = tensor_train(T_4d, max_rank=max_r)
        T_recon = tt_reconstruct(cores)
        err = tensor_norm(T_4d - T_recon) / tensor_norm(T_4d)
        ranks = tt_ranks(cores)
        total_params = sum(c.size for c in cores)
        print(f"  Max rank {str(max_r):>4s}: TT-ranks = {ranks}, "
              f"error = {err:.6f}, params = {total_params}")
    
    # --- 5. Mode-n product ---
    print("\n--- Mode-n Product ---")
    T_mp = rng.standard_normal((4, 5, 6))
    M = rng.standard_normal((3, 5))  # Will multiply along mode 1
    result = mode_n_product(T_mp, M, 1)
    print(f"  T shape: {T_mp.shape}, M shape: {M.shape}")
    print(f"  T ×₁ M shape: {result.shape}")
    
    # Verify against unfolding
    T_1 = mode_n_unfold(T_mp, 1)
    result_unfold = mode_n_fold(M @ T_1, 1, (4, 3, 6))
    err = tensor_norm(result - result_unfold)
    print(f"  Verification error: {err:.2e}")
    
    # --- 6. Curse of dimensionality ---
    print("\n--- Curse of Dimensionality: TT to the Rescue ---")
    
    # High-dimensional function: f(x₁,...,xₙ) = Σ sin(xᵢ)
    # This has TT-rank 2!
    for D in [3, 4, 5, 6]:
        n_grid = 10
        # Build tensor explicitly (only feasible for small D)
        grids = [np.linspace(0, np.pi, n_grid) for _ in range(D)]
        shape_hd = tuple([n_grid] * D)
        
        T_hd = np.zeros(shape_hd)
        for idx in np.ndindex(*shape_hd):
            T_hd[idx] = sum(np.sin(grids[d][idx[d]]) for d in range(D))
        
        cores = tensor_train(T_hd, max_rank=3)
        T_recon = tt_reconstruct(cores)
        err = tensor_norm(T_hd - T_recon) / tensor_norm(T_hd)
        ranks = tt_ranks(cores)
        
        full_size = n_grid**D
        tt_size = sum(c.size for c in cores)
        
        print(f"  D={D}: full = {full_size:>8d}, TT = {tt_size:>6d} "
              f"({tt_size/full_size*100:>6.2f}%), "
              f"ranks = {ranks}, error = {err:.2e}")
    
    print("\n  TT decomposition breaks the curse of dimensionality!")
    print("  Storage grows linearly in D instead of exponentially.")
