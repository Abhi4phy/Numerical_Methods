"""
Sparse Matrix Methods
======================
Efficient storage and operations for matrices where most entries are zero.

Real-world PDE discretizations produce matrices that are >99% zeros.
Storing and operating on only the non-zero entries is essential.

**Storage Formats:**

1. **COO (Coordinate)** — store (row, col, value) triplets
   - Easy to construct, inefficient for arithmetic
   
2. **CSR (Compressed Sparse Row)** — three arrays:
   - values: non-zero entries (row-major order)
   - col_indices: column index of each entry
   - row_ptr: index into values where each row starts
   Best for row slicing and matrix-vector products.
   
3. **CSC (Compressed Sparse Column)** — like CSR but column-major
   Best for column slicing.

**Key Operations:**
- SpMV (sparse matrix-vector): y = A·x  — O(nnz) instead of O(n²)
- Sparse direct solvers (LU with fill-reducing ordering)
- Iterative solvers (CG, GMRES) only need SpMV

**Fill-in:**
Direct factorization (LU, Cholesky) creates new non-zeros.
Reordering (Reverse Cuthill-McKee, nested dissection) minimizes fill.

Physics: FEM/FDM stiffness matrices, graph Laplacians, network problems.

Where to start:
━━━━━━━━━━━━━━
Understand COO format → convert to CSR → use spmv_csr() →
try sparse_cg() for solving systems.
"""

import numpy as np
import time


class SparseMatrixCOO:
    """
    Coordinate (COO) format sparse matrix.
    
    Store non-zeros as (row, col, value) triplets.
    Easy to build incrementally.
    """
    
    def __init__(self, shape):
        self.shape = shape
        self.rows = []
        self.cols = []
        self.data = []
    
    def add(self, row, col, value):
        """Add a non-zero entry."""
        if abs(value) > 0:
            self.rows.append(row)
            self.cols.append(col)
            self.data.append(value)
    
    @property
    def nnz(self):
        return len(self.data)
    
    def to_dense(self):
        """Convert to dense matrix."""
        A = np.zeros(self.shape)
        for r, c, v in zip(self.rows, self.cols, self.data):
            A[r, c] += v  # += handles duplicates
        return A
    
    def to_csr(self):
        """Convert to CSR format."""
        return SparseMatrixCSR.from_coo(self)
    
    def matvec(self, x):
        """Matrix-vector product y = A @ x."""
        y = np.zeros(self.shape[0])
        for r, c, v in zip(self.rows, self.cols, self.data):
            y[r] += v * x[c]
        return y


class SparseMatrixCSR:
    """
    Compressed Sparse Row (CSR) format.
    
    Three arrays:
    - data: non-zero values (sorted by row, then column)
    - col_idx: column index for each entry in data
    - row_ptr: row_ptr[i] = index in data where row i starts
               row_ptr[n] = nnz
    """
    
    def __init__(self, data, col_idx, row_ptr, shape):
        self.data = np.array(data, dtype=float)
        self.col_idx = np.array(col_idx, dtype=int)
        self.row_ptr = np.array(row_ptr, dtype=int)
        self.shape = shape
    
    @staticmethod
    def from_coo(coo):
        """Convert COO to CSR."""
        m, n = coo.shape
        nnz = coo.nnz
        
        # Sort by row, then column
        order = sorted(range(nnz), key=lambda k: (coo.rows[k], coo.cols[k]))
        
        data = [coo.data[k] for k in order]
        col_idx = [coo.cols[k] for k in order]
        rows_sorted = [coo.rows[k] for k in order]
        
        # Build row_ptr
        row_ptr = np.zeros(m + 1, dtype=int)
        for r in rows_sorted:
            row_ptr[r + 1] += 1
        row_ptr = np.cumsum(row_ptr)
        
        return SparseMatrixCSR(data, col_idx, row_ptr, (m, n))
    
    @staticmethod
    def from_dense(A, tol=1e-15):
        """Convert dense matrix to CSR."""
        m, n = A.shape
        data = []
        col_idx = []
        row_ptr = [0]
        
        for i in range(m):
            for j in range(n):
                if abs(A[i, j]) > tol:
                    data.append(A[i, j])
                    col_idx.append(j)
            row_ptr.append(len(data))
        
        return SparseMatrixCSR(data, col_idx, row_ptr, (m, n))
    
    @property
    def nnz(self):
        return len(self.data)
    
    def matvec(self, x):
        """
        Sparse matrix-vector product: y = A @ x.
        
        Complexity: O(nnz) instead of O(n²).
        """
        m = self.shape[0]
        y = np.zeros(m)
        
        for i in range(m):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            for k in range(start, end):
                y[i] += self.data[k] * x[self.col_idx[k]]
        
        return y
    
    def to_dense(self):
        """Convert to dense matrix."""
        A = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for k in range(self.row_ptr[i], self.row_ptr[i+1]):
                A[i, self.col_idx[k]] = self.data[k]
        return A
    
    def diagonal(self):
        """Extract diagonal entries."""
        n = min(self.shape)
        diag = np.zeros(n)
        for i in range(n):
            for k in range(self.row_ptr[i], self.row_ptr[i+1]):
                if self.col_idx[k] == i:
                    diag[i] = self.data[k]
                    break
        return diag
    
    def get(self, i, j):
        """Get entry A[i, j]."""
        for k in range(self.row_ptr[i], self.row_ptr[i+1]):
            if self.col_idx[k] == j:
                return self.data[k]
        return 0.0


def sparse_poisson_1d(N):
    """
    Build the 1D Poisson matrix (-u'' discretization) in CSR format.
    
    Tridiagonal: [-1, 2, -1] / h²
    This matrix has only 3N-2 non-zeros out of N² entries.
    """
    h = 1.0 / (N + 1)
    coo = SparseMatrixCOO((N, N))
    
    for i in range(N):
        coo.add(i, i, 2.0 / h**2)
        if i > 0:
            coo.add(i, i-1, -1.0 / h**2)
        if i < N-1:
            coo.add(i, i+1, -1.0 / h**2)
    
    return coo.to_csr()


def sparse_poisson_2d(Nx, Ny=None):
    """
    Build the 2D Poisson matrix using 5-point stencil.
    
    For an Nx×Ny grid, the matrix is (Nx·Ny) × (Nx·Ny).
    Only 5 non-zeros per row → nnz ≈ 5N out of N² entries.
    """
    if Ny is None:
        Ny = Nx
    N = Nx * Ny
    hx = 1.0 / (Nx + 1)
    hy = 1.0 / (Ny + 1)
    
    coo = SparseMatrixCOO((N, N))
    
    for j in range(Ny):
        for i in range(Nx):
            idx = j * Nx + i
            coo.add(idx, idx, 2.0/hx**2 + 2.0/hy**2)
            
            if i > 0:      coo.add(idx, idx-1, -1.0/hx**2)
            if i < Nx-1:   coo.add(idx, idx+1, -1.0/hx**2)
            if j > 0:      coo.add(idx, idx-Nx, -1.0/hy**2)
            if j < Ny-1:   coo.add(idx, idx+Nx, -1.0/hy**2)
    
    return coo.to_csr()


def sparse_cg(A_csr, b, x0=None, tol=1e-10, max_iter=None):
    """
    Conjugate Gradient using sparse matvec.
    
    Only requires A.matvec(x) — never touches dense matrix.
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    if max_iter is None:
        max_iter = 2 * n
    
    x = x0.copy()
    r = b - A_csr.matvec(x)
    p = r.copy()
    rs_old = np.dot(r, r)
    
    residuals = [np.sqrt(rs_old)]
    
    for k in range(max_iter):
        Ap = A_csr.matvec(p)
        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        residuals.append(np.sqrt(rs_new))
        
        if np.sqrt(rs_new) < tol:
            return x, residuals
        
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    
    return x, residuals


def bandwidth(A_csr):
    """Compute the bandwidth of a sparse matrix."""
    max_bw = 0
    for i in range(A_csr.shape[0]):
        for k in range(A_csr.row_ptr[i], A_csr.row_ptr[i+1]):
            bw = abs(i - A_csr.col_idx[k])
            max_bw = max(max_bw, bw)
    return max_bw


def rcm_ordering(A_csr):
    """
    Reverse Cuthill-McKee ordering — reduces bandwidth.
    
    BFS-based algorithm that produces a permutation minimizing
    the matrix bandwidth.
    """
    n = A_csr.shape[0]
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i in range(n):
        for k in range(A_csr.row_ptr[i], A_csr.row_ptr[i+1]):
            j = A_csr.col_idx[k]
            if j != i:
                adj[i].append(j)
    
    # Degree of each node
    degrees = [len(adj[i]) for i in range(n)]
    
    # Start from node with minimum degree
    visited = [False] * n
    result = []
    
    # Handle disconnected components
    while len(result) < n:
        # Find unvisited node with minimum degree
        start = -1
        min_deg = n + 1
        for i in range(n):
            if not visited[i] and degrees[i] < min_deg:
                min_deg = degrees[i]
                start = i
        
        # BFS from start
        queue = [start]
        visited[start] = True
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # Sort neighbors by degree
            neighbors = sorted([nb for nb in adj[node] if not visited[nb]],
                              key=lambda x: degrees[x])
            
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
    
    # Reverse
    result.reverse()
    return np.array(result)


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SPARSE MATRIX METHODS DEMO")
    print("=" * 60)

    # --- Example 1: 1D Poisson ---
    # Demonstrates sparse storage and efficient solver for a tridiagonal system
    # TO TEST: Change N (100 to 50, 200, 500) to see scaling effects,
    # modify tolerance (1e-12 to 1e-8, 1e-15), or try different initial guesses x0
    # Observe CG convergence rate and sparse matrix efficiency
    print("\n--- 1D Poisson (N=100) ---")
    N = 100
    A_csr = sparse_poisson_1d(N)
    print(f"Matrix size: {A_csr.shape}")
    print(f"Non-zeros:   {A_csr.nnz} / {N*N} ({100*A_csr.nnz/(N*N):.1f}%)")
    print(f"Bandwidth:   {bandwidth(A_csr)}")
    
    # Solve
    h = 1.0 / (N + 1)
    x_grid = np.linspace(h, 1-h, N)
    b = np.pi**2 * np.sin(np.pi * x_grid)
    
    x_sol, residuals = sparse_cg(A_csr, b, tol=1e-12)
    u_exact = np.sin(np.pi * x_grid)
    err = np.max(np.abs(x_sol - u_exact))
    print(f"CG converged in {len(residuals)} iters, max error = {err:.6e}")

    # --- Example 2: 2D Poisson ---
    # Shows 2D discretization with 5-point stencil, much sparser than 1D
    # TO TEST: Change grid size (20x20 to 10x10, 30x30), modify RHS to different functions,
    # adjust tolerance, or try using direct solve with sparse LU for comparison
    # Observe memory and time scaling with grid refinement
    print("\n--- 2D Poisson (20×20 grid) ---")
    Nx = 20
    A_2d = sparse_poisson_2d(Nx)
    N_total = Nx * Nx
    print(f"Matrix size: {A_2d.shape}")
    print(f"Non-zeros:   {A_2d.nnz} / {N_total**2} ({100*A_2d.nnz/N_total**2:.2f}%)")
    
    # RHS
    h2 = 1.0 / (Nx + 1)
    b_2d = np.zeros(N_total)
    for j in range(Nx):
        for i in range(Nx):
            x_pt = (i+1) * h2
            y_pt = (j+1) * h2
            b_2d[j*Nx + i] = 2*np.pi**2 * np.sin(np.pi*x_pt) * np.sin(np.pi*y_pt)
    
    x_2d, res_2d = sparse_cg(A_2d, b_2d, tol=1e-10)
    print(f"CG converged in {len(res_2d)} iterations")

    # --- Timing: sparse vs dense ---
    print("\n--- Timing: Sparse vs Dense matvec ---")
    for N in [100, 500, 1000]:
        A_sp = sparse_poisson_1d(N)
        A_dense = A_sp.to_dense()
        x_test = np.random.randn(N)
        
        t0 = time.perf_counter()
        for _ in range(100):
            A_sp.matvec(x_test)
        t_sparse = (time.perf_counter() - t0) / 100
        
        t0 = time.perf_counter()
        for _ in range(100):
            A_dense @ x_test
        t_dense = (time.perf_counter() - t0) / 100
        
        sparsity = 100 * A_sp.nnz / N**2
        print(f"  N={N:5d}: sparse={t_sparse*1e6:.0f}μs, dense={t_dense*1e6:.0f}μs, "
              f"nnz%={sparsity:.1f}%")

    # --- Example 3: Timing and RCM reordering ---
    # Compares sparse vs dense performance and bandwidth reduction techniques
    # TO TEST: Change loop iterations (100 to 1000), modify test sizes N list,
    # test different sparse matrix structures (assemble custom patterns),
    # or implement other reordering algorithms for comparison
    # Observe how sparsity percentage and matrix size affect relative speedup
    print("\n--- Reverse Cuthill-McKee bandwidth reduction ---")
    N = 50
    A_sp = sparse_poisson_1d(N)
    bw_orig = bandwidth(A_sp)
    perm = rcm_ordering(A_sp)
    print(f"Original bandwidth: {bw_orig}")
    print(f"1D Poisson is already optimal (tridiagonal), bandwidth = 1")

    # For 2D case
    A_2d_10 = sparse_poisson_2d(10)
    bw_2d = bandwidth(A_2d_10)
    print(f"2D Poisson (10×10): bandwidth = {bw_2d}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Sparsity pattern
        A_dense_small = sparse_poisson_2d(8).to_dense()
        axes[0, 0].spy(A_dense_small, markersize=2, color='navy')
        axes[0, 0].set_title('2D Poisson Sparsity (8×8 grid)')
        axes[0, 0].set_xlabel('Column')
        axes[0, 0].set_ylabel('Row')
        
        # 1D solution
        axes[0, 1].plot(x_grid, x_sol, 'b-', linewidth=2, label='Sparse CG')
        axes[0, 1].plot(x_grid, u_exact, 'r--', linewidth=1.5, label='Exact')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('u(x)')
        axes[0, 1].set_title('1D Poisson Solution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # CG convergence
        axes[1, 0].semilogy(residuals, 'b-')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Residual')
        axes[1, 0].set_title('CG Convergence')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 2D solution
        u_2d = x_2d.reshape(Nx, Nx)
        c = axes[1, 1].imshow(u_2d, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
        plt.colorbar(c, ax=axes[1, 1])
        axes[1, 1].set_title('2D Poisson Solution')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        
        plt.tight_layout()
        plt.savefig("sparse_matrices.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
