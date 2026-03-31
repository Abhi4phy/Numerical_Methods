"""
Density Matrix Renormalization Group (DMRG)
=============================================
A powerful variational method for 1D quantum many-body systems.

**The Many-Body Problem:**
A chain of L quantum spins has Hilbert space dimension d^L (exponential!).
Exact diagonalization is impossible for L > ~20.

DMRG systematically truncates the Hilbert space by keeping only the most
important states (via SVD / density matrix eigenvalues).

**The Key Idea (Matrix Product States):**
Instead of storing the full state vector (d^L components), represent
the quantum state as a product of matrices:

    |ψ⟩ = Σ_{s₁...s_L} A^{s₁} A^{s₂} ... A^{s_L} |s₁ s₂ ... s_L⟩

Each A^{sᵢ} is a matrix of dimension m × m (the "bond dimension").
Accuracy improves with m, with exact result at m = d^{L/2}.

**Algorithm (Infinite-size DMRG):**
1. Start with small system + environment
2. Form "superblock" Hamiltonian
3. Find ground state of superblock
4. Form reduced density matrix ρ = Tr_env |ψ⟩⟨ψ|
5. Keep m largest eigenvalues of ρ → truncated basis
6. Grow system by one site, repeat

**Finite-size DMRG:**
Sweep left and right through the chain, optimizing each site.

**This implementation:** Simplified DMRG for the spin-1/2 Heisenberg chain:
    H = J Σᵢ (Sᵢˣ Sᵢ₊₁ˣ + Sᵢʸ Sᵢ₊₁ʸ + Sᵢᶻ Sᵢ₊₁ᶻ)

Physics: magnetism, quantum phase transitions, topological phases.

Where to start:
━━━━━━━━━━━━━━
Understand how SVD truncation works (01_Linear_Algebra/svd.py).
Run the Heisenberg chain and check energy against exact Bethe ansatz.
"""

import numpy as np
from scipy.sparse import kron as sp_kron, identity as sp_eye
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


# --- Spin-1/2 operators ---
Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=float)
Sp = np.array([[0, 1], [0, 0]], dtype=float)  # S+
Sm = np.array([[0, 0], [1, 0]], dtype=float)  # S-
I2 = np.eye(2)


class Block:
    """
    Represents a block of sites in DMRG.
    
    Stores the block Hamiltonian and boundary operators
    needed for constructing the superblock.
    """
    
    def __init__(self, length, basis_size, H, Sz_op, Sp_op, Sm_op):
        self.length = length
        self.basis_size = basis_size
        self.H = H          # Block Hamiltonian
        self.Sz = Sz_op      # Boundary Sz operator
        self.Sp = Sp_op      # Boundary S+ operator
        self.Sm = Sm_op      # Boundary S- operator


def initial_block():
    """Create a single-site block."""
    return Block(
        length=1,
        basis_size=2,
        H=csr_matrix(np.zeros((2, 2))),
        Sz_op=csr_matrix(Sz),
        Sp_op=csr_matrix(Sp),
        Sm_op=csr_matrix(Sm)
    )


def enlarge_block(block, J=1.0):
    """
    Add one site to the block.
    
    New block Hamiltonian:
    H_new = H_old ⊗ I + I ⊗ H_site + J·(S_boundary · S_new)
    
    The interaction term connects the old boundary to the new site.
    """
    d = 2  # Single site dimension
    
    # Enlarge operators: old ⊗ I_site
    H_enlarged = sp_kron(block.H, sp_eye(d), format='csr')
    
    # Interaction: J * (S_z·S_z + 0.5*(S+·S- + S-·S+))
    interaction = J * (
        sp_kron(block.Sz, csr_matrix(Sz), format='csr')
        + 0.5 * sp_kron(block.Sp, csr_matrix(Sm), format='csr')
        + 0.5 * sp_kron(block.Sm, csr_matrix(Sp), format='csr')
    )
    
    H_enlarged = H_enlarged + interaction
    
    # New boundary operators are just the new site operators
    new_basis = block.basis_size * d
    Sz_new = sp_kron(sp_eye(block.basis_size), csr_matrix(Sz), format='csr')
    Sp_new = sp_kron(sp_eye(block.basis_size), csr_matrix(Sp), format='csr')
    Sm_new = sp_kron(sp_eye(block.basis_size), csr_matrix(Sm), format='csr')
    
    return Block(
        length=block.length + 1,
        basis_size=new_basis,
        H=H_enlarged,
        Sz_op=Sz_new,
        Sp_op=Sp_new,
        Sm_op=Sm_new
    )


def superblock_hamiltonian(sys_block, env_block, J=1.0):
    """
    Construct the superblock Hamiltonian: H_sys ⊗ I_env + I_sys ⊗ H_env + H_int
    
    The interaction connects the boundary of system to boundary of environment.
    """
    sys_dim = sys_block.basis_size
    env_dim = env_block.basis_size
    
    # System and environment Hamiltonians
    H = (sp_kron(sys_block.H, sp_eye(env_dim), format='csr')
         + sp_kron(sp_eye(sys_dim), env_block.H, format='csr'))
    
    # System-environment interaction
    H_int = J * (
        sp_kron(sys_block.Sz, env_block.Sz, format='csr')
        + 0.5 * sp_kron(sys_block.Sp, env_block.Sm, format='csr')
        + 0.5 * sp_kron(sys_block.Sm, env_block.Sp, format='csr')
    )
    
    return H + H_int


def truncate(block, psi, m_states, sys_dim, env_dim):
    """
    DMRG truncation step.
    
    1. Reshape ground state |ψ⟩ into matrix Ψ_{ij} (system × environment)
    2. Form reduced density matrix ρ = Ψ·Ψ†
    3. Keep m largest eigenstates → truncation matrix O
    4. Transform all operators: A → O† A O
    
    Returns truncated block and truncation error.
    """
    # Reshape to matrix
    psi_matrix = psi.reshape(sys_dim, env_dim)
    
    # Reduced density matrix (SVD is more stable than ρ = ψ·ψ†)
    U, s, Vh = np.linalg.svd(psi_matrix, full_matrices=False)
    
    # Keep m_states largest singular values
    m = min(m_states, len(s))
    
    # Truncation error = sum of discarded singular values squared
    trunc_error = 1.0 - np.sum(s[:m]**2) / np.sum(s**2)
    
    # Truncation matrix
    O = csr_matrix(U[:, :m])
    O_dense = U[:, :m]
    
    # Transform operators
    H_trunc = csr_matrix(O_dense.T @ block.H.toarray() @ O_dense)
    Sz_trunc = csr_matrix(O_dense.T @ block.Sz.toarray() @ O_dense)
    Sp_trunc = csr_matrix(O_dense.T @ block.Sp.toarray() @ O_dense)
    Sm_trunc = csr_matrix(O_dense.T @ block.Sm.toarray() @ O_dense)
    
    return Block(
        length=block.length,
        basis_size=m,
        H=H_trunc,
        Sz_op=Sz_trunc,
        Sp_op=Sp_trunc,
        Sm_op=Sm_trunc
    ), trunc_error


def infinite_dmrg(L_target, m_states, J=1.0, verbose=True):
    """
    Infinite-size DMRG algorithm.
    
    Grows the chain symmetrically: system + environment, each
    growing by one site per step.
    
    Parameters
    ----------
    L_target : int
        Target chain length (must be even, ≥ 4)
    m_states : int
        Number of states to keep (bond dimension)
    J : float
        Heisenberg coupling constant
    verbose : bool
        Print progress
    
    Returns
    -------
    energies : list
        Ground state energy per site at each step
    trunc_errors : list
        Truncation error at each step
    """
    if L_target % 2 != 0:
        L_target += 1
    
    block = initial_block()
    energies = []
    trunc_errors = []
    
    while 2 * (block.length + 1) <= L_target:
        # Enlarge system and environment blocks
        sys_block = enlarge_block(block, J)
        env_block = enlarge_block(block, J)  # Same as system (infinite DMRG)
        
        L_current = 2 * sys_block.length
        
        # Build and diagonalize superblock
        H_super = superblock_hamiltonian(sys_block, env_block, J)
        
        # Find ground state
        try:
            E0, psi0 = eigsh(H_super, k=1, which='SA')
            E0 = E0[0]
            psi0 = psi0[:, 0]
        except Exception:
            # For very small systems, use dense diagonalization
            evals, evecs = np.linalg.eigh(H_super.toarray())
            E0 = evals[0]
            psi0 = evecs[:, 0]
        
        e_per_site = E0 / L_current
        energies.append((L_current, e_per_site))
        
        # Truncate
        if sys_block.basis_size > m_states:
            block, err = truncate(sys_block, psi0, m_states,
                                  sys_block.basis_size, env_block.basis_size)
            trunc_errors.append(err)
        else:
            block = sys_block
            trunc_errors.append(0.0)
        
        if verbose:
            print(f"  L={L_current:4d}, E/L={e_per_site:.10f}, "
                  f"m={block.basis_size}, trunc_err={trunc_errors[-1]:.2e}")
    
    return energies, trunc_errors


def exact_heisenberg_energy(L, J=1.0):
    """
    Exact diagonalization of the Heisenberg chain (for small L).
    
    For comparison with DMRG results.
    """
    if L > 16:
        return None  # Too large for ED
    
    dim = 2**L
    H = np.zeros((dim, dim))
    
    for i in range(L - 1):  # Open boundary conditions
        for state in range(dim):
            # Extract spins at sites i and i+1
            si = (state >> i) & 1
            si1 = (state >> (i+1)) & 1
            
            # Sz_i · Sz_{i+1}
            sz_i = 0.5 - si
            sz_i1 = 0.5 - si1
            H[state, state] += J * sz_i * sz_i1
            
            # S+_i S-_{i+1} + S-_i S+_{i+1} (spin flip terms)
            if si == 1 and si1 == 0:  # can flip i down, i+1 up
                new_state = state ^ (1 << i) ^ (1 << (i+1))
                H[state, new_state] += J * 0.5
            if si == 0 and si1 == 1:  # can flip i up, i+1 down
                new_state = state ^ (1 << i) ^ (1 << (i+1))
                H[state, new_state] += J * 0.5
    
    evals = np.linalg.eigvalsh(H)
    return evals[0]


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DMRG — DENSITY MATRIX RENORMALIZATION GROUP")
    print("=" * 60)

    # TO TEST: Change chain lengths L and m_states in infinite_dmrg for small systems.
    # Observe DMRG-vs-exact energy error scaling as system size grows.
    # --- 1. Small chain: compare with exact ---
    print("\n--- Small Chain: DMRG vs Exact Diagonalization ---")
    small_chain_results = []
    for L in [4, 6, 8, 10, 12]:
        E_exact = exact_heisenberg_energy(L)
        
        # Run DMRG
        energies, _ = infinite_dmrg(L, m_states=20, verbose=False)
        if energies:
            E_dmrg = energies[-1][1] * L
            small_chain_results.append((L, E_exact, E_dmrg))
            print(f"  L={L:3d}: E_exact={E_exact:.8f}, E_DMRG={E_dmrg:.8f}, "
                  f"error={abs(E_dmrg - E_exact):.2e}")

    # TO TEST: Sweep bond dimensions m and target length L; compare against Bethe-limit energy density.
    # Observe truncation error and E/L improving monotonically with larger m.
    # --- 2. Convergence with bond dimension ---
    print("\n--- Bond Dimension Convergence (L=20) ---")
    L = 20
    # Exact E/L for infinite chain (Bethe ansatz): E/L = 1/4 - ln(2) ≈ -0.443147
    E_bethe = 0.25 - np.log(2)
    print(f"  Bethe ansatz (infinite chain): E/L = {E_bethe:.6f}")
    
    bond_results = []
    for m in [4, 8, 16, 32]:
        energies, trunc = infinite_dmrg(L, m_states=m, verbose=False)
        if energies:
            e_per_site = energies[-1][1]
            bond_results.append((m, e_per_site))
            print(f"  m={m:3d}: E/L = {e_per_site:.8f}, "
                  f"trunc_err = {trunc[-1]:.2e}")

    # TO TEST: Increase/decrease L and m_states for runtime-accuracy tradeoff studies.
    # Observe how truncation error behavior changes as entanglement burden increases.
    # --- 3. Larger chain ---
    print("\n--- Larger Chain (L=30, m=20) ---")
    energies_40, trunc_40 = infinite_dmrg(30, m_states=20, verbose=True)

    # TO TEST: Compare critical-like settings versus noncritical variants by adjusting model/coupling assumptions.
    # Observe qualitative links between entanglement expectations and required bond dimension.
    # --- 4. Entanglement growth ---
    print("\n--- Entanglement Entropy ---")
    print("  In 1D, entanglement entropy S ∝ (c/3)·log(L) for critical systems")
    print("  The Heisenberg chain is critical with central charge c = 1")
    print("  DMRG works well because entanglement is bounded (1D area law)")
    
    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Energy per site vs L
        ax = axes[0, 0]
        Ls = [e[0] for e in energies_40]
        eps = [e[1] for e in energies_40]
        ax.plot(Ls, eps, 'bo-', markersize=5)
        ax.axhline(E_bethe, color='r', linestyle='--', 
                   label=f'Bethe (∞): {E_bethe:.6f}')
        ax.set_xlabel('Chain length L')
        ax.set_ylabel('E / L')
        ax.set_title('Ground State Energy per Site')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Truncation error
        ax = axes[0, 1]
        ax.semilogy(Ls, trunc_40, 'rs-', markersize=5)
        ax.set_xlabel('Chain length L')
        ax.set_ylabel('Truncation error')
        ax.set_title(f'DMRG Truncation Error (m=20)')
        ax.grid(True, alpha=0.3)
        
        # Bond dimension convergence
        ax = axes[1, 0]
        ms = [m for m, _ in bond_results]
        e_vs_m = [e for _, e in bond_results]
        
        ax.semilogx(ms, e_vs_m, 'go-', markersize=8, linewidth=2)
        ax.axhline(E_bethe, color='r', linestyle='--', label='Bethe ansatz')
        ax.set_xlabel('Bond dimension m')
        ax.set_ylabel('E / L')
        ax.set_title('Convergence with Bond Dimension')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # DMRG vs exact
        ax = axes[1, 1]
        Ls_exact = [L for L, _, _ in small_chain_results]
        e_exact = [E_exact / L for L, E_exact, _ in small_chain_results]
        e_dmrg_list = [E_dmrg / L for L, _, E_dmrg in small_chain_results]
        
        ax.semilogy(Ls_exact, np.abs(np.array(e_dmrg_list) - np.array(e_exact)), 
                    'ko-', markersize=8)
        ax.set_xlabel('Chain length L')
        ax.set_ylabel('|E_DMRG - E_exact| / L')
        ax.set_title('DMRG Error vs Exact')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("dmrg.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
