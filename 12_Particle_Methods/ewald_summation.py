"""
Ewald Summation
================
Efficient computation of long-range interactions in periodic systems.

**The Problem:**
In a periodic system, the electrostatic/gravitational potential is:

    Φ(rᵢ) = Σⱼ Σ_n qⱼ / |rᵢ - rⱼ + nL|

where n ranges over ALL periodic images: n ∈ Z³.
This sum converges VERY slowly (conditionally convergent for charges).

**Ewald's Trick (1921):**
Split the interaction into two rapidly convergent sums:

1. **Short-range (real space):**
   Screen each charge with a Gaussian: erfc(α|r|)/|r|
   Converges exponentially fast — cut off after a few box lengths.

2. **Long-range (reciprocal space / Fourier):**
   The smooth screening charges are summed in Fourier space.
   Also converges exponentially fast.

3. **Self-energy correction:**
   Subtract the interaction of each charge with its own screening cloud.

**Total:**
    E = E_real + E_reciprocal + E_self

**Parameter α controls the split:**
- Large α: more work in real space, less in k-space (and vice versa)
- Optimal α ~ π/L balances both sums

**Computational Complexity:**
- Direct sum: O(N · N_images) — impractical
- Ewald: O(N^{3/2}) with optimal parameter choice
- Particle-Mesh Ewald (PME): O(N log N) using FFT

Physics: crystals, ionic liquids, periodic MD, Madelung constants.

Where to start:
━━━━━━━━━━━━━━
Compute the Madelung constant of NaCl — a classic test.
Compare direct sum (very slow) vs Ewald (fast and accurate).
Prerequisite: fast_fourier_transform.py
"""

import numpy as np
from scipy.special import erfc


def ewald_energy_3d(pos, charges, L, alpha=None, rcut=None, kmax=None):
    """
    Ewald summation for Coulomb energy in a 3D periodic box.
    
    E = E_real + E_reciprocal + E_self
    
    Parameters
    ----------
    pos : array (N, 3)
        Particle positions in [0, L)³
    charges : array (N,)
        Particle charges
    L : float
        Box size (cubic box)
    alpha : float, optional
        Ewald splitting parameter (default: 5/L)
    rcut : float, optional
        Real-space cutoff (default: L/2)
    kmax : int, optional
        Max reciprocal vector index (default: 5)
    
    Returns
    -------
    E_total : float
        Total electrostatic energy
    E_real : float
        Real-space contribution
    E_recip : float
        Reciprocal-space contribution
    E_self : float
        Self-energy correction
    """
    N = len(charges)
    
    if alpha is None:
        alpha = 5.0 / L
    if rcut is None:
        rcut = L / 2
    if kmax is None:
        kmax = 5
    
    V = L**3  # Box volume
    
    # --- Real-space sum ---
    E_real = 0.0
    
    # Loop over all image cells within rcut
    n_images = int(np.ceil(rcut / L))
    
    for i in range(N):
        for j in range(i, N):
            for nx in range(-n_images, n_images + 1):
                for ny in range(-n_images, n_images + 1):
                    for nz in range(-n_images, n_images + 1):
                        if i == j and nx == 0 and ny == 0 and nz == 0:
                            continue
                        
                        dr = pos[j] - pos[i] + np.array([nx, ny, nz]) * L
                        r = np.linalg.norm(dr)
                        
                        if r < rcut:
                            factor = 1.0 if i == j else 2.0  # Avoid double counting
                            E_real += 0.5 * factor * charges[i] * charges[j] * erfc(alpha * r) / r
    
    # --- Reciprocal-space sum ---
    E_recip = 0.0
    
    for mx in range(-kmax, kmax + 1):
        for my in range(-kmax, kmax + 1):
            for mz in range(-kmax, kmax + 1):
                if mx == 0 and my == 0 and mz == 0:
                    continue
                
                k = 2 * np.pi * np.array([mx, my, mz]) / L
                k2 = np.sum(k**2)
                
                # Structure factor S(k) = Σ qⱼ exp(ik·rⱼ)
                S_k = np.sum(charges * np.exp(1j * np.dot(pos, k)))
                
                E_recip += (1.0 / k2) * np.exp(-k2 / (4 * alpha**2)) * np.abs(S_k)**2
    
    E_recip *= 2 * np.pi / V
    
    # --- Self-energy correction ---
    E_self = -alpha / np.sqrt(np.pi) * np.sum(charges**2)
    
    E_total = E_real + E_recip + E_self
    
    return E_total, E_real, E_recip, E_self


def ewald_forces_3d(pos, charges, L, alpha=None, rcut=None, kmax=None):
    """
    Ewald summation for forces (negative gradient of energy).
    
    Returns force on each particle.
    """
    N = len(charges)
    
    if alpha is None:
        alpha = 5.0 / L
    if rcut is None:
        rcut = L / 2
    if kmax is None:
        kmax = 5
    
    V = L**3
    forces = np.zeros((N, 3))
    
    # --- Real-space forces ---
    n_images = int(np.ceil(rcut / L))
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for nx in range(-n_images, n_images + 1):
                for ny in range(-n_images, n_images + 1):
                    for nz in range(-n_images, n_images + 1):
                        dr = pos[j] - pos[i] + np.array([nx, ny, nz]) * L
                        r = np.linalg.norm(dr)
                        
                        if r < rcut and r > 0:
                            f = charges[i] * charges[j] * (
                                erfc(alpha * r) / r**2
                                + 2 * alpha / np.sqrt(np.pi) * np.exp(-alpha**2 * r**2) / r
                            ) * dr / r
                            forces[i] -= f  # Force on i from j
    
    # --- Reciprocal-space forces ---
    for mx in range(-kmax, kmax + 1):
        for my in range(-kmax, kmax + 1):
            for mz in range(-kmax, kmax + 1):
                if mx == 0 and my == 0 and mz == 0:
                    continue
                
                k = 2 * np.pi * np.array([mx, my, mz]) / L
                k2 = np.sum(k**2)
                
                # Structure factor
                phases = np.dot(pos, k)
                S_k = np.sum(charges * np.exp(1j * phases))
                
                prefactor = 4 * np.pi / (V * k2) * np.exp(-k2 / (4 * alpha**2))
                
                for i in range(N):
                    forces[i] += charges[i] * prefactor * np.imag(
                        S_k * np.exp(-1j * phases[i])) * k
    
    return forces


def madelung_constant_nacl(L_max=5, alpha=None):
    """
    Compute the Madelung constant of NaCl using Ewald summation.
    
    For NaCl (rock salt structure), the Madelung constant is:
    M ≈ 1.747565 (for the conventional definition)
    
    The electrostatic energy per ion pair is:
    E = -M · e² / (4πε₀ · a)
    where a is the nearest-neighbor distance.
    """
    # NaCl unit cell: 8 ions
    # Na+ at (0,0,0), (1/2,1/2,0), (1/2,0,1/2), (0,1/2,1/2)
    # Cl- at (1/2,0,0), (0,1/2,0), (0,0,1/2), (1/2,1/2,1/2)
    
    a = 1.0  # Lattice constant
    
    pos = np.array([
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],  # Na+
        [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5],  # Cl-
    ]) * a
    
    charges = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
    
    E_total, E_real, E_recip, E_self = ewald_energy_3d(
        pos, charges, L=a, alpha=alpha or 5.0/a, rcut=a/2, kmax=L_max)
    
    # Madelung constant: E_per_pair = -M / a (for unit charges)
    # 4 NaCl pairs in the unit cell
    n_pairs = 4
    M = -E_total * a / n_pairs
    
    return M, E_total


def direct_sum_madelung(n_shells):
    """
    Direct sum Madelung constant (for comparison — very slow).
    
    Sum alternating charges on a cubic lattice.
    """
    M = 0.0
    for i in range(-n_shells, n_shells + 1):
        for j in range(-n_shells, n_shells + 1):
            for k in range(-n_shells, n_shells + 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                r = np.sqrt(i**2 + j**2 + k**2)
                M += (-1)**(i + j + k) / r
    return -M


def ewald_1d(charges, positions, L, alpha=None, kmax=50):
    """
    Simplified 1D Ewald summation for a periodic chain.
    
    Useful for understanding the basic idea.
    """
    N = len(charges)
    if alpha is None:
        alpha = 5.0 / L
    
    # Real space
    E_real = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            for n in range(-3, 4):
                r = abs(positions[j] - positions[i] + n * L)
                if r > 0:
                    E_real += charges[i] * charges[j] * erfc(alpha * r) / r
    
    # Reciprocal space
    E_recip = 0.0
    for m in range(1, kmax + 1):
        k = 2 * np.pi * m / L
        S_k = np.sum(charges * np.exp(1j * k * positions))
        E_recip += np.exp(-k**2 / (4 * alpha**2)) / k * np.abs(S_k)**2
    E_recip *= 2 / L
    
    # Self energy
    E_self = -alpha / np.sqrt(np.pi) * np.sum(charges**2)
    
    return E_real + E_recip + E_self


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EWALD SUMMATION DEMO")
    print("=" * 60)

    # TO TEST: Change reciprocal cutoff kmax and compare against known Madelung constant.
    # Observe convergence rate and diminishing absolute error with larger k-space truncation.
    # --- 1. Madelung Constant of NaCl ---
    print("\n--- Madelung Constant of NaCl ---")
    M_exact = 1.7475645946  # Known value
    
    for kmax in [3, 5, 7, 10]:
        M, E = madelung_constant_nacl(L_max=kmax)
        print(f"  kmax={kmax:2d}: M = {M:.8f}, error = {abs(M - M_exact):.2e}")
    
    print(f"  Exact:    M = {M_exact:.10f}")

    # Direct sum comparison (slow!)
    print("\n  Direct sum comparison:")
    for n in [3, 5, 10, 20]:
        M_direct = direct_sum_madelung(n)
        print(f"    shells={n:3d}: M = {M_direct:.6f}, "
              f"error = {abs(M_direct - M_exact):.4e}")

    # TO TEST: Sweep alpha while holding cutoffs fixed to study real/reciprocal workload balance.
    # Observe stability of total Madelung value versus component redistribution.
    # --- 2. Ewald parameter sensitivity ---
    print("\n--- Ewald Parameter α Sensitivity ---")
    a = 1.0
    for alpha in [2.0, 3.0, 5.0, 7.0, 10.0]:
        M, E = madelung_constant_nacl(L_max=7, alpha=alpha)
        print(f"  α = {alpha:5.1f}: M = {M:.8f}")

    # TO TEST: Vary alpha, rcut, and kmax jointly to inspect E_real/E_recip/E_self cancellation.
    # Observe whether E_total remains consistent as decomposition terms individually shift.
    # --- 3. Energy components ---
    print("\n--- Energy Decomposition ---")
    pos = np.array([
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5],
    ])
    charges = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
    
    E_total, E_real, E_recip, E_self = ewald_energy_3d(
        pos, charges, L=1.0, alpha=5.0, rcut=0.5, kmax=7)
    
    print(f"  E_real     = {E_real:12.6f}")
    print(f"  E_reciprocal = {E_recip:12.6f}")
    print(f"  E_self     = {E_self:12.6f}")
    print(f"  E_total    = {E_total:12.6f}")

    # TO TEST: Modify kmax and alpha in ewald_1d for alternating-charge chain.
    # Observe convergence toward the analytic ln(2) reference.
    # --- 4. 1D Chain ---
    print("\n--- 1D Alternating Chain ---")
    # +-+-+-+- chain, Madelung constant = ln(2)
    M_1d_exact = np.log(2)
    
    N_chain = 2
    positions = np.array([0.0, 0.5])
    charges_1d = np.array([1.0, -1.0])
    L_1d = 1.0
    
    E_1d = ewald_1d(charges_1d, positions, L_1d, alpha=5.0, kmax=50)
    M_1d = -E_1d * L_1d / 2  # Nearest neighbor distance = L/2
    
    print(f"  1D Madelung: computed = {M_1d:.8f}, exact = {M_1d_exact:.8f}")
    print(f"  Error: {abs(M_1d - M_1d_exact):.2e}")

    # TO TEST: Change random seed, particle count N_rand, and neutrality enforcement.
    # Observe energy sensitivity and verify near-zero net charge before interpreting totals.
    # --- 5. Random charges (charge-neutral) ---
    print("\n--- Random Charge-Neutral System ---")
    N_rand = 20
    rng = np.random.default_rng(42)
    pos_rand = rng.random((N_rand, 3))
    charges_rand = rng.standard_normal(N_rand)
    charges_rand -= charges_rand.mean()  # Ensure neutrality
    
    E, Er, Ek, Es = ewald_energy_3d(pos_rand, charges_rand, L=1.0, 
                                      alpha=5.0, rcut=0.5, kmax=5)
    print(f"  N = {N_rand} random charges (neutral)")
    print(f"  Total energy: {E:.6f}")
    print(f"  Sum of charges: {charges_rand.sum():.2e} (should be ~0)")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Convergence with kmax
        ax = axes[0, 0]
        kmax_vals = range(1, 12)
        M_vals = [madelung_constant_nacl(L_max=k)[0] for k in kmax_vals]
        ax.plot(list(kmax_vals), M_vals, 'bo-')
        ax.axhline(M_exact, color='r', linestyle='--', label=f'Exact: {M_exact:.6f}')
        ax.set_xlabel('k_max')
        ax.set_ylabel('Madelung constant M')
        ax.set_title('Ewald Convergence with Reciprocal Cutoff')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Error vs kmax
        ax = axes[0, 1]
        M_errors = [abs(m - M_exact) for m in M_vals]
        ax.semilogy(list(kmax_vals), M_errors, 'rs-')
        ax.set_xlabel('k_max')
        ax.set_ylabel('|M - M_exact|')
        ax.set_title('Ewald Error Convergence')
        ax.grid(True, alpha=0.3)
        
        # Direct sum convergence
        ax = axes[1, 0]
        n_range = [2, 3, 5, 7, 10, 15, 20, 30]
        M_direct_vals = [direct_sum_madelung(n) for n in n_range]
        ax.plot(n_range, M_direct_vals, 'go-', label='Direct sum')
        ax.axhline(M_exact, color='r', linestyle='--', label='Exact')
        ax.set_xlabel('Number of shells')
        ax.set_ylabel('Madelung constant')
        ax.set_title('Direct Sum: Slow Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # NaCl structure
        ax = axes[1, 1]
        from mpl_toolkits.mplot3d import Axes3D
        ax.remove()
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        
        # Plot NaCl unit cell with periodic images
        for dx in range(2):
            for dy in range(2):
                for dz in range(2):
                    offset = np.array([dx, dy, dz])
                    pos_shifted = pos + offset
                    na = pos_shifted[charges > 0]
                    cl = pos_shifted[charges < 0]
                    ax.scatter(na[:, 0], na[:, 1], na[:, 2], c='blue', 
                              s=100, alpha=0.7, label='Na+' if dx==0 and dy==0 and dz==0 else '')
                    ax.scatter(cl[:, 0], cl[:, 1], cl[:, 2], c='red',
                              s=100, alpha=0.7, label='Cl-' if dx==0 and dy==0 and dz==0 else '')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('NaCl Crystal Structure')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig("ewald_summation.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
