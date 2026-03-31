"""
N-Body Methods
===============
Efficient computation of gravitational/electrostatic interactions
among N particles.

**The N-Body Problem:**
Given N particles with positions rᵢ and masses mᵢ, compute the
force on each particle:

    Fᵢ = Σⱼ≠ᵢ G·mᵢ·mⱼ·(rⱼ - rᵢ) / |rⱼ - rᵢ|³

Direct summation: O(N²) — prohibitive for N > 10⁴.

**Methods Implemented:**

1. **Direct Summation** O(N²):
   Brute force — exact, simple, slow.

2. **Barnes-Hut Tree** O(N log N):
   Build a hierarchical tree (quadtree/octree). For distant groups
   of particles, approximate their effect by their center of mass.
   Controlled by opening angle θ (typically 0.5-1.0).
   Accuracy: O(θ²) per interaction.

3. **Particle-Mesh (PM)** O(N + M log M):
   - Assign particle masses to a grid (CIC/NGP interpolation)
   - Solve Poisson on the grid via FFT: ∇²φ = -4πGρ
   - Interpolate forces back to particles
   Fast but limited by grid resolution.

4. **P3M (Particle-Particle-Particle-Mesh)**:
   Combine PM for long range + direct sum for short range.
   Best of both worlds.

Physics: gravitational dynamics, galaxy formation, plasma PIC,
         molecular dynamics, cosmological simulations.

Where to start:
━━━━━━━━━━━━━━
Run the direct summation on 100 particles first. Then compare
with Barnes-Hut for 1000+ particles — see the speed difference.
Prerequisite: symplectic_integrators.py (for time integration)
"""

import numpy as np
import time


# ============================================================
# Direct N-Body
# ============================================================

def direct_forces(pos, mass, G=1.0, softening=0.01):
    """
    Direct O(N²) force computation.
    
    Parameters
    ----------
    pos : array (N, dim)
        Particle positions
    mass : array (N,)
        Particle masses
    G : float
        Gravitational constant
    softening : float
        Softening length to avoid singularity
    
    Returns
    -------
    forces : array (N, dim)
    potential : float
        Total potential energy
    """
    N = len(mass)
    dim = pos.shape[1]
    forces = np.zeros((N, dim))
    potential = 0.0
    eps2 = softening**2
    
    for i in range(N):
        for j in range(i + 1, N):
            dr = pos[j] - pos[i]
            r2 = np.sum(dr**2) + eps2
            r = np.sqrt(r2)
            r3 = r * r2
            
            fij = G * mass[i] * mass[j] / r3
            forces[i] += fij * dr
            forces[j] -= fij * dr
            
            potential -= G * mass[i] * mass[j] / r
    
    return forces, potential


def direct_forces_vectorized(pos, mass, G=1.0, softening=0.01):
    """Vectorized O(N²) — faster with NumPy but same complexity."""
    N = len(mass)
    dim = pos.shape[1]
    forces = np.zeros((N, dim))
    eps2 = softening**2
    
    for i in range(N):
        dr = pos - pos[i]  # (N, dim)
        r2 = np.sum(dr**2, axis=1) + eps2
        r2[i] = 1.0  # avoid self
        r3 = r2**1.5
        
        fmag = G * mass[i] * mass / r3
        fmag[i] = 0.0
        
        forces[i] = np.sum(fmag[:, np.newaxis] * dr, axis=0)
    
    return forces


# ============================================================
# Barnes-Hut Tree (2D Quadtree)
# ============================================================

class QuadTreeNode:
    """Node in the Barnes-Hut quadtree."""
    
    def __init__(self, center, size):
        self.center = np.array(center, dtype=float)
        self.size = size
        self.mass = 0.0
        self.com = np.zeros(2)  # Center of mass
        self.children = [None, None, None, None]  # NW, NE, SW, SE
        self.particle_idx = -1  # Index if leaf with one particle
        self.n_particles = 0
    
    def is_leaf(self):
        return all(c is None for c in self.children)
    
    def quadrant(self, pos):
        """Determine which quadrant a position falls in."""
        if pos[1] >= self.center[1]:  # North
            return 0 if pos[0] < self.center[0] else 1  # NW or NE
        else:  # South
            return 2 if pos[0] < self.center[0] else 3  # SW or SE
    
    def child_center(self, quadrant):
        """Get center of a child quadrant."""
        offset = self.size / 4
        offsets = [(-offset, offset), (offset, offset),
                   (-offset, -offset), (offset, -offset)]
        return self.center + np.array(offsets[quadrant])


def build_quadtree(pos, mass, center=None, size=None):
    """
    Build Barnes-Hut quadtree from particle positions.
    
    Parameters
    ----------
    pos : array (N, 2)
        Particle positions
    mass : array (N,)
        Particle masses
    """
    N = len(mass)
    
    if center is None:
        center = 0.5 * (pos.max(axis=0) + pos.min(axis=0))
    if size is None:
        size = 1.1 * np.max(pos.max(axis=0) - pos.min(axis=0))
    
    root = QuadTreeNode(center, size)
    
    for i in range(N):
        _insert(root, i, pos[i], mass[i], pos, mass)
    
    _compute_com(root)
    
    return root


def _insert(node, idx, position, m, pos, mass, depth=0):
    """Insert a particle into the quadtree."""
    if depth > 50:  # Safety limit
        return
    
    if node.n_particles == 0:
        # Empty node — store particle
        node.particle_idx = idx
        node.n_particles = 1
        node.mass = m
        node.com = position.copy()
        return
    
    if node.is_leaf() and node.n_particles == 1:
        # Need to subdivide — move existing particle down
        old_idx = node.particle_idx
        node.particle_idx = -1
        
        q = node.quadrant(pos[old_idx])
        if node.children[q] is None:
            node.children[q] = QuadTreeNode(node.child_center(q), node.size / 2)
        _insert(node.children[q], old_idx, pos[old_idx], mass[old_idx], 
                pos, mass, depth + 1)
    
    # Insert new particle
    q = node.quadrant(position)
    if node.children[q] is None:
        node.children[q] = QuadTreeNode(node.child_center(q), node.size / 2)
    _insert(node.children[q], idx, position, m, pos, mass, depth + 1)
    
    node.n_particles += 1


def _compute_com(node):
    """Compute center of mass for each node (post-order traversal)."""
    if node is None:
        return
    
    if node.is_leaf():
        return
    
    node.mass = 0.0
    node.com = np.zeros(2)
    
    for child in node.children:
        if child is not None:
            _compute_com(child)
            node.mass += child.mass
            node.com += child.mass * child.com
    
    if node.mass > 0:
        node.com /= node.mass


def barnes_hut_force(node, pos_i, mass_i, theta=0.7, G=1.0, softening=0.01):
    """
    Compute force on particle i using Barnes-Hut approximation.
    
    Parameters
    ----------
    node : QuadTreeNode
        Root of the quadtree
    pos_i : array (2,)
        Position of target particle
    theta : float
        Opening angle criterion. s/d < θ → use multipole.
        θ = 0: exact (N²), θ = 0.5: good accuracy, θ = 1.0: fast
    """
    if node is None or node.n_particles == 0:
        return np.zeros(2)
    
    dr = node.com - pos_i
    d2 = np.sum(dr**2) + softening**2
    d = np.sqrt(d2)
    
    # If leaf with one particle (and not self)
    if node.is_leaf() and node.n_particles == 1:
        if d2 > softening**2 * 1.1:  # Not self
            return G * mass_i * node.mass * dr / (d * d2)
        return np.zeros(2)
    
    # Opening criterion: s/d < θ
    if node.size / d < theta:
        # Far enough — use center of mass approximation
        return G * mass_i * node.mass * dr / (d * d2)
    
    # Too close — recurse into children
    force = np.zeros(2)
    for child in node.children:
        if child is not None:
            force += barnes_hut_force(child, pos_i, mass_i, theta, G, softening)
    
    return force


def compute_bh_forces(pos, mass, theta=0.7, G=1.0, softening=0.01):
    """Compute all forces using Barnes-Hut."""
    tree = build_quadtree(pos, mass)
    N = len(mass)
    forces = np.zeros((N, 2))
    
    for i in range(N):
        forces[i] = barnes_hut_force(tree, pos[i], mass[i], theta, G, softening)
    
    return forces


# ============================================================
# Particle-Mesh (PM) Method
# ============================================================

def particle_mesh_forces(pos, mass, N_grid, L, G=1.0):
    """
    Particle-Mesh method using FFT.
    
    1. Assign masses to grid (Cloud-in-Cell)
    2. Solve Poisson equation via FFT
    3. Compute force field  
    4. Interpolate forces to particles
    
    Parameters
    ----------
    pos : array (N, 2)
        Particle positions in [0, L]²
    mass : array (N,)
        Particle masses
    N_grid : int
        Grid resolution
    L : float
        Box size (periodic boundary)
    G : float
        Gravitational constant
    """
    N = len(mass)
    dx = L / N_grid
    
    # --- 1. CIC mass assignment ---
    rho = np.zeros((N_grid, N_grid))
    
    for i in range(N):
        # Wrap to [0, L)
        px = pos[i, 0] % L
        py = pos[i, 1] % L
        
        # Grid indices
        gx = px / dx
        gy = py / dx
        
        ix = int(gx) % N_grid
        iy = int(gy) % N_grid
        
        # CIC weights
        fx = gx - int(gx)
        fy = gy - int(gy)
        
        ix1 = (ix + 1) % N_grid
        iy1 = (iy + 1) % N_grid
        
        rho[iy, ix]   += mass[i] * (1-fx) * (1-fy)
        rho[iy, ix1]  += mass[i] * fx * (1-fy)
        rho[iy1, ix]  += mass[i] * (1-fx) * fy
        rho[iy1, ix1] += mass[i] * fx * fy
    
    rho /= dx**2  # Convert to density
    
    # --- 2. Solve Poisson via FFT ---
    # ∇²φ = 4πGρ → φ̂(k) = -4πG ρ̂(k) / k²
    rho_hat = np.fft.fft2(rho)
    
    kx = np.fft.fftfreq(N_grid, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(N_grid, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0  # Avoid division by zero (mean field)
    
    phi_hat = -4 * np.pi * G * rho_hat / K2
    phi_hat[0, 0] = 0.0  # Remove mean potential
    
    phi = np.real(np.fft.ifft2(phi_hat))
    
    # --- 3. Force field (negative gradient) ---
    Fx_grid = -(np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dx)
    Fy_grid = -(np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * dx)
    
    # --- 4. CIC interpolation of forces to particles ---
    forces = np.zeros((N, 2))
    
    for i in range(N):
        px = pos[i, 0] % L
        py = pos[i, 1] % L
        
        gx = px / dx
        gy = py / dx
        
        ix = int(gx) % N_grid
        iy = int(gy) % N_grid
        
        fx = gx - int(gx)
        fy = gy - int(gy)
        
        ix1 = (ix + 1) % N_grid
        iy1 = (iy + 1) % N_grid
        
        forces[i, 0] = (Fx_grid[iy, ix] * (1-fx) * (1-fy)
                        + Fx_grid[iy, ix1] * fx * (1-fy)
                        + Fx_grid[iy1, ix] * (1-fx) * fy
                        + Fx_grid[iy1, ix1] * fx * fy) * mass[i]
        
        forces[i, 1] = (Fy_grid[iy, ix] * (1-fx) * (1-fy)
                        + Fy_grid[iy, ix1] * fx * (1-fy)
                        + Fy_grid[iy1, ix] * (1-fx) * fy
                        + Fy_grid[iy1, ix1] * fx * fy) * mass[i]
    
    return forces, phi, rho


def generate_plummer_sphere(N, M=1.0, a=1.0, rng=None):
    """
    Generate N particles from a Plummer distribution (2D projection).
    
    ρ(r) = (3M / 4πa³) · (1 + r²/a²)^{-5/2}
    
    A classic model for globular clusters / galaxies.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Sample radii from Plummer distribution
    u = rng.random(N)
    r = a / np.sqrt(u**(-2/3) - 1)
    
    # Random angles
    theta = 2 * np.pi * rng.random(N)
    
    pos = np.zeros((N, 2))
    pos[:, 0] = r * np.cos(theta)
    pos[:, 1] = r * np.sin(theta)
    
    mass = np.full(N, M / N)
    
    # Velocity from virial equilibrium
    v_esc = np.sqrt(2 * M / np.sqrt(r**2 + a**2))
    v = 0.3 * v_esc * rng.random(N)  # Random fraction of escape velocity
    vel = np.zeros((N, 2))
    phi_v = 2 * np.pi * rng.random(N)
    vel[:, 0] = v * np.cos(phi_v)
    vel[:, 1] = v * np.sin(phi_v)
    
    return pos, vel, mass


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("N-BODY METHODS DEMO")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # TO TEST: Sweep Barnes-Hut opening angle theta and softening while keeping the same particle realization.
    # Observe force error versus runtime tradeoff relative to direct-sum reference.
    # --- 1. Accuracy comparison ---
    print("\n--- Barnes-Hut Accuracy vs Direct ---")
    N = 200
    pos = rng.random((N, 2)) * 10
    mass = rng.random(N)
    
    # Direct (reference)
    t0 = time.perf_counter()
    F_direct = direct_forces_vectorized(pos, mass, softening=0.1)
    t_direct = time.perf_counter() - t0
    
    for theta in [0.3, 0.5, 0.7, 1.0]:
        t0 = time.perf_counter()
        F_bh = compute_bh_forces(pos, mass, theta=theta, softening=0.1)
        t_bh = time.perf_counter() - t0
        
        rel_err = np.mean(np.sqrt(np.sum((F_bh - F_direct)**2, axis=1)) / 
                          (np.sqrt(np.sum(F_direct**2, axis=1)) + 1e-10))
        print(f"  θ={theta:.1f}: rel_err={rel_err:.4e}, "
              f"time BH={t_bh*1e3:.1f}ms vs Direct={t_direct*1e3:.1f}ms")

    # TO TEST: Increase N and compare timing scaling for direct O(N^2) and BH O(N log N).
    # Observe crossover point where BH becomes substantially faster.
    # --- 2. Timing comparison ---
    print("\n--- Scaling: Direct vs Barnes-Hut ---")
    for N in [100, 500, 1000, 2000]:
        pos_t = rng.random((N, 2)) * 10
        mass_t = np.ones(N) / N
        
        t0 = time.perf_counter()
        F_d = direct_forces_vectorized(pos_t, mass_t, softening=0.1)
        t_d = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        F_b = compute_bh_forces(pos_t, mass_t, theta=0.7, softening=0.1)
        t_b = time.perf_counter() - t0
        
        print(f"  N={N:5d}: Direct={t_d*1e3:8.1f}ms, BH={t_b*1e3:8.1f}ms, "
              f"speedup={t_d/t_b:.1f}x")

    # TO TEST: Vary N_grid and box size L for fixed particle count.
    # Observe runtime/resolution tradeoff and smoothness of grid-derived forces.
    # --- 3. Particle-Mesh ---
    print("\n--- Particle-Mesh Method ---")
    N_pm = 500
    L = 10.0
    pos_pm = rng.random((N_pm, 2)) * L
    mass_pm = np.ones(N_pm) / N_pm
    
    for N_grid in [32, 64, 128]:
        t0 = time.perf_counter()
        F_pm, phi, rho = particle_mesh_forces(pos_pm, mass_pm, N_grid, L)
        t_pm = time.perf_counter() - t0
        print(f"  Grid {N_grid:3d}×{N_grid:3d}: time = {t_pm*1e3:.1f}ms")

    # TO TEST: Change dt_sim, n_sim, and softening in the leapfrog loop.
    # Observe cluster morphology evolution and qualitative integration stability.
    # --- 4. Simple N-body integration ---
    print("\n--- Plummer Cluster Evolution ---")
    N_cl = 200
    pos_cl, vel_cl, mass_cl = generate_plummer_sphere(N_cl, M=10.0, a=2.0, rng=rng)
    
    # Leapfrog integration
    dt_sim = 0.05
    n_sim = 500
    
    pos_history = [pos_cl.copy()]
    
    F, PE = direct_forces(pos_cl, mass_cl, softening=0.2)
    acc = F / mass_cl[:, np.newaxis]
    
    for step in range(n_sim):
        vel_cl += 0.5 * dt_sim * acc
        pos_cl += dt_sim * vel_cl
        F, PE = direct_forces(pos_cl, mass_cl, softening=0.2)
        acc = F / mass_cl[:, np.newaxis]
        vel_cl += 0.5 * dt_sim * acc
        
        if step % 100 == 0:
            pos_history.append(pos_cl.copy())
    
    pos_history.append(pos_cl.copy())
    print(f"  Evolved {N_cl} particles for {n_sim} steps")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Force comparison
        ax = axes[0, 0]
        F_mag_direct = np.sqrt(np.sum(F_direct**2, axis=1))
        F_mag_bh = np.sqrt(np.sum(compute_bh_forces(pos[:200], mass[:200], 
                                                     theta=0.7, softening=0.1)**2, axis=1))
        ax.scatter(F_mag_direct, F_mag_bh, s=5, alpha=0.5)
        lim = [min(F_mag_direct.min(), F_mag_bh.min()), 
               max(F_mag_direct.max(), F_mag_bh.max())]
        ax.plot(lim, lim, 'r--', linewidth=2)
        ax.set_xlabel('|F| Direct')
        ax.set_ylabel('|F| Barnes-Hut')
        ax.set_title('Force Comparison (θ=0.7)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Scaling
        ax = axes[0, 1]
        Ns = [100, 500, 1000, 2000]
        t_directs = []
        t_bhs = []
        for N_test in Ns:
            p = rng.random((N_test, 2)) * 10
            m = np.ones(N_test) / N_test
            t0 = time.perf_counter()
            direct_forces_vectorized(p, m, softening=0.1)
            t_directs.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            compute_bh_forces(p, m, theta=0.7, softening=0.1)
            t_bhs.append(time.perf_counter() - t0)
        
        ax.loglog(Ns, t_directs, 'ro-', label='Direct O(N²)')
        ax.loglog(Ns, t_bhs, 'bs-', label='Barnes-Hut O(N log N)')
        ax.set_xlabel('N particles')
        ax.set_ylabel('Time (s)')
        ax.set_title('Scaling Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # PM density
        ax = axes[0, 2]
        _, phi_plot, rho_plot = particle_mesh_forces(pos_pm, mass_pm, 64, L)
        c = ax.imshow(rho_plot, cmap='inferno', origin='lower', 
                     extent=[0, L, 0, L])
        ax.scatter(pos_pm[:, 0], pos_pm[:, 1], s=1, c='white', alpha=0.5)
        plt.colorbar(c, ax=ax)
        ax.set_title('PM: Density Field')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Cluster snapshots
        for idx, (i, label) in enumerate([(0, 't=0'), 
                                           (len(pos_history)//2, 't=mid'),
                                           (-1, 't=final')]):
            ax = axes[1, idx]
            ax.scatter(pos_history[i][:, 0], pos_history[i][:, 1], 
                      s=10, alpha=0.5, c='navy')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'Plummer Cluster ({label})')
            ax.set_aspect('equal')
            lim_val = 15
            ax.set_xlim(-lim_val, lim_val)
            ax.set_ylim(-lim_val, lim_val)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("nbody_methods.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
