"""
Level Set Methods
=================
Track interfaces (boundaries) implicitly using a scalar field φ(x,t).

**Core Idea:**
Instead of tracking interface points explicitly, define a function
φ(x,t) where the interface is the ZERO LEVEL SET: {x : φ(x,t) = 0}.

**Advantages over explicit tracking:**
- Topology changes (merging, splitting) handled automatically
- Easy extension to 3D
- Geometric quantities computed from φ:
    * Normal:     n = ∇φ / |∇φ|
    * Curvature:  κ = ∇·(∇φ / |∇φ|)

**Level Set Equation:**
    ∂φ/∂t + v·∇φ = 0    (advection by velocity v)

For motion by curvature:
    ∂φ/∂t = b·κ·|∇φ|

For motion in normal direction with speed F:
    ∂φ/∂t + F|∇φ| = 0  (Hamilton-Jacobi equation)

**Signed Distance Function:**
φ(x) = ± distance to interface, with φ < 0 inside, φ > 0 outside.
Need periodic reinitialization:
    ∂φ/∂τ + sign(φ₀)(|∇φ| - 1) = 0

**Numerical Methods:**
- Upwind schemes for Hamilton-Jacobi equations
- ENO/WENO for high-order accuracy
- Fast Marching Method for static problems (Eikonal equation)

**Applications:**
- Multi-phase flows (bubble dynamics)
- Crystal growth
- Image segmentation
- Combustion fronts
- Topology optimization

Where to start:
━━━━━━━━━━━━━━
1. Initialize a circle as a signed distance function
2. Advect it with constant velocity — watch it move
3. Try motion by curvature — watch a bumpy shape become smooth
4. Then try two merging circles
Prerequisites: finite_difference_method.py, euler_method.py
"""

import numpy as np


# ============================================================
# Signed Distance Functions (initialization)
# ============================================================

def signed_distance_circle(X, Y, cx, cy, r):
    """Signed distance function for a circle."""
    return np.sqrt((X - cx)**2 + (Y - cy)**2) - r


def signed_distance_union(phi1, phi2):
    """Union: min of two SDFs."""
    return np.minimum(phi1, phi2)


def signed_distance_intersection(phi1, phi2):
    """Intersection: max of two SDFs."""
    return np.maximum(phi1, phi2)


def signed_distance_difference(phi1, phi2):
    """Difference: phi1 minus phi2."""
    return np.maximum(phi1, -phi2)


def signed_distance_rectangle(X, Y, x0, y0, x1, y1):
    """Approximate SDF for a rectangle (not exact at corners)."""
    dx = np.maximum(np.maximum(x0 - X, X - x1), 0)
    dy = np.maximum(np.maximum(y0 - Y, Y - y1), 0)
    inside = np.maximum(np.maximum(x0 - X, X - x1), np.maximum(y0 - Y, Y - y1))
    outside = np.sqrt(dx**2 + dy**2)
    return np.where((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1),
                   inside, outside)


# ============================================================
# Numerical Operators
# ============================================================

def gradient_upwind(phi, dx, dy, vel_x=None, vel_y=None):
    """
    Upwind finite differences for |∇φ|.
    
    Uses upwind direction based on velocity or sign of φ.
    """
    Ny, Nx = phi.shape
    
    # Forward and backward differences
    Dxp = np.zeros_like(phi)  # D+ x (forward)
    Dxm = np.zeros_like(phi)  # D- x (backward)
    Dyp = np.zeros_like(phi)
    Dym = np.zeros_like(phi)
    
    Dxp[:, :-1] = (phi[:, 1:] - phi[:, :-1]) / dx
    Dxp[:, -1] = Dxp[:, -2]
    
    Dxm[:, 1:] = (phi[:, 1:] - phi[:, :-1]) / dx
    Dxm[:, 0] = Dxm[:, 1]
    
    Dyp[:-1, :] = (phi[1:, :] - phi[:-1, :]) / dy
    Dyp[-1, :] = Dyp[-2, :]
    
    Dym[1:, :] = (phi[1:, :] - phi[:-1, :]) / dy
    Dym[0, :] = Dym[1, :]
    
    return Dxp, Dxm, Dyp, Dym


def godunov_hamiltonian(Dxp, Dxm, Dyp, Dym, sign_phi):
    """
    Godunov's Hamiltonian for |∇φ| with upwinding based on sign(φ).
    
    For reinitialization equation.
    """
    grad_phi_sq = np.zeros_like(Dxp)
    
    # Where sign_phi > 0: use max(D-,0)² + min(D+,0)²
    pos = sign_phi > 0
    grad_phi_sq[pos] = (np.maximum(Dxm[pos], 0)**2 + np.minimum(Dxp[pos], 0)**2 +
                        np.maximum(Dym[pos], 0)**2 + np.minimum(Dyp[pos], 0)**2)
    
    # Where sign_phi < 0: use min(D-,0)² + max(D+,0)²
    neg = sign_phi < 0
    grad_phi_sq[neg] = (np.minimum(Dxm[neg], 0)**2 + np.maximum(Dxp[neg], 0)**2 +
                        np.minimum(Dym[neg], 0)**2 + np.maximum(Dyp[neg], 0)**2)
    
    return np.sqrt(grad_phi_sq)


def curvature(phi, dx, dy):
    """
    Compute mean curvature κ = ∇·(∇φ/|∇φ|) using central differences.
    """
    Ny, Nx = phi.shape
    
    # Central differences
    phi_x = np.zeros_like(phi)
    phi_y = np.zeros_like(phi)
    phi_xx = np.zeros_like(phi)
    phi_yy = np.zeros_like(phi)
    phi_xy = np.zeros_like(phi)
    
    phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * dx)
    phi_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * dy)
    
    phi_xx[:, 1:-1] = (phi[:, 2:] - 2*phi[:, 1:-1] + phi[:, :-2]) / dx**2
    phi_yy[1:-1, :] = (phi[2:, :] - 2*phi[1:-1, :] + phi[:-2, :]) / dy**2
    
    phi_xy[1:-1, 1:-1] = (phi[2:, 2:] - phi[2:, :-2] - 
                           phi[:-2, 2:] + phi[:-2, :-2]) / (4*dx*dy)
    
    grad_mag = np.sqrt(phi_x**2 + phi_y**2 + 1e-30)
    
    kappa = (phi_xx * phi_y**2 - 2 * phi_x * phi_y * phi_xy + 
             phi_yy * phi_x**2) / (grad_mag**3)
    
    return kappa


def heaviside_smooth(phi, epsilon):
    """Smoothed Heaviside function."""
    H = np.zeros_like(phi)
    H[phi > epsilon] = 1.0
    band = np.abs(phi) <= epsilon
    H[band] = 0.5 * (1 + phi[band]/epsilon + np.sin(np.pi*phi[band]/epsilon)/np.pi)
    return H


def delta_smooth(phi, epsilon):
    """Smoothed Dirac delta function."""
    d = np.zeros_like(phi)
    band = np.abs(phi) <= epsilon
    d[band] = 0.5 / epsilon * (1 + np.cos(np.pi * phi[band] / epsilon))
    return d


# ============================================================
# Level Set Evolution
# ============================================================

def advect_level_set(phi, vx, vy, dx, dy, dt, n_steps=1):
    """
    Advect level set by velocity field: ∂φ/∂t + v·∇φ = 0
    
    Uses upwind scheme.
    """
    for _ in range(n_steps):
        Dxp, Dxm, Dyp, Dym = gradient_upwind(phi, dx, dy)
        
        # Upwind based on velocity sign
        dphi_dx = np.where(vx > 0, Dxm, Dxp)
        dphi_dy = np.where(vy > 0, Dym, Dyp)
        
        phi = phi - dt * (vx * dphi_dx + vy * dphi_dy)
    
    return phi


def motion_by_curvature(phi, dx, dy, dt, n_steps=1, b=1.0):
    """
    Move interface by curvature: ∂φ/∂t = b·κ·|∇φ|
    
    This smooths the interface (like heat equation on the boundary).
    """
    for _ in range(n_steps):
        kappa = curvature(phi, dx, dy)
        
        # Central gradient magnitude
        phi_x = np.zeros_like(phi)
        phi_y = np.zeros_like(phi)
        phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2*dx)
        phi_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2*dy)
        grad_mag = np.sqrt(phi_x**2 + phi_y**2 + 1e-30)
        
        phi = phi + dt * b * kappa * grad_mag
    
    return phi


def motion_normal_direction(phi, F, dx, dy, dt, n_steps=1):
    """
    Move interface in normal direction: ∂φ/∂t + F|∇φ| = 0
    
    F > 0: expansion, F < 0: contraction
    """
    for _ in range(n_steps):
        Dxp, Dxm, Dyp, Dym = gradient_upwind(phi, dx, dy)
        sign_F = np.sign(F) if isinstance(F, np.ndarray) else np.sign(F)
        grad_mag = godunov_hamiltonian(Dxp, Dxm, Dyp, Dym, 
                                        np.ones_like(phi) * np.sign(F) if np.isscalar(F) else np.sign(F))
        phi = phi - dt * F * grad_mag
    
    return phi


def reinitialize(phi, dx, dy, dt_reinit=None, n_steps=20):
    """
    Reinitialize φ to a signed distance function.
    
    Solve: ∂φ/∂τ + sign(φ₀)(|∇φ| - 1) = 0
    to steady state.
    """
    if dt_reinit is None:
        dt_reinit = 0.5 * min(dx, dy)
    
    phi0 = phi.copy()
    sign_phi = phi0 / np.sqrt(phi0**2 + (min(dx, dy))**2)
    
    for _ in range(n_steps):
        Dxp, Dxm, Dyp, Dym = gradient_upwind(phi, dx, dy)
        grad_mag = godunov_hamiltonian(Dxp, Dxm, Dyp, Dym, sign_phi)
        phi = phi - dt_reinit * sign_phi * (grad_mag - 1)
    
    return phi


def interface_area(phi, dx, dy, epsilon=None):
    """Compute interface length (2D) / area (3D) using smoothed delta."""
    if epsilon is None:
        epsilon = 1.5 * max(dx, dy)
    delta = delta_smooth(phi, epsilon)
    
    phi_x = np.zeros_like(phi)
    phi_y = np.zeros_like(phi)
    phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2*dx)
    phi_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2*dy)
    grad_mag = np.sqrt(phi_x**2 + phi_y**2 + 1e-30)
    
    return np.sum(delta * grad_mag) * dx * dy


def enclosed_area(phi, dx, dy, epsilon=None):
    """Compute enclosed area using smoothed Heaviside."""
    if epsilon is None:
        epsilon = 1.5 * max(dx, dy)
    H = heaviside_smooth(-phi, epsilon)  # H=1 inside (φ<0)
    return np.sum(H) * dx * dy


# ============================================================
# Fast Marching Method (for distance computation)
# ============================================================

def fast_marching_1d(phi_init, dx):
    """
    Simplified 1D Fast Marching Method.
    
    Solve the Eikonal equation |∇φ| = 1 from known boundary values.
    """
    N = len(phi_init)
    phi = np.full(N, np.inf)
    known = np.zeros(N, dtype=bool)
    
    # Initialize from zero crossings
    for i in range(N-1):
        if phi_init[i] * phi_init[i+1] <= 0:
            # Linear interpolation to find exact crossing
            frac = abs(phi_init[i]) / (abs(phi_init[i]) + abs(phi_init[i+1]) + 1e-30)
            phi[i] = frac * dx
            phi[i+1] = (1 - frac) * dx
            known[i] = True
            known[i+1] = True
    
    # March outward
    for _ in range(N):
        # Find smallest unknown
        trial = np.where(~known)[0]
        if len(trial) == 0:
            break
        
        # Find trial point adjacent to known with smallest value
        candidates = []
        for idx in trial:
            if idx > 0 and known[idx-1]:
                val = phi[idx-1] + dx
                candidates.append((val, idx))
            if idx < N-1 and known[idx+1]:
                val = phi[idx+1] + dx
                candidates.append((val, idx))
        
        if not candidates:
            break
        
        candidates.sort()
        new_val, new_idx = candidates[0]
        phi[new_idx] = new_val
        known[new_idx] = True
    
    # Restore signs
    phi *= np.sign(phi_init + 1e-30)
    
    return phi


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LEVEL SET METHODS DEMO")
    print("=" * 60)

    # --- 1. Circle advection ---
    print("\n--- Circle Advection ---")
    N = 100
    L = 4.0
    dx = dy = L / N
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)
    
    # Initialize circle
    phi = signed_distance_circle(X, Y, -0.5, 0.0, 0.5)
    
    # Constant velocity
    vx = np.ones_like(phi)
    vy = 0.2 * np.ones_like(phi)
    
    dt = 0.5 * dx / np.sqrt(2)
    n_steps = 50
    
    area0 = enclosed_area(phi, dx, dy)
    phi_adv = advect_level_set(phi, vx, vy, dx, dy, dt, n_steps)
    area1 = enclosed_area(phi_adv, dx, dy)
    
    print(f"  Grid: {N}×{N}, dx = {dx:.4f}")
    print(f"  Advection for {n_steps} steps (dt = {dt:.4f})")
    print(f"  Area before: {area0:.4f}")
    print(f"  Area after:  {area1:.4f}")
    print(f"  Area change: {abs(area1-area0)/area0*100:.2f}%")

    # --- 2. Motion by curvature ---
    print("\n--- Motion by Curvature ---")
    
    # Start with a bumpy circle
    theta = np.arctan2(Y, X)
    r_bumpy = 1.0 + 0.3 * np.cos(5 * theta)
    phi_bumpy = np.sqrt(X**2 + Y**2) - r_bumpy
    
    perimeter0 = interface_area(phi_bumpy, dx, dy)
    area0 = enclosed_area(phi_bumpy, dx, dy)
    
    dt_curv = 0.25 * dx**2
    phi_smooth = motion_by_curvature(phi_bumpy, dx, dy, dt_curv, n_steps=200)
    
    perimeter1 = interface_area(phi_smooth, dx, dy)
    area1 = enclosed_area(phi_smooth, dx, dy)
    
    print(f"  Bumpy circle → smooth:")
    print(f"  Perimeter: {perimeter0:.3f} → {perimeter1:.3f}")
    print(f"  Area:      {area0:.3f} → {area1:.3f}")
    print(f"  (Curvature flow reduces perimeter, shrinks area)")

    # --- 3. Topology change: merging circles ---
    print("\n--- Merging Circles ---")
    
    phi1 = signed_distance_circle(X, Y, -0.6, 0.0, 0.7)
    phi2 = signed_distance_circle(X, Y, 0.6, 0.0, 0.7)
    phi_two = signed_distance_union(phi1, phi2)
    
    # Expand
    phi_merged = motion_normal_direction(phi_two, -0.5, dx, dy, 0.5*dx, 30)
    phi_merged = reinitialize(phi_merged, dx, dy)
    
    area_init = enclosed_area(phi_two, dx, dy)
    area_merged = enclosed_area(phi_merged, dx, dy)
    print(f"  Initial (two circles): area = {area_init:.3f}")
    print(f"  After expansion: area = {area_merged:.3f}")

    # --- 4. CSG operations ---
    print("\n--- Constructive Solid Geometry ---")
    
    circ = signed_distance_circle(X, Y, 0, 0, 1.0)
    rect = signed_distance_rectangle(X, Y, -0.5, -0.5, 0.5, 0.5)
    
    union = signed_distance_union(circ, rect)
    inter = signed_distance_intersection(circ, rect)
    diff = signed_distance_difference(circ, rect)
    
    print(f"  Circle area:       {enclosed_area(circ, dx, dy):.3f} (exact: {np.pi:.3f})")
    print(f"  Rectangle area:    {enclosed_area(rect, dx, dy):.3f} (exact: 1.000)")
    print(f"  Union area:        {enclosed_area(union, dx, dy):.3f}")
    print(f"  Intersection area: {enclosed_area(inter, dx, dy):.3f}")
    print(f"  Difference area:   {enclosed_area(diff, dx, dy):.3f}")

    # --- 5. Reinitialization ---
    print("\n--- Reinitialization to SDF ---")
    
    # Start with a non-SDF level set
    phi_distorted = np.sin(2*np.pi*X/L) * np.sin(2*np.pi*Y/L)
    
    # Check |∇φ| = 1 property
    phi_x = np.zeros_like(phi_distorted)
    phi_y = np.zeros_like(phi_distorted)
    phi_x[:, 1:-1] = (phi_distorted[:, 2:] - phi_distorted[:, :-2]) / (2*dx)
    phi_y[1:-1, :] = (phi_distorted[2:, :] - phi_distorted[:-2, :]) / (2*dy)
    grad_before = np.sqrt(phi_x**2 + phi_y**2)
    
    phi_reinit = reinitialize(phi_distorted, dx, dy, n_steps=50)
    
    phi_x[:, 1:-1] = (phi_reinit[:, 2:] - phi_reinit[:, :-2]) / (2*dx)
    phi_y[1:-1, :] = (phi_reinit[2:, :] - phi_reinit[:-2, :]) / (2*dy)
    grad_after = np.sqrt(phi_x**2 + phi_y**2)
    
    print(f"  Before: mean |∇φ| = {np.mean(grad_before[5:-5,5:-5]):.3f} "
          f"(should be 1.0)")
    print(f"  After:  mean |∇φ| = {np.mean(grad_after[5:-5,5:-5]):.3f}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Advected circle
        ax = axes[0, 0]
        ax.contour(X, Y, phi, levels=[0], colors='b', linewidths=2)
        ax.contour(X, Y, phi_adv, levels=[0], colors='r', linewidths=2)
        ax.set_title('Circle Advection (blue→red)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Motion by curvature
        ax = axes[0, 1]
        ax.contour(X, Y, phi_bumpy, levels=[0], colors='b', linewidths=2)
        ax.contour(X, Y, phi_smooth, levels=[0], colors='r', linewidths=2)
        ax.set_title('Curvature Flow (blue→red)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Merging
        ax = axes[0, 2]
        ax.contour(X, Y, phi_two, levels=[0], colors='b', linewidths=2)
        ax.contour(X, Y, phi_merged, levels=[0], colors='r', linewidths=2)
        ax.set_title('Merging Circles')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # CSG operations
        for idx, (name, sdf) in enumerate([
            ('Union', union), ('Intersection', inter), ('Difference', diff)
        ]):
            ax = axes[1, idx]
            ax.contourf(X, Y, -sdf, levels=50, cmap='RdBu', alpha=0.5)
            ax.contour(X, Y, sdf, levels=[0], colors='k', linewidths=2)
            ax.contour(X, Y, circ, levels=[0], colors='b', linestyles='--')
            ax.contour(X, Y, rect, levels=[0], colors='g', linestyles='--')
            ax.set_title(f'CSG: {name}')
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig("level_set.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
