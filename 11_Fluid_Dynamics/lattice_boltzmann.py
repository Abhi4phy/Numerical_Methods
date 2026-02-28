"""
Lattice Boltzmann Method (LBM)
================================
A mesoscopic approach to fluid dynamics that simulates fluid flow
by tracking the statistical distribution of fictitious particles
on a lattice.

**Instead of solving Navier-Stokes directly**, LBM evolves a
particle distribution function f(x, v, t) via two simple steps:

1. **Collision** (local relaxation):
   f_i → f_i - (f_i - f_i^eq) / τ   (BGK approximation)

2. **Streaming** (propagation):
   f_i(x + c_i·Δt, t+Δt) = f_i(x, t)   (shift along lattice links)

**Why LBM?**
- Inherently parallel (local operations)
- Easy to handle complex geometries (bounce-back BC)
- Naturally captures rarefied / multiphase flows
- Second-order accurate, recovers Navier-Stokes in the limit

**D2Q9 Lattice (2D, 9 velocities):**
    c₀ = (0,0)                         — rest
    c₁₋₄ = (±1,0), (0,±1)            — axial
    c₅₋₈ = (±1,±1)                    — diagonal

**BGK Collision Operator:**
The equilibrium distribution:
    f_i^eq = w_i · ρ · [1 + (c_i·u)/c_s² + (c_i·u)²/(2c_s⁴) - u²/(2c_s²)]

where c_s = 1/√3 (lattice speed of sound), w_i are weights.

Kinematic viscosity: ν = c_s²(τ - 1/2)Δt

Physics: incompressible flow, porous media, multiphase, thermal flows.

Where to start:
━━━━━━━━━━━━━━
Run the lid-driven cavity example. Watch vortices form.
Prerequisite: finite_difference_method.py (understand Navier-Stokes first)
"""

import numpy as np


class LatticeBoltzmann2D:
    """
    D2Q9 Lattice Boltzmann solver with BGK collision.
    
    The D2Q9 model uses 9 velocity directions on a 2D square lattice.
    """
    
    # D2Q9 lattice velocities
    cx = np.array([0, 1, 0, -1,  0, 1, -1, -1,  1])
    cy = np.array([0, 0, 1,  0, -1, 1,  1, -1, -1])
    
    # Weights
    w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    
    # Opposite direction indices (for bounce-back)
    opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
    
    # Speed of sound squared
    cs2 = 1.0 / 3.0
    
    def __init__(self, Nx, Ny, tau):
        """
        Parameters
        ----------
        Nx, Ny : int
            Lattice dimensions
        tau : float
            Relaxation time (must be > 0.5)
            viscosity = cs²(τ - 0.5)
        """
        self.Nx = Nx
        self.Ny = Ny
        self.tau = tau
        self.nu = self.cs2 * (tau - 0.5)  # kinematic viscosity
        
        # Distribution functions: f[q, y, x]
        self.f = np.zeros((9, Ny, Nx))
        
        # Macroscopic fields
        self.rho = np.ones((Ny, Nx))
        self.ux = np.zeros((Ny, Nx))
        self.uy = np.zeros((Ny, Nx))
        
        # Obstacle mask (True = solid)
        self.obstacle = np.zeros((Ny, Nx), dtype=bool)
        
        # Initialize to equilibrium
        self._init_equilibrium()
    
    def _init_equilibrium(self):
        """Initialize f to equilibrium state."""
        self.f = self.equilibrium(self.rho, self.ux, self.uy)
    
    def equilibrium(self, rho, ux, uy):
        """
        Compute equilibrium distribution:
        f_i^eq = w_i ρ [1 + (c_i·u)/c_s² + (c_i·u)²/(2c_s⁴) - u²/(2c_s²)]
        """
        feq = np.zeros((9, self.Ny, self.Nx))
        
        u_sq = ux**2 + uy**2
        
        for i in range(9):
            cu = self.cx[i] * ux + self.cy[i] * uy
            feq[i] = self.w[i] * rho * (
                1.0 
                + cu / self.cs2 
                + cu**2 / (2 * self.cs2**2)
                - u_sq / (2 * self.cs2)
            )
        
        return feq
    
    def compute_macroscopic(self):
        """Compute density and velocity from distribution functions."""
        self.rho = np.sum(self.f, axis=0)
        self.ux = np.sum(self.f * self.cx.reshape(9,1,1), axis=0) / self.rho
        self.uy = np.sum(self.f * self.cy.reshape(9,1,1), axis=0) / self.rho
    
    def collide(self):
        """BGK collision step: f → f - (f - feq) / τ"""
        feq = self.equilibrium(self.rho, self.ux, self.uy)
        self.f -= (self.f - feq) / self.tau
    
    def stream(self):
        """Streaming step: move distributions along lattice links."""
        f_new = np.zeros_like(self.f)
        
        for i in range(9):
            f_new[i] = np.roll(np.roll(self.f[i], self.cx[i], axis=1), 
                               self.cy[i], axis=0)
        
        self.f = f_new
    
    def bounce_back(self):
        """No-slip boundary condition at obstacle nodes."""
        for i in range(9):
            # At obstacle nodes, reflect to opposite direction
            self.f[i][self.obstacle] = self.f[self.opp[i]][self.obstacle]
    
    def step(self):
        """One LBM time step: collide → stream → boundary."""
        self.collide()
        self.stream()
        self.bounce_back()
        self.compute_macroscopic()
    
    def set_lid_velocity(self, u_lid):
        """
        Set moving lid boundary condition (top wall).
        
        Zou-He boundary condition for the moving top wall.
        """
        # Top wall: y = Ny-1
        rho_wall = (self.f[0, -1, :] + self.f[1, -1, :] + self.f[3, -1, :] 
                   + 2 * (self.f[2, -1, :] + self.f[5, -1, :] + self.f[6, -1, :]))
        
        self.f[4, -1, :] = self.f[2, -1, :]
        self.f[7, -1, :] = self.f[5, -1, :] - 0.5 * rho_wall * u_lid
        self.f[8, -1, :] = self.f[6, -1, :] + 0.5 * rho_wall * u_lid
    
    def set_walls(self, walls='all'):
        """Set bounce-back on domain boundaries."""
        if walls in ('all', 'bottom'):
            self.obstacle[0, :] = True
        if walls in ('all', 'top'):
            self.obstacle[-1, :] = True
        if walls in ('all', 'left'):
            self.obstacle[:, 0] = True
        if walls in ('all', 'right'):
            self.obstacle[:, -1] = True
    
    def add_cylinder(self, cx, cy, r):
        """Add a circular obstacle."""
        X, Y = np.meshgrid(np.arange(self.Nx), np.arange(self.Ny))
        self.obstacle |= ((X - cx)**2 + (Y - cy)**2 < r**2)
    
    def vorticity(self):
        """Compute vorticity ω = ∂uy/∂x - ∂ux/∂y."""
        return (np.roll(self.uy, -1, axis=1) - np.roll(self.uy, 1, axis=1) 
               - np.roll(self.ux, -1, axis=0) + np.roll(self.ux, 1, axis=0))
    
    def speed(self):
        """Compute velocity magnitude."""
        return np.sqrt(self.ux**2 + self.uy**2)


def lid_driven_cavity(Nx, Ny, Re, n_steps, u_lid=0.1, print_every=1000):
    """
    Classic lid-driven cavity flow.
    
    Parameters
    ----------
    Nx, Ny : int
        Grid size
    Re : float
        Reynolds number = u_lid * Nx / ν
    n_steps : int
        Number of LBM steps
    u_lid : float
        Lid velocity (in lattice units, keep < 0.1 for stability)
    """
    # Compute tau from Reynolds number
    nu = u_lid * Nx / Re
    tau = nu / (1.0/3.0) + 0.5
    
    print(f"  Re = {Re}, tau = {tau:.4f}, nu = {nu:.6f}")
    
    lbm = LatticeBoltzmann2D(Nx, Ny, tau)
    
    # Set walls (all four, but top will be overridden by lid)
    lbm.obstacle[0, :] = True      # bottom
    lbm.obstacle[:, 0] = True      # left
    lbm.obstacle[:, -1] = True     # right
    # Top is moving lid, not obstacle
    
    for step in range(n_steps):
        lbm.collide()
        lbm.stream()
        lbm.bounce_back()
        
        # Apply lid BC (top wall moving right)
        lbm.set_lid_velocity(u_lid)
        
        lbm.compute_macroscopic()
        
        if (step + 1) % print_every == 0:
            max_speed = lbm.speed().max()
            print(f"    Step {step+1}/{n_steps}, max|u| = {max_speed:.6f}")
    
    return lbm


def flow_around_cylinder(Nx, Ny, Re, n_steps, cyl_r=None, print_every=2000):
    """
    Flow around a circular cylinder.
    Demonstrates vortex shedding (von Kármán street) at Re > ~40.
    """
    u_in = 0.04
    nu = u_in * (2 * (cyl_r or Ny//10)) / Re
    tau = nu / (1.0/3.0) + 0.5
    
    if cyl_r is None:
        cyl_r = Ny // 10
    
    print(f"  Re = {Re}, tau = {tau:.4f}, cylinder radius = {cyl_r}")
    
    lbm = LatticeBoltzmann2D(Nx, Ny, tau)
    
    # Add cylinder
    cx, cy = Nx // 4, Ny // 2
    lbm.add_cylinder(cx, cy, cyl_r)
    
    # Initialize with uniform flow
    lbm.ux[:, :] = u_in
    lbm._init_equilibrium()
    
    for step in range(n_steps):
        lbm.collide()
        lbm.stream()
        lbm.bounce_back()
        
        # Inlet: set inflow velocity
        lbm.ux[:, 0] = u_in
        lbm.uy[:, 0] = 0.0
        rho_in = 1.0
        lbm.f[:, :, 0] = lbm.equilibrium(
            rho_in * np.ones((Ny, 1)), 
            u_in * np.ones((Ny, 1)), 
            np.zeros((Ny, 1)))[:, :, 0]
        
        # Top/bottom: no-slip
        lbm.f[:, 0, :] = lbm.f[lbm.opp, 0, :]
        lbm.f[:, -1, :] = lbm.f[lbm.opp, -1, :]
        
        lbm.compute_macroscopic()
        
        # Fix obstacle velocity to zero
        lbm.ux[lbm.obstacle] = 0
        lbm.uy[lbm.obstacle] = 0
        
        if (step + 1) % print_every == 0:
            print(f"    Step {step+1}/{n_steps}, max|u| = {lbm.speed().max():.6f}")
    
    return lbm


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LATTICE BOLTZMANN METHOD DEMO")
    print("=" * 60)

    # --- 1. Lid-driven cavity ---
    print("\n--- Lid-Driven Cavity (Re=100) ---")
    Nx, Ny = 100, 100
    lbm_cavity = lid_driven_cavity(Nx, Ny, Re=100, n_steps=10000, 
                                    u_lid=0.1, print_every=2000)
    
    # Centerline velocity profiles
    ux_center = lbm_cavity.ux[:, Nx//2]
    uy_center = lbm_cavity.uy[Ny//2, :]
    
    print(f"  Max velocity: {lbm_cavity.speed().max():.6f}")
    print(f"  Center vortex location: check plot")

    # --- 2. Poiseuille flow (validation) ---
    print("\n--- Poiseuille Flow (analytic solution) ---")
    Nx_p, Ny_p = 50, 30
    tau_p = 0.8
    nu_p = (1.0/3.0) * (tau_p - 0.5)
    
    # Body force (pressure gradient)
    dpdx = 1e-5
    
    lbm_pois = LatticeBoltzmann2D(Nx_p, Ny_p, tau_p)
    lbm_pois.obstacle[0, :] = True   # Bottom wall
    lbm_pois.obstacle[-1, :] = True  # Top wall
    
    for step in range(5000):
        lbm_pois.collide()
        
        # Add body force
        for i in range(9):
            lbm_pois.f[i] += lbm_pois.w[i] * dpdx * lbm_pois.cx[i] / lbm_pois.cs2
        
        lbm_pois.stream()
        lbm_pois.bounce_back()
        lbm_pois.compute_macroscopic()
    
    # Analytic: u(y) = (dpdx / 2ν) · y · (H - y)
    H = Ny_p - 2  # Channel height (excluding walls)
    y = np.arange(Ny_p)
    u_exact = (dpdx / (2 * nu_p)) * (y - 0.5) * (H + 0.5 - y)
    u_exact[0] = 0
    u_exact[-1] = 0
    
    ux_profile = lbm_pois.ux[:, Nx_p // 2]
    rel_err = np.max(np.abs(ux_profile[1:-1] - u_exact[1:-1])) / np.max(u_exact)
    print(f"  Channel flow, tau = {tau_p}")
    print(f"  Max velocity: exact = {u_exact.max():.6f}, LBM = {ux_profile.max():.6f}")
    print(f"  Relative error: {rel_err:.4e}")

    # --- 3. Flow around cylinder ---
    print("\n--- Flow Around Cylinder (Re=40) ---")
    lbm_cyl = flow_around_cylinder(200, 80, Re=40, n_steps=5000, 
                                    cyl_r=8, print_every=2000)

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Cavity: velocity magnitude
        ax = axes[0, 0]
        speed = lbm_cavity.speed()
        c = ax.imshow(speed, cmap='hot', origin='lower')
        plt.colorbar(c, ax=ax)
        ax.set_title('Cavity: Speed |u|')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Cavity: streamlines
        ax = axes[0, 1]
        Y, X = np.mgrid[0:Ny, 0:Nx]
        ax.streamplot(X, Y, lbm_cavity.ux, lbm_cavity.uy, 
                     color=speed, cmap='hot', density=2, linewidth=0.5)
        ax.set_title('Cavity: Streamlines')
        ax.set_aspect('equal')
        ax.set_xlim(0, Nx)
        ax.set_ylim(0, Ny)
        
        # Cavity centerline profiles
        ax = axes[0, 2]
        y_norm = np.linspace(0, 1, Ny)
        ax.plot(ux_center / 0.1, y_norm, 'b-', label='ux (x=L/2)')
        x_norm = np.linspace(0, 1, Nx)
        ax.plot(x_norm, uy_center / 0.1 + 0.5, 'r-', label='uy (y=L/2) shifted')
        ax.set_xlabel('Normalized velocity')
        ax.set_ylabel('y/L or x/L')
        ax.set_title('Centerline Velocities')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Poiseuille validation
        ax = axes[1, 0]
        ax.plot(y, u_exact, 'r-', linewidth=2, label='Exact')
        ax.plot(y, ux_profile, 'bo', markersize=4, label='LBM')
        ax.set_xlabel('y')
        ax.set_ylabel('u_x')
        ax.set_title('Poiseuille: LBM vs Exact')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cylinder: velocity
        ax = axes[1, 1]
        speed_cyl = lbm_cyl.speed()
        speed_cyl[lbm_cyl.obstacle] = np.nan
        c2 = ax.imshow(speed_cyl, cmap='viridis', origin='lower')
        plt.colorbar(c2, ax=ax)
        ax.set_title('Cylinder: Speed')
        
        # Cylinder: vorticity
        ax = axes[1, 2]
        vort = lbm_cyl.vorticity()
        vort[lbm_cyl.obstacle] = np.nan
        vmax = np.nanpercentile(np.abs(vort), 98)
        c3 = ax.imshow(vort, cmap='RdBu_r', origin='lower', 
                       vmin=-vmax, vmax=vmax)
        plt.colorbar(c3, ax=ax)
        ax.set_title('Cylinder: Vorticity')
        
        plt.tight_layout()
        plt.savefig("lattice_boltzmann.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
