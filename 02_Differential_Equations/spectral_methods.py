"""
Spectral Methods
=================
Approximate PDE solutions using global basis functions (Fourier modes,
Chebyshev polynomials) rather than local finite differences.

Key idea: Represent the solution as a sum of basis functions and determine
the coefficients by requiring the PDE to be satisfied exactly at collocation
points or in a Galerkin sense.

Advantages:
- Exponential convergence for smooth problems (vs. polynomial for FDM/FEM).
- Very efficient with FFT for periodic domains.
- Excellent for turbulence simulations, quantum mechanics.

Types demonstrated:
1. Fourier spectral method (periodic BCs)
2. Chebyshev collocation (non-periodic BCs)
"""

import numpy as np


def fourier_spectral_poisson(f_values, L):
    """
    Solve -u''(x) = f(x) on [0, L] with periodic BCs using Fourier spectral method.
    
    Method:
        1. Take FFT of f → f̂_k
        2. In Fourier space: -(-ik)² û_k = f̂_k  →  û_k = f̂_k / k²
        3. Take inverse FFT to get u
    
    Parameters
    ----------
    f_values : ndarray – f evaluated at uniform grid points
    L : float – domain length
    
    Returns
    -------
    u : ndarray – solution values at grid points
    """
    N = len(f_values)
    # Wavenumbers
    k = np.fft.fftfreq(N, d=L/N) * 2 * np.pi

    # FFT of source term
    f_hat = np.fft.fft(f_values)

    # Solve in Fourier space: û_k = f̂_k / k²
    u_hat = np.zeros(N, dtype=complex)
    for i in range(N):
        if abs(k[i]) > 1e-10:       # Skip k=0 (mean value)
            u_hat[i] = f_hat[i] / k[i]**2

    # Inverse FFT
    u = np.real(np.fft.ifft(u_hat))
    return u


def fourier_spectral_derivative(u, L, order=1):
    """
    Compute the n-th derivative of u using Fourier spectral differentiation.
    
    d^n u / dx^n  ↔  (ik)^n û_k
    
    Spectral accuracy for smooth periodic functions!
    """
    N = len(u)
    k = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    u_hat = np.fft.fft(u)
    deriv_hat = (1j * k) ** order * u_hat
    return np.real(np.fft.ifft(deriv_hat))


def chebyshev_differentiation_matrix(N):
    """
    Compute the Chebyshev differentiation matrix D on N+1 Chebyshev points.
    
    The Chebyshev points are: x_j = cos(jπ/N), j = 0, 1, ..., N
    These cluster near the boundaries — excellent for non-periodic problems.
    
    Returns
    -------
    D : ndarray, shape (N+1, N+1) – differentiation matrix
    x : ndarray – Chebyshev points
    """
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
    c *= (-1.0) ** np.arange(N + 1)

    X = np.outer(x, np.ones(N + 1))
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D -= np.diag(np.sum(D, axis=1))

    return D, x


def chebyshev_solve_bvp(f_func, N, bc_left=0.0, bc_right=0.0):
    """
    Solve -u''(x) = f(x) on [-1, 1] using Chebyshev collocation.
    
    u(-1) = bc_left, u(1) = bc_right.
    """
    D, x = chebyshev_differentiation_matrix(N)
    D2 = D @ D  # Second derivative matrix

    # The equation at interior points: -D2 @ u = f
    # Apply BCs: u[0] = bc_right (x=1), u[N] = bc_left (x=-1)

    # Modify system for BCs
    A = -D2.copy()
    rhs = f_func(x)

    # First and last rows enforce BCs
    A[0, :] = 0;   A[0, 0] = 1;   rhs[0] = bc_right
    A[N, :] = 0;   A[N, N] = 1;   rhs[N] = bc_left

    u = np.linalg.solve(A, rhs)
    return x, u


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SPECTRAL METHODS DEMO")
    print("=" * 60)

    # --- Fourier spectral: periodic Poisson ---
    # TO TEST: Modify domain length L, number of modes N, and the source term f(x).
    # Parameters: L=2π (domain period), N=64 (modes), source f(x)=sin(x)+4sin(2x).
    # Initial values: Try L=2, add N=128 for finer resolution, modify forcing terms.
    # Observe: Max error should be ~1e-12 or better (spectral accuracy!).
    # Insight: Fourier is best for smooth periodic functions. Try f(x)=exp(sin(x)) for smoothness test.
    print("\n--- Fourier Spectral: -u'' = f(x), periodic ---")
    L = 2 * np.pi
    N = 64
    x = np.linspace(0, L, N, endpoint=False)
    
    # Source: f(x) = sin(x) + 4sin(2x)
    # Exact: u(x) = sin(x) + sin(2x)
    f_values = np.sin(x) + 4 * np.sin(2 * x)
    u_exact = np.sin(x) + np.sin(2 * x)

    u = fourier_spectral_poisson(f_values, L)
    u -= np.mean(u) - np.mean(u_exact)  # Fix arbitrary constant
    error = np.max(np.abs(u - u_exact))
    print(f"  Max error (N={N}): {error:.2e}")

    # Spectral derivative
    # TO TEST: Try different orders (order=2,3,4) and different functions.
    # Parameters: order (1,2,3,...), function g(x), domain L.
    # Initial values: g=sin(3x), order=1. Try g=exp(sin(x)), order=2.
    # Observe: Error magnitude (should be very small for smooth functions).
    # Insight: Even order=4 derivatives are accurate due to FFT spectral accuracy.
    print("\n--- Spectral Derivative ---")
    g = np.sin(3 * x)
    g_prime_exact = 3 * np.cos(3 * x)
    g_prime = fourier_spectral_derivative(g, L, order=1)
    print(f"  d/dx[sin(3x)] error: {np.max(np.abs(g_prime - g_prime_exact)):.2e}")

    # --- Chebyshev: non-periodic BVP ---
    # TO TEST: Vary grid resolution N, change boundary values bc_left/bc_right, or modify forcing f_cheb.
    # Parameters: N (Chebyshev nodes), f_cheb (RHS function), bc_left=0, bc_right=0 (Dirichlet BCs).
    # Initial values: N=[8,16,32,64], f=-π²sin(πx), exact=sin(πx).
    # Observe: Max error decreases exponentially (spectral convergence) with N.
    # Try: Change exact solution (u=exp(-x), u=x⁴) or add non-homogeneous BCs.
    print("\n--- Chebyshev Collocation: -u'' = π²sin(πx) ---")
    f_cheb = lambda x: np.pi**2 * np.sin(np.pi * x)
    u_cheb_exact = lambda x: np.sin(np.pi * x)

    for N in [8, 16, 32, 64]:
        x_c, u_c = chebyshev_solve_bvp(f_cheb, N)
        error = np.max(np.abs(u_c - u_cheb_exact(x_c)))
        print(f"  N = {N:3d}  |  Max error = {error:.4e}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Fourier solution
        axes[0].plot(x, u, 'b-', label='Spectral')
        axes[0].plot(x, u_exact, 'r--', label='Exact')
        axes[0].set_title("Fourier Spectral (Periodic)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Chebyshev solution
        x_c, u_c = chebyshev_solve_bvp(f_cheb, 16)
        axes[1].plot(x_c, u_c, 'bo-', label=f'Chebyshev N=16')
        x_fine = np.linspace(-1, 1, 200)
        axes[1].plot(x_fine, u_cheb_exact(x_fine), 'r-', label='Exact')
        axes[1].set_title("Chebyshev Collocation")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Convergence comparison
        Ns = [4, 8, 12, 16, 24, 32, 48, 64]
        errors = []
        for N in Ns:
            x_c, u_c = chebyshev_solve_bvp(f_cheb, N)
            errors.append(np.max(np.abs(u_c - u_cheb_exact(x_c))))
        axes[2].semilogy(Ns, errors, 'bo-', label='Chebyshev error')
        axes[2].set_xlabel('N')
        axes[2].set_ylabel('Max error')
        axes[2].set_title('Exponential Convergence')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("spectral_methods.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
