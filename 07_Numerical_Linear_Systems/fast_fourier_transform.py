"""
Fast Fourier Transform (FFT)
================================
Efficiently compute the Discrete Fourier Transform (DFT):

    X[k] = Σ_{n=0}^{N-1} x[n] · exp(-2πi k n / N)

Brute-force DFT: O(N²).  FFT (Cooley–Tukey): O(N log N).

The key insight: decompose the DFT of size N into two DFTs of size N/2
by separating even- and odd-indexed elements:

    X[k] = E[k] + W^k · O[k]       (k = 0, ..., N/2-1)
    X[k+N/2] = E[k] - W^k · O[k]

where W = exp(-2πi/N) (twiddle factor), E = DFT of evens, O = DFT of odds.

Applications:
- Spectral analysis of signals
- Solving PDEs with periodic boundary conditions
- Convolution in O(N log N) instead of O(N²)
- Polynomial multiplication
- Image processing
"""

import numpy as np


def dft(x):
    """
    Direct DFT computation — O(N²). For reference/testing.
    
    X[k] = Σ x[n] exp(-2πi kn/N)
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N)
    k = n.reshape(-1, 1)
    W = np.exp(-2j * np.pi * k * n / N)
    return W @ x


def idft(X):
    """Inverse DFT: x[n] = (1/N) Σ X[k] exp(+2πi kn/N)."""
    X = np.asarray(X, dtype=complex)
    N = len(X)
    n = np.arange(N)
    k = n.reshape(-1, 1)
    W = np.exp(2j * np.pi * k * n / N)
    return (W @ X) / N


def fft_recursive(x):
    """
    Cooley-Tukey FFT — recursive radix-2 implementation.
    
    Requires N = power of 2.
    Complexity: O(N log N).
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    
    if N == 1:
        return x.copy()
    
    if N % 2 != 0:
        raise ValueError("N must be a power of 2")
    
    # Recursively compute FFT of even and odd parts
    E = fft_recursive(x[0::2])  # Even indices
    O = fft_recursive(x[1::2])  # Odd indices
    
    # Twiddle factors
    k = np.arange(N // 2)
    W = np.exp(-2j * np.pi * k / N)
    
    # Butterfly combination
    X = np.empty(N, dtype=complex)
    X[:N//2] = E + W * O
    X[N//2:] = E - W * O
    
    return X


def fft_iterative(x):
    """
    Iterative (in-place) Cooley-Tukey FFT.
    
    Uses bit-reversal permutation + butterfly operations.
    More memory-efficient than recursive version.
    """
    x = np.asarray(x, dtype=complex).copy()
    N = len(x)
    
    if N & (N - 1) != 0:
        raise ValueError("N must be a power of 2")
    
    # Bit-reversal permutation
    bits = int(np.log2(N))
    for i in range(N):
        j = int(bin(i)[2:].zfill(bits)[::-1], 2)
        if i < j:
            x[i], x[j] = x[j], x[i]
    
    # Butterfly stages
    stage_size = 2
    while stage_size <= N:
        half = stage_size // 2
        W = np.exp(-2j * np.pi / stage_size)
        
        for start in range(0, N, stage_size):
            w = 1.0
            for k in range(half):
                t = w * x[start + k + half]
                x[start + k + half] = x[start + k] - t
                x[start + k] = x[start + k] + t
                w *= W
        
        stage_size *= 2
    
    return x


def ifft(X):
    """Inverse FFT via FFT: x = conj(FFT(conj(X))) / N."""
    X = np.asarray(X, dtype=complex)
    return np.conj(fft_recursive(np.conj(X))) / len(X)


def fft_convolution(a, b):
    """
    Fast convolution using FFT.
    
    conv(a, b) = IFFT(FFT(a) · FFT(b))
    
    Pad to next power of 2 for efficiency.
    """
    n = len(a) + len(b) - 1
    # Pad to next power of 2
    N = 1
    while N < n:
        N *= 2
    
    A = fft_recursive(np.pad(a, (0, N - len(a))))
    B = fft_recursive(np.pad(b, (0, N - len(b))))
    
    return ifft(A * B)[:n].real


def power_spectrum(x, dt=1.0):
    """
    Compute the power spectral density.
    
    PSD[k] = |X[k]|² / N
    
    Returns frequencies and PSD.
    """
    N = len(x)
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, dt)
    psd = np.abs(X)**2 / N
    
    # Return positive frequencies only
    pos = freqs >= 0
    return freqs[pos], psd[pos]


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FAST FOURIER TRANSFORM (FFT) DEMO")
    print("=" * 60)

    # --- Test 1: Verify FFT = DFT ---
    print("\n--- Test 1: Verify FFT matches DFT ---")
    np.random.seed(42)
    N = 64
    x = np.random.randn(N) + 1j * np.random.randn(N)
    
    X_dft = dft(x)
    X_fft_rec = fft_recursive(x)
    X_fft_iter = fft_iterative(x)
    X_numpy = np.fft.fft(x)
    
    print(f"||FFT_rec  - DFT||  = {np.linalg.norm(X_fft_rec - X_dft):.2e}")
    print(f"||FFT_iter - DFT||  = {np.linalg.norm(X_fft_iter - X_dft):.2e}")
    print(f"||FFT_rec  - numpy|| = {np.linalg.norm(X_fft_rec - X_numpy):.2e}")

    # --- Test 2: Signal analysis ---
    print("\n--- Test 2: Spectral analysis of a signal ---")
    dt = 0.001  # 1 ms sampling
    t = np.arange(0, 1.0, dt)  # 1 second
    N = len(t)
    
    # Signal: 50 Hz + 120 Hz + noise
    signal = 0.7 * np.sin(2*np.pi*50*t) + 1.0 * np.sin(2*np.pi*120*t)
    signal += 0.5 * np.random.randn(N)
    
    freqs, psd = power_spectrum(signal, dt)
    
    # Find peaks
    peak_idx = np.argsort(psd)[-5:]
    peak_freqs = freqs[peak_idx]
    print(f"Top frequency peaks: {sorted(peak_freqs)[:3]} Hz")
    print(f"Expected: 50 Hz and 120 Hz")

    # --- Test 3: FFT convolution ---
    print("\n--- Test 3: FFT convolution ---")
    a = np.array([1, 2, 3, 4])
    b = np.array([1, 1, 1])
    
    conv_fft = fft_convolution(a, b)
    conv_direct = np.convolve(a, b)
    
    print(f"FFT conv:    {np.round(conv_fft, 6)}")
    print(f"Direct conv: {conv_direct}")
    print(f"Match: {np.allclose(conv_fft, conv_direct)}")

    # --- Test 4: Timing comparison ---
    print("\n--- Test 4: Timing ---")
    import time
    
    for N in [64, 256, 1024, 4096]:
        x = np.random.randn(N)
        
        t0 = time.perf_counter()
        for _ in range(10):
            fft_recursive(x)
        t_fft = (time.perf_counter() - t0) / 10
        
        if N <= 1024:
            t0 = time.perf_counter()
            for _ in range(3):
                dft(x)
            t_dft = (time.perf_counter() - t0) / 3
            ratio = t_dft / t_fft
        else:
            t_dft = float('nan')
            ratio = float('nan')
        
        print(f"N={N:5d}: FFT={t_fft*1000:.3f}ms, "
              f"DFT={t_dft*1000:.3f}ms, speedup={ratio:.1f}x")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Time domain signal
        axes[0, 0].plot(t[:200], signal[:200], 'b-', alpha=0.8)
        axes[0, 0].set_xlabel('Time [s]')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Signal: 50 Hz + 120 Hz + noise')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Power spectrum
        axes[0, 1].plot(freqs, psd, 'r-')
        axes[0, 1].set_xlabel('Frequency [Hz]')
        axes[0, 1].set_ylabel('Power')
        axes[0, 1].set_title('Power Spectral Density')
        axes[0, 1].set_xlim(0, 200)
        axes[0, 1].axvline(50, color='g', linestyle='--', alpha=0.5, label='50 Hz')
        axes[0, 1].axvline(120, color='b', linestyle='--', alpha=0.5, label='120 Hz')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Butterfly diagram for N=8
        axes[1, 0].set_title('FFT Butterfly Structure (N=8)')
        axes[1, 0].set_xlim(-0.5, 3.5)
        axes[1, 0].set_ylim(-0.5, 7.5)
        
        # Draw the butterfly connections for N=8
        labels = ['x[0]', 'x[4]', 'x[2]', 'x[6]', 'x[1]', 'x[5]', 'x[3]', 'x[7]']
        for i in range(8):
            axes[1, 0].text(-0.3, i, labels[i], ha='right', va='center', fontsize=8)
            axes[1, 0].text(3.3, i, f'X[{i}]', ha='left', va='center', fontsize=8)
        
        # Stage connections (simplified)
        for stage in range(3):
            block = 2**(stage+1)
            half = block // 2
            for start in range(0, 8, block):
                for k in range(half):
                    y1 = start + k
                    y2 = start + k + half
                    x_pos = stage
                    axes[1, 0].plot([x_pos, x_pos+1], [y1, y1], 'b-', linewidth=0.5)
                    axes[1, 0].plot([x_pos, x_pos+1], [y2, y2], 'b-', linewidth=0.5)
                    axes[1, 0].plot([x_pos, x_pos+1], [y2, y1], 'r--', linewidth=0.5, alpha=0.5)
                    axes[1, 0].plot([x_pos, x_pos+1], [y1, y2], 'r--', linewidth=0.5, alpha=0.5)
        
        axes[1, 0].set_xlabel('Stage')
        axes[1, 0].set_ylabel('Index')
        axes[1, 0].grid(True, alpha=0.2)
        
        # IFFT reconstruction
        X = np.fft.fft(signal)
        # Zero out small components
        X_filtered = X.copy()
        threshold = 0.1 * np.max(np.abs(X))
        X_filtered[np.abs(X_filtered) < threshold] = 0
        signal_filtered = np.fft.ifft(X_filtered).real
        
        axes[1, 1].plot(t[:200], signal[:200], 'b-', alpha=0.4, label='Original')
        axes[1, 1].plot(t[:200], signal_filtered[:200], 'r-', linewidth=1.5, label='Filtered')
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].set_title('FFT Filtering (remove noise)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("fft.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
