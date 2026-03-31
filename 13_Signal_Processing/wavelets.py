"""
Wavelet Transform
==================
Multi-resolution analysis — decompose signals into frequency
components localized in BOTH time and frequency.

**Fourier vs Wavelets:**
- FFT: perfect frequency resolution, NO time resolution
- STFT: fixed time-frequency tradeoff (fixed window)
- Wavelets: ADAPTIVE — narrow in time for high freq, wide for low freq

**The Continuous Wavelet Transform:**
    W(a, b) = ∫ f(t) · ψ*((t-b)/a) dt / √a

where ψ is the "mother wavelet", a is scale, b is translation.

**The Discrete Wavelet Transform (DWT):**
Uses dyadic scales a = 2ʲ and translations b = k·2ʲ.

Efficient O(N) implementation via filter banks:
    1. Apply low-pass filter h → approximation coefficients (cA)
    2. Apply high-pass filter g → detail coefficients (cD)
    3. Downsample by 2
    4. Repeat on cA for multi-level decomposition

**Reconstruction (inverse DWT):**
    Upsample → filter → add: x = h'*cA + g'*cD

**Common Wavelets:**
- Haar: simplest, discontinuous
- Daubechies (dbN): compact support, N vanishing moments
- Symlets: near-symmetric Daubechies
- Mexican hat: Ricker wavelet (2nd derivative of Gaussian)

**Applications:**
- Signal denoising (threshold detail coefficients)
- Image compression (JPEG 2000 uses wavelets)
- Turbulence analysis (intermittency detection)
- Gravitational wave detection
- Quantum chemistry (wavelet bases)

Where to start:
━━━━━━━━━━━━━━
Run the Haar decomposition on a simple signal.
Then try denoising — it's magical how thresholding wavelet
coefficients removes noise while preserving edges.
Prerequisite: fast_fourier_transform.py
"""

import numpy as np


# ============================================================
# Wavelet Filter Banks
# ============================================================

def haar_filters():
    """Haar wavelet filters — simplest wavelet."""
    h = np.array([1, 1]) / np.sqrt(2)        # Low-pass
    g = np.array([1, -1]) / np.sqrt(2)        # High-pass
    h_rec = np.array([1, 1]) / np.sqrt(2)     # Reconstruction low-pass
    g_rec = np.array([-1, 1]) / np.sqrt(2)    # Reconstruction high-pass
    return h, g, h_rec, g_rec


def db4_filters():
    """Daubechies-4 wavelet filters (4 coefficients)."""
    h = np.array([
        (1 + np.sqrt(3)) / (4 * np.sqrt(2)),
        (3 + np.sqrt(3)) / (4 * np.sqrt(2)),
        (3 - np.sqrt(3)) / (4 * np.sqrt(2)),
        (1 - np.sqrt(3)) / (4 * np.sqrt(2)),
    ])
    g = np.array([h[3], -h[2], h[1], -h[0]])
    h_rec = h[::-1]
    g_rec = np.array([-h[3], h[2], -h[1], h[0]])
    return h, g, h_rec, g_rec


def db6_filters():
    """Daubechies-6 wavelet filters (6 coefficients)."""
    h = np.array([
        0.3326705529500826,
        0.8068915093110928,
        0.4598775021184915,
        -0.1350110200102546,
        -0.0854412738820267,
        0.0352262918857095,
    ])
    g = np.array([h[5], -h[4], h[3], -h[2], h[1], -h[0]])
    h_rec = h[::-1]
    g_rec = g[::-1]
    return h, g, h_rec, g_rec


# ============================================================
# Discrete Wavelet Transform
# ============================================================

def dwt_1d(signal, h, g):
    """
    One level of the Discrete Wavelet Transform.
    
    1. Convolve with low-pass (h) and high-pass (g) filters
    2. Downsample by 2
    
    Parameters
    ----------
    signal : array
        Input signal (length should be even)
    h : array
        Low-pass filter
    g : array
        High-pass filter
    
    Returns
    -------
    cA : array
        Approximation coefficients (low frequency)
    cD : array
        Detail coefficients (high frequency)
    """
    N = len(signal)
    L = len(h)
    
    # Periodic convolution + downsampling
    cA = np.zeros(N // 2)
    cD = np.zeros(N // 2)
    
    for i in range(N // 2):
        for j in range(L):
            idx = (2 * i + j) % N
            cA[i] += h[j] * signal[idx]
            cD[i] += g[j] * signal[idx]
    
    return cA, cD


def idwt_1d(cA, cD, h_rec, g_rec):
    """
    Inverse DWT — one level of reconstruction.
    
    1. Upsample by 2 (insert zeros)
    2. Convolve with reconstruction filters
    3. Add results
    """
    N = len(cA) * 2
    L = len(h_rec)
    
    # Upsample
    up_cA = np.zeros(N)
    up_cD = np.zeros(N)
    up_cA[::2] = cA
    up_cD[::2] = cD
    
    # Convolve and sum
    signal = np.zeros(N)
    
    for i in range(N):
        for j in range(L):
            idx = (i - j) % N
            signal[i] += h_rec[j] * up_cA[idx] + g_rec[j] * up_cD[idx]
    
    return signal


def wavedec(signal, wavelet='haar', level=None):
    """
    Multi-level wavelet decomposition.
    
    Parameters
    ----------
    signal : array
        Input signal (length should be power of 2)
    wavelet : str
        'haar', 'db4', or 'db6'
    level : int, optional
        Number of decomposition levels (default: max possible)
    
    Returns
    -------
    coeffs : list
        [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        where n is the deepest level
    """
    filters = {
        'haar': haar_filters,
        'db4': db4_filters,
        'db6': db6_filters,
    }
    
    h, g, _, _ = filters[wavelet]()
    
    N = len(signal)
    if level is None:
        level = int(np.log2(N)) - 1
    
    coeffs = []
    current = signal.copy()
    
    for l in range(level):
        if len(current) < len(h):
            break
        cA, cD = dwt_1d(current, h, g)
        coeffs.append(cD)
        current = cA
    
    coeffs.append(current)  # Final approximation
    coeffs.reverse()  # [cA, cD_deepest, ..., cD_shallowest]
    
    return coeffs


def waverec(coeffs, wavelet='haar'):
    """
    Multi-level wavelet reconstruction.
    
    Parameters
    ----------
    coeffs : list
        Output from wavedec: [cA_n, cD_n, ..., cD_1]
    """
    filters = {
        'haar': haar_filters,
        'db4': db4_filters,
        'db6': db6_filters,
    }
    
    _, _, h_rec, g_rec = filters[wavelet]()
    
    current = coeffs[0]
    
    for cD in coeffs[1:]:
        current = idwt_1d(current, cD, h_rec, g_rec)
    
    return current


# ============================================================
# Wavelet Applications
# ============================================================

def wavelet_denoise(signal, wavelet='haar', threshold=None, mode='soft'):
    """
    Wavelet denoising via coefficient thresholding.
    
    1. Decompose signal into wavelet coefficients
    2. Threshold the detail coefficients
    3. Reconstruct
    
    Parameters
    ----------
    signal : array
        Noisy signal
    threshold : float, optional
        Threshold value (default: universal threshold σ√(2 log N))
    mode : str
        'soft' or 'hard' thresholding
    """
    coeffs = wavedec(signal, wavelet)
    
    if threshold is None:
        # Universal threshold (VisuShrink)
        # Estimate noise from finest level detail coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # Threshold detail coefficients (keep approximation)
    thresholded = [coeffs[0]]  # Keep approximation unchanged
    
    for cD in coeffs[1:]:
        if mode == 'soft':
            cD_thresh = np.sign(cD) * np.maximum(np.abs(cD) - threshold, 0)
        elif mode == 'hard':
            cD_thresh = cD * (np.abs(cD) > threshold)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        thresholded.append(cD_thresh)
    
    return waverec(thresholded, wavelet)


def wavelet_compress(signal, wavelet='haar', keep_fraction=0.1):
    """
    Wavelet compression — keep only the largest coefficients.
    
    Parameters
    ----------
    keep_fraction : float
        Fraction of coefficients to keep (0.1 = 10%)
    """
    coeffs = wavedec(signal, wavelet)
    
    # Flatten all coefficients
    all_coeffs = np.concatenate([c for c in coeffs])
    n_total = len(all_coeffs)
    n_keep = max(1, int(keep_fraction * n_total))
    
    # Find threshold
    sorted_abs = np.sort(np.abs(all_coeffs))[::-1]
    threshold = sorted_abs[n_keep - 1]
    
    # Zero out small coefficients
    compressed = []
    n_nonzero = 0
    for c in coeffs:
        c_comp = c * (np.abs(c) >= threshold)
        compressed.append(c_comp)
        n_nonzero += np.sum(c_comp != 0)
    
    reconstructed = waverec(compressed, wavelet)
    
    return reconstructed, n_nonzero, n_total


def continuous_wavelet_transform(signal, scales, wavelet='morlet', omega0=6.0):
    """
    Continuous Wavelet Transform (CWT).
    
    Parameters
    ----------
    signal : array
        Input signal
    scales : array
        Wavelet scales to evaluate
    wavelet : str
        'morlet' or 'mexican_hat'
    omega0 : float
        Central frequency for Morlet wavelet
    
    Returns
    -------
    cwt_matrix : array (n_scales, N)
        CWT coefficients
    """
    N = len(signal)
    t = np.arange(N)
    cwt_matrix = np.zeros((len(scales), N))
    
    for i, scale in enumerate(scales):
        # Generate wavelet at this scale
        t_wavelet = (t - N/2) / scale
        
        if wavelet == 'morlet':
            psi = (np.pi**(-0.25) * np.exp(1j * omega0 * t_wavelet) 
                   * np.exp(-t_wavelet**2 / 2))
            psi = np.real(psi)
        elif wavelet == 'mexican_hat':
            psi = (2 / (np.sqrt(3) * np.pi**0.25)) * (1 - t_wavelet**2) * np.exp(-t_wavelet**2 / 2)
        
        psi /= np.sqrt(scale)
        
        # Convolution via FFT
        cwt_matrix[i] = np.real(np.fft.ifft(
            np.fft.fft(signal) * np.conj(np.fft.fft(psi))))
    
    return cwt_matrix


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("WAVELET TRANSFORM DEMO")
    print("=" * 60)

    # TO TEST: Change signal length N, decomposition level, and wavelet family.
    # Observe coefficient energy distribution and reconstruction error behavior.
    # --- 1. Haar decomposition ---
    print("\n--- Haar Wavelet Decomposition ---")
    N = 256
    t = np.linspace(0, 1, N, endpoint=False)
    
    # Test signal: step + sine
    signal = np.zeros(N)
    signal[N//4:3*N//4] = 1.0
    signal += 0.3 * np.sin(2 * np.pi * 8 * t)
    
    coeffs = wavedec(signal, 'haar', level=4)
    
    print(f"  Signal length: {N}")
    print(f"  Decomposition levels: {len(coeffs) - 1}")
    for i, c in enumerate(coeffs):
        label = "cA" if i == 0 else f"cD{len(coeffs)-i}"
        print(f"    {label}: {len(c)} coefficients, "
              f"energy = {np.sum(c**2):.4f}")
    
    # Reconstruct
    reconstructed = waverec(coeffs, 'haar')
    err = np.max(np.abs(reconstructed - signal))
    print(f"  Reconstruction error: {err:.2e}")

    # TO TEST: Sweep noise_level, threshold mode (soft/hard), and wavelet type.
    # Observe SNR improvement and edge preservation versus oversmoothing artifacts.
    # --- 2. Denoising ---
    print("\n--- Wavelet Denoising ---")
    rng = np.random.default_rng(42)
    
    # Clean signal
    clean = np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 7 * t)
    clean[N//3:2*N//3] += 2.0  # Add a step
    
    # Add noise
    noise_level = 0.5
    noisy = clean + noise_level * rng.standard_normal(N)
    
    # Denoise
    for wavelet in ['haar', 'db4']:
        denoised = wavelet_denoise(noisy, wavelet=wavelet, mode='soft')
        snr_noisy = 10 * np.log10(np.sum(clean**2) / np.sum((noisy - clean)**2))
        snr_denoised = 10 * np.log10(np.sum(clean**2) / np.sum((denoised - clean)**2))
        print(f"  {wavelet}: SNR noisy = {snr_noisy:.1f} dB → "
              f"denoised = {snr_denoised:.1f} dB")

    # TO TEST: Vary keep_fraction aggressively (e.g., 0.8 to 0.02) and compare wavelets.
    # Observe RMSE growth relative to retained-coefficient ratio.
    # --- 3. Compression ---
    print("\n--- Wavelet Compression ---")
    for keep in [0.5, 0.2, 0.1, 0.05]:
        compressed, n_kept, n_total = wavelet_compress(clean, 'haar', keep)
        rmse = np.sqrt(np.mean((compressed - clean)**2))
        print(f"  Keep {keep*100:5.1f}%: {n_kept}/{n_total} coeffs, "
              f"RMSE = {rmse:.4f}")

    # TO TEST: Modify frequency components and time-localized bursts in signal_mr.
    # Observe which detail levels capture each scale-localized feature.
    # --- 4. Multi-resolution ---
    print("\n--- Multi-Resolution Analysis ---")
    # Signal with features at different scales
    signal_mr = (np.sin(2 * np.pi * 2 * t) 
                + 0.5 * np.sin(2 * np.pi * 16 * t) * (t > 0.5)
                + 0.3 * np.sin(2 * np.pi * 64 * t) * ((t > 0.2) & (t < 0.4)))
    
    coeffs_mr = wavedec(signal_mr, 'haar', level=6)
    print(f"  6-level decomposition of multi-scale signal:")
    for i, c in enumerate(coeffs_mr):
        label = "Approx" if i == 0 else f"Detail {len(coeffs_mr)-i}"
        energy = np.sum(c**2)
        print(f"    {label:>10}: {len(c):4d} coeffs, energy = {energy:.4f}")

    # --- 5. CWT scalogram ---
    print("\n--- Continuous Wavelet Transform ---")
    # Chirp signal (increasing frequency)
    N_cwt = 512
    t_cwt = np.linspace(0, 1, N_cwt, endpoint=False)
    chirp = np.sin(2 * np.pi * (5 * t_cwt + 20 * t_cwt**2))
    
    scales = np.arange(1, 64)
    cwt = continuous_wavelet_transform(chirp, scales, wavelet='morlet')
    print(f"  CWT of chirp signal: {cwt.shape} (scales × time)")

    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        
        # Original signal
        ax = axes[0, 0]
        ax.plot(t, signal, 'b-')
        ax.set_xlabel('t')
        ax.set_ylabel('Amplitude')
        ax.set_title('Original Signal')
        ax.grid(True, alpha=0.3)
        
        # Wavelet coefficients
        ax = axes[0, 1]
        all_c = np.concatenate([c for c in coeffs])
        ax.stem(all_c, linefmt='b-', markerfmt='b.', basefmt='gray')
        ax.set_xlabel('Coefficient index')
        ax.set_ylabel('Value')
        ax.set_title('Haar Wavelet Coefficients')
        
        # Denoising comparison
        ax = axes[0, 2]
        denoised_haar = wavelet_denoise(noisy, 'haar', mode='soft')
        ax.plot(t, clean, 'k-', linewidth=2, label='Clean', alpha=0.7)
        ax.plot(t, noisy, 'gray', alpha=0.3, linewidth=0.5, label='Noisy')
        ax.plot(t, denoised_haar, 'r-', linewidth=1.5, label='Denoised (Haar)')
        ax.set_xlabel('t')
        ax.set_title('Wavelet Denoising')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Multi-resolution
        for level in range(min(4, len(coeffs_mr))):
            ax = axes[1, level] if level < 3 else None
            if ax is None:
                break
            c = coeffs_mr[level]
            label = "Approximation" if level == 0 else f"Detail {len(coeffs_mr)-level}"
            ax.plot(c, 'b-', linewidth=0.8)
            ax.set_title(f'{label} ({len(c)} pts)')
            ax.grid(True, alpha=0.3)
        
        # Compression
        ax = axes[2, 0]
        ax.plot(t, clean, 'k-', linewidth=2, label='Original')
        for keep in [0.2, 0.1, 0.05]:
            comp, _, _ = wavelet_compress(clean, 'haar', keep)
            ax.plot(t, comp, '--', alpha=0.7, label=f'{keep*100:.0f}% kept')
        ax.set_xlabel('t')
        ax.set_title('Wavelet Compression')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # CWT scalogram
        ax = axes[2, 1]
        ax.imshow(np.abs(cwt), aspect='auto', cmap='hot',
                 extent=[0, 1, scales[-1], scales[0]])
        ax.set_xlabel('Time')
        ax.set_ylabel('Scale')
        ax.set_title('CWT Scalogram (Chirp)')
        
        # Chirp signal
        ax = axes[2, 2]
        ax.plot(t_cwt, chirp, 'b-', linewidth=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.set_title('Chirp Signal (freq increases)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("wavelets.png", dpi=150)
        plt.show()
    except ImportError:
        print("(matplotlib not available)")
