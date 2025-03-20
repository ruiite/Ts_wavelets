import numpy as np
import matplotlib.pyplot as plt

def haar_wavelet(t):
    """
    Haar mother wavelet function.
    Returns 1 for 0 <= t < 0.5, -1 for 0.5 <= t < 1, and 0 elsewhere.
    """
    return np.where((t >= 0) & (t < 0.5), 1,
                    np.where((t >= 0.5) & (t < 1), -1, 0))

def haar_wavelet_scaled(t, j, k):
    """
    Scaled and translated Haar wavelet.
    psi_{j,k}(t) = 2^(j/2) * psi(2^j * t - k)
    
    Parameters:
        t : array_like
            Time or spatial variable.
        j : int
            Scale parameter.
        k : int
            Translation parameter.
    """
    return (2**(j/2)) * haar_wavelet(2**j * t - k)

# Create a time axis
t = np.linspace(-1, 2, 1000)

# Plot the Haar wavelet at different scales and translations
plt.figure(figsize=(10, 6))

# Haar wavelet for j=0, k=0
plt.plot(t, haar_wavelet_scaled(t, j=0, k=0), label=r'Haar wavelet', linewidth=3)

# Haar wavelet for j=1, k=0 (narrower, shifted)
# plt.plot(t, haar_wavelet_scaled(t, j=1, k=0), label=r'$\psi_{1,0}(t)$ (j=1, k=0)')

# # Haar wavelet for j=1, k=1 (shifted to the right)
# plt.plot(t, haar_wavelet_scaled(t, j=1, k=1), label=r'$\psi_{1,1}(t)$ (j=1, k=1)')

plt.xlabel('t')
plt.ylabel(r'$\psi_{j,k}(t)$')
plt.title('Visualization of Haar Wavelet Functions')
plt.legend()
plt.grid(True)
plt.show()