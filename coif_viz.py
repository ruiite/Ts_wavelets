import numpy as np
import matplotlib.pyplot as plt
import pywt

# Choose the Daubechies wavelet, e.g., 'db2'

plt.figure(figsize=(10, 6))

for i in range(3):
    wavelet = pywt.Wavelet(f'coif{i+1}')
    phi, psi, x = wavelet.wavefun(level=1)
    plt.plot(x, psi, label=f"Coiflet coif{i+1} Wavelet", linewidth=3)


plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Coiflet Wavelet Function")
plt.legend()
plt.grid(True)
plt.show()