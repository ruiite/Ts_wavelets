import numpy as np
import pywt
import matplotlib.pyplot as plt

# Створення синтетичного довгого часового ряду
n = 10000  # довжина сигналу
t = np.linspace(0, 1, n)

#signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(n)
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(n)
# Виконання стаціонарного вейвлет-перетворення з використанням 3 рівнів
wavelet = 'db1'
level = 1
coeffs = pywt.swt(signal, wavelet, level)

# coeffs повертає список з кортежами (approximation, detail) для кожного рівня
print("Кількість рівнів:", len(coeffs))
for i, (approx, detail) in enumerate(coeffs, 1):
    print(f"Рівень {i}: розмір апроксимації = {approx.shape}, розмір деталей = {detail.shape}")

# Обчислення загальної кількості ознак при конкатенації коефіцієнтів
total_features = (level * signal.shape[0])
print("Загальна кількість ознак (тільки деталі):", total_features)

# Побудова графіків апроксимаційних та детальних коефіцієнтів останнього рівня
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(coeffs[-1][0])
plt.title("Апроксимаційні коефіцієнти на останньому рівні")
plt.subplot(2, 1, 2)
plt.plot(coeffs[-1][1])
plt.title("Детальні коефіцієнти на останньому рівні")
plt.tight_layout()
plt.show()