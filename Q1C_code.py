
# ============================================================
# ASTR 5900 - HW05
# Question 1(c): FFT vs DFT + Timing Analysis
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import time

# ------------------------------------------------------------
# Function
# ------------------------------------------------------------
def f(x):
    return np.exp(-50 * (x - 0.5)**2)

# ------------------------------------------------------------
# Manual DFT (same as part b)
# ------------------------------------------------------------
def DFT(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

# ------------------------------------------------------------
# Timing setup
# ------------------------------------------------------------
dft_times = []
fft_times = []

# DFT: N = 2^j, j = 3,...,11
dft_N = [2**j for j in range(3, 12)]

# FFT: N = 2^j, j = 3,...,20
fft_N = [2**j for j in range(3, 21)]

# ------------------------------------------------------------
# DFT timing
# ------------------------------------------------------------
for N in dft_N:
    x = np.linspace(0, 1, N, endpoint=False)
    xn = f(x)

    start = time.time()
    for _ in range(100):
        DFT(xn)
    end = time.time()

    avg_time = (end - start) / 100
    dft_times.append(avg_time)

# ------------------------------------------------------------
# FFT timing
# ------------------------------------------------------------
for N in fft_N:
    x = np.linspace(0, 1, N, endpoint=False)
    xn = f(x)

    start = time.time()
    for _ in range(100):
        np.fft.fft(xn)
    end = time.time()

    avg_time = (end - start) / 100
    fft_times.append(avg_time)

# ------------------------------------------------------------
# Plot time complexity
# ------------------------------------------------------------
plt.figure(figsize=(10,6))

plt.loglog(dft_N, dft_times, 'o-', label='DFT (Manual)')
plt.loglog(fft_N, fft_times, 's-', label='FFT (NumPy)')

plt.xlabel('N (Number of grid points)')
plt.ylabel('Execution Time (s)')
plt.title('Time Complexity: DFT vs FFT')
plt.legend()
plt.grid(True)

plt.savefig("time_complexity.png")
plt.show()

# ------------------------------------------------------------
# Compare FFT vs DFT for one N
# ------------------------------------------------------------
N = 128
x = np.linspace(0, 1, N, endpoint=False)
xn = f(x)

X_dft = DFT(xn)
X_fft = np.fft.fft(xn)

k_vals = np.fft.fftfreq(N)

plt.figure(figsize=(10,6))
plt.plot(k_vals, np.abs(X_dft), label='DFT')
plt.plot(k_vals, np.abs(X_fft), '--', label='FFT')

plt.xlabel('k')
plt.ylabel('|F(k)|')
plt.title('DFT vs FFT Comparison')
plt.legend()
plt.grid()

plt.savefig("fft_vs_dft.png")
plt.show()