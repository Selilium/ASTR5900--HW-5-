
# ============================================================
# ASTR 5900 - HW05
# Question 1(b): Manual DFT Implementation
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Function definition
# ------------------------------------------------------------
def f(x):
    return np.exp(-50 * (x - 0.5)**2)

# Analytical Fourier Transform (from part a)
def f_hat_analytic(k):
    return np.sqrt(np.pi / 50) * np.exp(-k**2 / 200) * np.exp(-1j * k / 2)

# ------------------------------------------------------------
# Manual DFT
# ------------------------------------------------------------
def DFT(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

# ------------------------------------------------------------
# Inverse DFT
# ------------------------------------------------------------
def IDFT(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
        x[n] /= N
    return x

# ------------------------------------------------------------
# Domain
# ------------------------------------------------------------
N_values = [32, 64, 128]

plt.figure(figsize=(10, 6))

for N in N_values:
    x = np.linspace(0, 1, N, endpoint=False)
    dx = x[1] - x[0]
    xn = f(x)

    # Compute DFT
    Xk = DFT(xn)

    # k values (frequency)
    k_vals = 2 * np.pi * np.fft.fftfreq(N, d=dx)

    # Sort for plotting
    idx = np.argsort(k_vals)
    k_sorted = k_vals[idx]
    Xk_sorted = Xk[idx]

    plt.plot(k_sorted, np.abs(Xk_sorted), label=f"DFT N={N}")

# Analytical curve
k_cont = np.linspace(-50, 50, 1000)
plt.plot(k_cont, np.abs(f_hat_analytic(k_cont)), 'k--', label="Analytical")

plt.xlabel("k")
plt.ylabel(r"$|\hat{f}(k)|$")
plt.title("DFT vs Analytical Fourier Transform")
plt.legend()
plt.grid()
plt.savefig("dft_comparison.png")
plt.show()

# ------------------------------------------------------------
# Inverse Transform Check
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))

for N in N_values:
    x = np.linspace(0, 1, N, endpoint=False)
    xn = f(x)

    Xk = DFT(xn)
    x_reconstructed = IDFT(Xk)

    plt.plot(x, xn, label=f"Original N={N}")
    plt.plot(x, x_reconstructed.real, '--', label=f"Reconstructed N={N}")

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Inverse DFT Reconstruction")
plt.legend()
plt.grid()
plt.savefig("inverse_check.png")
plt.show()