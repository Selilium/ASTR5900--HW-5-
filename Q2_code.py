
# ============================================================
# ASTR 5900 - HW05
# Question 2: Heat Equation via Fourier Transform
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
alpha = 0.005
dt = 0.001
T = 5
N = 128

# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------
x = np.linspace(0, 1, N, endpoint=False)
dx = x[1] - x[0]

# Initial condition
u0 = np.exp(-50 * (x - 0.5)**2)

# ------------------------------------------------------------
# DFT + IDFT (reuse from Q1)
# ------------------------------------------------------------
def DFT(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def IDFT(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
        x[n] /= N
    return x

# ------------------------------------------------------------
# k-space setup
# ------------------------------------------------------------
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

# initial Fourier transform
u_hat = DFT(u0)

# ------------------------------------------------------------
# Time evolution
# ------------------------------------------------------------
time_steps = int(T / dt)
u_xt = []

for t in range(time_steps):
    # Euler update in k-space
    u_hat = u_hat + dt * (-alpha * k**2 * u_hat)

    # Back to real space
    u = IDFT(u_hat).real
    u_xt.append(u)

u_xt = np.array(u_xt)

# ------------------------------------------------------------
# Color plot (space vs time)
# ------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.imshow(u_xt, extent=[0,1,0,T], aspect='auto', origin='lower')
plt.colorbar(label='u(x,t)')
plt.xlabel('x')
plt.ylabel('time')
plt.title('Heat Equation Evolution')

plt.savefig("heat_evolution.png")
plt.show()

# ------------------------------------------------------------
# Total heat over time
# ------------------------------------------------------------
total_heat = np.sum(u_xt, axis=1) * dx

plt.figure()
plt.plot(np.linspace(0,T,time_steps), total_heat)
plt.xlabel('time')
plt.ylabel('Total Heat')
plt.title('Total Heat vs Time')
plt.grid()

plt.savefig("total_heat.png")
plt.show()