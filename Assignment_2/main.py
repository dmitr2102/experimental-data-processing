# Reconstructing the past: estimating solar radio flux F10.7 from sunspot records
# Dmitrii Maliukov, Dmitrii Plotnikov, Timofei Kozlov, Skoltech, 2025

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Part I – Calm Walk: Learning the Panda's Rhythm

# Let's start by simulating the panda's true walk (random walk).
# A random walk means each step depends on the previous one plus some noise.

# Generate a random walk trajectory for the panda.
def generate_random_walk(n_points, sigma_w2, x0=10):
    # Gaussian noise with variance sigma_w2 (so std = sqrt(variance))
    noise = np.random.normal(0, np.sqrt(sigma_w2), n_points)
    
    # The panda starts at x0 meters and then keeps adding noisy steps
    X = np.zeros(n_points)
    X[0] = x0
    for i in range(1, n_points):
        X[i] = X[i-1] + noise[i]
    return X

# Generate noisy measurements of the panda's walk.
def generate_measurements(X, sigma_eta2):
    # Measurements are the true positions + extra sensor noise
    noise = np.random.normal(0, np.sqrt(sigma_eta2), len(X))
    Z = X + noise
    return Z

# Apply exponential smoothing filter to noisy measurements.
def exponential_smoothing(Z, alpha):
    smoothed = np.zeros(len(Z))
    smoothed[0] = Z[0]  # Start from first measurement
    for i in range(1, len(Z)):
        # Weighted combination: a bit of new measurement, a bit of old estimate
        smoothed[i] = alpha * Z[i] + (1 - alpha) * smoothed[i-1]
    return smoothed

# Parameters
sigma_w2 = 8   # process noise variance (group specific)
sigma_eta2 = 16 # measurement noise variance (group specific)
n_points = 3000 # points: 300 / 3000

# Generate data
X_true = generate_random_walk(n_points, sigma_w2)
Z_meas = generate_measurements(X_true, sigma_eta2)

# Compute smoothing coefficient alpha (from given formula)
chi = sigma_w2 / sigma_eta2
alpha = (-chi + np.sqrt(chi**2 + 4*chi)) / 2

print(alpha)

# Apply exponential smoothing
X_smooth = exponential_smoothing(Z_meas, alpha)

# Plot results
plt.figure(figsize=(10,5))
plt.plot(X_true, label="True Panda Path")
plt.plot(Z_meas, label="Noisy Measurements", alpha=0.5)
plt.plot(X_smooth, label="Exponential Smoothing", linewidth=2)
plt.xlabel("Step")
plt.ylabel("Position (m)")
plt.title("Calm Walk: Panda Tracking")
plt.legend()
plt.show()

# Part II – Shaky Cameras: Harsh Conditions

# Here we increase the noise A LOT to simulate bad visibility
sigma_w2_harsh = 282
sigma_eta2_harsh = 972
n_points_harsh = 300

# Generate new data
X_true_harsh = generate_random_walk(n_points_harsh, sigma_w2_harsh)
Z_meas_harsh = generate_measurements(X_true_harsh, sigma_eta2_harsh)

# Compute alpha for harsh conditions
chi_harsh = sigma_w2_harsh / sigma_eta2_harsh
alpha_harsh = (-chi_harsh + np.sqrt(chi_harsh**2 + 4*chi_harsh)) / 2

print(alpha_harsh)

# Exponential smoothing in harsh conditions
X_smooth_harsh = exponential_smoothing(Z_meas_harsh, alpha_harsh)

# Now let's find the running mean window size M
sigma_ES2 = sigma_eta2_harsh * alpha_harsh / (2 - alpha_harsh)  # formula given
M = int(round(sigma_eta2_harsh / sigma_ES2))  # rearranged RM variance = ES variance

def running_mean(Z, M):
    """Compute running mean (moving average) with window size M."""
    cumsum = np.cumsum(np.insert(Z, 0, 0))
    return (cumsum[M:] - cumsum[:-M]) / M

# Running mean (shorter output due to windowing, so align for plotting)
X_runmean = running_mean(Z_meas_harsh, M)
steps = np.arange(M-1, n_points_harsh)

# Plot results for harsh conditions
plt.figure(figsize=(10,5))
plt.plot(X_true_harsh, label="True Panda Path")
plt.plot(Z_meas_harsh, label="Noisy Measurements", alpha=0.5)
plt.plot(X_smooth_harsh, label="Exponential Smoothing", linewidth=2)
plt.plot(steps, X_runmean, label=f"Running Mean (M={M})", linewidth=2)
plt.xlabel("Step")
plt.ylabel("Position (m)")
plt.title("Shaky Cameras: Tracking Panda under Harsh Conditions")
plt.legend()
plt.show()