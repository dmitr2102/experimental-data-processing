# Reconstructing the past: estimating solar radio flux F10.7 from sunspot records
# Dmitrii Maliukov, Dmitrii Plotnikov, Timofei Kozlov, Skoltech, 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path

def smoothing(data: pd.DataFrame):
    R = []
    for i in range(0, len(data)):
        if i < 6:
            R.append(sum(data[2][:6])/6)
        elif i > len(data) - 7:
            R.append(sum(data[2][-6:])/6)
        else:
            R.append((data[2][i-6])/24 + sum(data[2][i-5:i+5])/12 + (data[2][i+6])/24)
    data[4] = R
            

try:
    # Read radio flux monthly mean
    rf = pd.read_csv(os.path.join("data", "Radio_flux_monthly_mean.txt"), sep=r"\s+", header = None)
    # Read sunspot number monthly mean
    ss = pd.read_csv(os.path.join("data", "Sunspot_number_monthly_mean.txt"), sep=r"\s+", header = None)
except:
        # Read radio flux monthly mean
    rf = pd.read_csv(os.path.join("Assignment_1", "data", "Radio_flux_monthly_mean.txt"), sep=r"\s+", header = None)
    # Read sunspot number monthly mean
    ss = pd.read_csv(os.path.join("Assignment_1", "data", "Sunspot_number_monthly_mean.txt"), sep=r"\s+", header = None)

rf_min_year = min(rf[0])
ss_min_year = min(ss[0])
min_year = min(min(ss[0]), min(rf[0]))

rf[3] = (rf[0] - min_year) * 12 + rf[1]
ss[3] = (ss[0] - min_year) * 12 + ss[1]

plt.figure(figsize=(10, 10))
plt.title("Radio flux & sunspot number monthly mean")
plt.xlabel("Month")
plt.ylabel("Flux\nSunspot number")
plt.plot(rf[3], rf[2], linewidth=1, color = 'r')
plt.plot(ss[3], ss[2], linewidth=1, color = 'b')
plt.legend(["Radio flux", "Sunspot number"])
plt.savefig("Figure.png")

smoothing(rf)
smoothing(ss)

plt.figure(figsize=(12, 6))
plt.title("RF smmoth")
plt.plot(rf[3], rf[2], linewidth=1, color = 'r')
plt.plot(rf[3], rf[4], linewidth=1, color = 'b')
plt.savefig("RF smooth.png")

plt.figure(figsize=(12, 6))
plt.title("SS smooth")
plt.plot(ss[3], ss[2], linewidth=1, color = 'r')
plt.plot(ss[3], ss[4], linewidth=1, color = 'b')
plt.savefig("SS smooth.png")

F = rf[2]
R = np.matrix([[1]*len(ss), ss[4], ss[4] ** 2, ss[4] ** 3])
print(np.size(R))