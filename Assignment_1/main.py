# Reconstructing the past: estimating solar radio flux F10.7 from sunspot records
# Dmitrii Maliukov, Dmitrii Plotnikov, Timofei Kozlov, Skoltech, 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path

def smoothing(data: pd.DataFrame):
    for i in range(0, len(data)):
        if i < 6 or i > len(data) - 6:
            data

# Read radio flux monthly mean
rf = pd.read_csv(os.path.join("data", "Radio_flux_monthly_mean.txt"), sep=r"\s+", header = None)
# Read sunspot number monthly mean
ss = pd.read_csv(os.path.join("data", "Sunspot_number_monthly_mean.txt"), sep=r"\s+", header = None)

rf_min_year = min(rf[0])
ss_min_year = min(ss[0])

rf[3] = (rf[0] - rf_min_year) * 12 + rf[1]
ss[3] = (ss[0] - ss_min_year) * 12 + ss[1]

print(rf)
print(ss)

plt.figure(figsize=(10, 10))
plt.title("Radio flux & sunspot number monthly mean")
plt.xlabel("Month")
plt.ylabel("Number")
plt.plot(rf[3], rf[2], linewidth=1, color = 'r')
plt.plot(ss[3], ss[2], linewidth=1, color = 'c')
plt.legend(["Radio flux", "Sunspot number"])
plt.savefig("Figure.png")