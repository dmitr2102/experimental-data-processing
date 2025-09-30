# Reconstructing the past: estimating solar radio flux F10.7 from sunspot records
# Dmitrii Maliukov, Dmitrii Plotnikov, Timofei Kozlov, Skoltech, 2025

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

def smoothing(value): #smooth array
    arr = np.array(value)
    n = len(arr)
    smoothed = np.zeros_like(arr)

    for i in range(n):
        if i < 6:
            smoothed[i] = np.mean(arr[:6])
        elif i >= n - 6:
            smoothed[i] = np.mean(arr[-6:])
        else:
            window = arr[i-5:i+5]
            smoothed[i] = arr[i-6]/24 + np.sum(window)/12 + arr[i-6]/24 + arr[i+6]/24

    return smoothed
            
# read data from txt's
radio_df = pd.read_csv('data/Radio_flux_monthly_mean.txt', header=None, delimiter='\s+')
spot_df = pd.read_csv('data/Sunspot_number_monthly_mean.txt', header=None, delimiter='\s+')
# year, month --> absolute month
radio_df['abs_month'] = radio_df[1] + 12*(radio_df[0]-spot_df[0][0])
spot_df['abs_month'] = spot_df[1] + 12*(spot_df[0]-spot_df[0][0])
# smoothing
radio_df['RadioFlux_smoothed'] = smoothing(radio_df[2].copy())
spot_df['SpotNum_smoothed'] = smoothing(spot_df[2].copy())

# print(radio_df)
# print(spot_df)

radio_data = pd.DataFrame({
    'Year': radio_df[0],
    'Month': radio_df[1],
    'AbsMonth': radio_df['abs_month'],
    'RadioFlux': radio_df[2], 
    'RadioFlux_smoothed': radio_df['RadioFlux_smoothed']
})
#print(radio_data)

spot_data = pd.DataFrame({
    #'Year': spot_df[0],
    #'Month': spot_df[1],
    'AbsMonth': spot_df['abs_month'],
    'SpotNum': spot_df[2],
    'SpotNum_smoothed': spot_df['SpotNum_smoothed']
})
#print(spot_data)

# CHANGE RECORDED PERIOD HERE
merged_df = pd.merge(radio_data, spot_data, on='AbsMonth', how='inner') # inner - только общие годы
print(merged_df)

# ======= 3, 4 =======
fig, ax = plt.subplots(figsize=(12, 6)) # window size

ax.plot(merged_df['AbsMonth'], merged_df['RadioFlux'],
        color="pink",
        label='Radio Flux')

ax.plot(merged_df['AbsMonth'], merged_df['RadioFlux_smoothed'],
        color="purple",
        label='Radio Flux (smoothed)')

ax.plot(merged_df['AbsMonth'], merged_df['SpotNum'],
        color="orange",
        label='Sunspot Number')

ax.plot(merged_df['AbsMonth'], merged_df['SpotNum_smoothed'],
        color="red",
        label='Sunspot Number (smoothed)')

ax.set_xlabel('Time')
ax.set_ylabel('Radio Flux and Spots number', color='black')
ax.tick_params(axis='y', labelcolor='black') 
ax.legend(loc='upper left')
plt.title('Radio activity and Sunspots number for each month since 1951 November')
plt.tight_layout()
plt.show()

# ======= 5 ======= for spots and radio (they proportional)
plt.figure(figsize=(10, 6))
plt.scatter(merged_df['SpotNum_smoothed'], merged_df['RadioFlux_smoothed'],
            s=5,             #Size
            alpha=0.5,       #transparency
            color='purple',
            label='Smoothed Radio Flux')

plt.xlabel('Sunspots')
plt.ylabel('Radio Flux')
plt.title('Scatter Plot of smoothed sunspot and F10.7 data')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

# ======= 7 ======= 
#[1, R, R^2, R^3]^T
#X = [1, R^2, R^3, R^4]
f = merged_df['RadioFlux_smoothed']
R = merged_df['SpotNum_smoothed']
X = np.column_stack([
    np.ones(len(R)),
    R,
    R**2, 
    R**3 
]) #need to check

model = LinearRegression(fit_intercept=False)  # fit_intercept=False, т.к. мы вручную добавили 1
model.fit(X, f)

# Вектор коэффициентов β
beta = model.coef_  # [β0, β1, β2, β3]
print(f"Коэффициенты β: {beta}")
print(f"β0 (intercept): {beta[0]}")
print(f"β1: {beta[1]}")
print(f"β2: {beta[2]}")
print(f"β3: {beta[3]}")

r_squared = model.score(X, f)
print(f"R²: {r_squared}")






# ========= LEARNING LOG =========

# radio_df['Year'] = radio_df[0].str[0:4] # just another wat to extract data (skill issue)
# radio_df['Month'] = radio_df[0].str[5:7]
# radio_df['Flux'] = radio_df[0].str[-5:]

# radio_df['Year'] = radio_df[0].str.extract(r'(\S+)') # (skill issue 2)
# radio_df['Month'] = radio_df[0].str.extract(r'\S+\s+(\S+)')
# radio_df['Flux'] = radio_df[0].str.extract(r'\s*(\S+)$') # $ is end
# del radio_df[0] # delete initial col

# spotNum_df[0] = spotNum_df[0].astype(str).str.replace('\r', '', regex=False)

# spotNum_df['Year'] = spotNum_df[0].str.extract(r'(\S+)')
# spotNum_df['Month'] = spotNum_df[0].str.extract(r'\S+\s+(\S+)')
# spotNum_df['Amount'] = spotNum_df[0].str.extract(r'\s*(\S+)$') # $ is end
# del spotNum_df[0] # delete initial col

    
    # for i in range(len(value)):
    #     if i < 6:
    #         value[i] = (value[0]+value[1]+value[2]+value[3]+value[4]+value[5])/6
    #     elif i >= len(value) - 6:
    #         value[i] = (value[len(value)-6] + value[len(value)-5] + value[len(value)-4] + value[len(value)-3] + value[len(value)-2] + value[len(value)-1])/6
    #     else:
    #         value[i] = value[i-6]/24 + (value[i-5] + value[i-4] + value[i-3] + value[i-2] + value[i-1] + value[i] + value[i+1] + value[i+2] + value[i+3] + value[i+4] + value[i+5])/12 + value[i+6]/24
    # return value

