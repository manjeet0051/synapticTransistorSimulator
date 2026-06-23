import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import r2_score, mean_squared_error

print("="*60)
print("IGZO TRANSISTOR MODEL VALIDATION")
print("="*60)

DATA_DIR = "data"
MODEL_DIR = "models"

# ==========================================================
# 1. LOAD MODELS
# ==========================================================

print("\n[1/6] Loading trained models...")

poly_model = joblib.load(os.path.join(MODEL_DIR,"poly_model.pkl"))
nn_model   = joblib.load(os.path.join(MODEL_DIR,"nn_model.pkl"))
gain_model = joblib.load(os.path.join(MODEL_DIR,"gain_model.pkl"))

print("Models loaded successfully.")

# ==========================================================
# 2. LOAD TRANSFER CURVE DATA
# ==========================================================

print("\n[2/6] Loading transfer curve data...")

df = pd.read_csv(os.path.join(DATA_DIR,"IDVD_IDBG_IGZO_MgO.csv"))

curve_map = {
    "CW10": ("Unnamed: 0","CW10"),
    "CW11": ("Unnamed: 2","Cw11"),
    "CW12": ("Unnamed: 4","CW12"),
    "CW13": ("Unnamed: 6","CW13"),
}

vgs_all = []
ids_all = []

for name,(vcol,icol) in curve_map.items():

    vgs = pd.to_numeric(df[vcol],errors="coerce").dropna().values
    ids = pd.to_numeric(df[icol],errors="coerce").dropna().values

    n = min(len(vgs),len(ids))

    vgs = vgs[:n]
    ids = ids[:n]

    mask = ids > 0

    vgs_all.append(vgs[mask])
    ids_all.append(ids[mask])

vgs_all = np.concatenate(vgs_all)
ids_all = np.concatenate(ids_all)

print("Total transfer data points:",len(vgs_all))

# ==========================================================
# 3. MODEL PREDICTION
# ==========================================================

print("\n[3/6] Predicting IDS using ML models...")

X = vgs_all.reshape(-1,1)

ids_poly = 10**poly_model.predict(X)
ids_nn   = 10**nn_model.predict(X)

# ==========================================================
# 4. TRANSFER CURVE VALIDATION GRAPH
# ==========================================================

print("\n[4/6] Plotting transfer curve comparison...")

vgs_fine = np.linspace(vgs_all.min(),vgs_all.max(),400).reshape(-1,1)

ids_poly_fine = 10**poly_model.predict(vgs_fine)
ids_nn_fine   = 10**nn_model.predict(vgs_fine)

plt.figure(figsize=(8,6))

plt.semilogy(vgs_all,ids_all,'o',alpha=0.3,label="Experimental")

plt.semilogy(vgs_fine,ids_poly_fine,label="Polynomial Model",linewidth=2)

plt.semilogy(vgs_fine,ids_nn_fine,label="Neural Network Model",linewidth=2)

plt.xlabel("VGS (V)")
plt.ylabel("IDS (A)")
plt.title("Transfer Curve Validation")

plt.legend()
plt.grid(True)

plt.show()

# ==========================================================
# 5. ERROR METRICS
# ==========================================================

print("\n[5/6] Calculating accuracy metrics...")

r2_poly = r2_score(np.log10(ids_all),np.log10(ids_poly))
r2_nn   = r2_score(np.log10(ids_all),np.log10(ids_nn))

rmse_poly = np.sqrt(mean_squared_error(ids_all,ids_poly))
rmse_nn   = np.sqrt(mean_squared_error(ids_all,ids_nn))

print("\nTransfer Curve Accuracy")

print("Polynomial Model R²:",round(r2_poly,5))
print("Neural Network R²:",round(r2_nn,5))

print("\nRMSE Error")

print("Polynomial RMSE:",rmse_poly)
print("Neural Network RMSE:",rmse_nn)

# ==========================================================
# 6. EPSC GAIN VALIDATION
# ==========================================================

print("\n[6/6] Validating EPSC Gain model...")

df_gain = pd.read_csv(os.path.join(DATA_DIR,"filterchar_IGZO_MgO_new.csv"))

freq = pd.to_numeric(df_gain["frequesncy"],errors="coerce").dropna().values
gain = pd.to_numeric(df_gain["EPSC Gain"],errors="coerce").dropna().values

n = min(len(freq),len(gain))

freq = freq[:n]
gain = gain[:n]

freq_log = np.log10(freq).reshape(-1,1)

gain_pred = 10**gain_model.predict(freq_log)

plt.figure(figsize=(8,6))

plt.semilogy(freq,gain,'o',label="Experimental")

plt.semilogy(freq,gain_pred,label="Gain Model Prediction",linewidth=2)

plt.xlabel("Frequency (Hz)")
plt.ylabel("EPSC Gain")
plt.title("Synaptic Gain Validation")

plt.legend()
plt.grid(True)

plt.show()

r2_gain = r2_score(np.log10(gain),np.log10(gain_pred))

print("\nEPSC Gain Model R²:",round(r2_gain,5))

print("\n"+"="*60)
print("VALIDATION COMPLETE")
print("="*60)