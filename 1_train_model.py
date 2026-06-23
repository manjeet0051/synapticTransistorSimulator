"""
==============================================================
STEP 1: IGZO/MgO Synaptic Transistor — Model Training
==============================================================
Ye script:
  - Experimental data load karti hai
  - Device parameters extract karti hai
  - ML model train karti hai
  - Model save karti hai (models/ folder mein)
  - Training plots banati hai

Run karo: python 1_train_model.py
==============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Windows GUI backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────
DATA_DIR   = "data"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

IDVD_FILE  = os.path.join(DATA_DIR, "IDVD_IDBG_IGZO_MgO.csv")
FILT_FILE  = os.path.join(DATA_DIR, "filterchar_IGZO_MgO_new.csv")

print("=" * 62)
print("   IGZO/MgO Synaptic Transistor — Model Training")
print("=" * 62)

# ==============================================================
# 1. LOAD DATA
# ==============================================================
print("\n[1/5] Data load ho raha hai...")

df_iv = pd.read_csv(IDVD_FILE, header=0)
df_fc = pd.read_csv(FILT_FILE, header=0)

# Transfer curves (4 channel widths)
curve_map = {
    'CW10': ('Unnamed: 0', 'CW10'),
    'CW11': ('Unnamed: 2', 'Cw11'),
    'CW12': ('Unnamed: 4', 'CW12'),
    'CW13': ('Unnamed: 6', 'CW13'),
}
curves = {}
for name, (vcol, icol) in curve_map.items():
    vgs = pd.to_numeric(df_iv[vcol], errors='coerce').dropna().values
    ids = pd.to_numeric(df_iv[icol], errors='coerce').dropna().values
    n   = min(len(vgs), len(ids))
    vgs, ids = vgs[:n], ids[:n]
    mask = ids > 0
    curves[name] = {'VGS': vgs[mask], 'IDS': ids[mask]}

# Use CW10 as primary + merge all for richer training
vgs_all, ids_all = [], []
for c in curves.values():
    vgs_all.append(c['VGS'])
    ids_all.append(c['IDS'])
vgs_all = np.concatenate(vgs_all)
ids_all = np.concatenate(ids_all)

# EPSC Gain vs Frequency
freq_vals = pd.to_numeric(df_fc['frequesncy'], errors='coerce').dropna().values
gain_vals = pd.to_numeric(df_fc['EPSC Gain'],  errors='coerce').dropna().values
n_fg = min(len(freq_vals), len(gain_vals))
freq_vals, gain_vals = freq_vals[:n_fg], gain_vals[:n_fg]

print(f"   Transfer data : {len(vgs_all)} points (4 CW merged)")
print(f"   EPSC Gain     : {n_fg} frequency points")

# ==============================================================
# 2. DEVICE PARAMETER EXTRACTION
# ==============================================================
print("\n[2/5] Device parameters extract ho rahe hain...")

vgs_p = curves['CW10']['VGS']
ids_p = curves['CW10']['IDS']

ion   = ids_all.max()
ioff  = ids_all.min()
ratio = ion / ioff

# Subthreshold Swing
log_ids  = np.log10(ids_p)
idx_sort = np.argsort(vgs_p)
vgs_s    = vgs_p[idx_sort]
log_s    = uniform_filter1d(log_ids[idx_sort], size=3)
slope    = np.gradient(log_s, vgs_s)
valid_sl = slope[np.isfinite(slope)]
ss_mv    = abs(1.0 / valid_sl.max()) * 1000 if valid_sl.max() > 0 else float('nan')

# Vth via sqrt(IDS) extrapolation
mask_ab = vgs_p > 1.0
if mask_ab.sum() > 2:
    coeffs = np.polyfit(vgs_p[mask_ab], np.sqrt(ids_p[mask_ab]), 1)
    vth = -coeffs[1] / coeffs[0]
else:
    vth = 0.5

# Transconductance
gm = np.gradient(ids_p, vgs_p)

params = {
    'ION':       ion,
    'IOFF':      ioff,
    'ION_IOFF':  ratio,
    'SS_mVdec':  ss_mv,
    'Vth_V':     vth,
    'gm_max':    np.nanmax(gm),
}

print(f"   ION        = {ion:.3e} A")
print(f"   IOFF       = {ioff:.3e} A")
print(f"   ION/IOFF   = {ratio:.2e}")
print(f"   SS         = {ss_mv:.1f} mV/dec")
print(f"   Vth        = {vth:.3f} V")
print(f"   gm max     = {np.nanmax(gm):.3e} A/V")

joblib.dump(params, os.path.join(MODEL_DIR, "device_params.pkl"))
print("   Saved: models/device_params.pkl")

# ==============================================================
# 3. TRAIN TRANSFER CURVE MODEL (IDS predictor)
# ==============================================================
print("\n[3/5] Transfer curve ML model train ho raha hai...")

X = vgs_all.reshape(-1, 1)
y = np.log10(ids_all)   # log scale mein train karo

# Model A: Polynomial Regression (fast, interpretable)
poly   = PolynomialFeatures(degree=8, include_bias=True)
scaler = StandardScaler()
linreg = LinearRegression()

poly_model = Pipeline([
    ('poly',   poly),
    ('scaler', scaler),
    ('reg',    linreg)
])
poly_model.fit(X, y)
y_poly = poly_model.predict(X)
r2_poly = r2_score(y, y_poly)
print(f"   Polynomial model R² = {r2_poly:.5f}")

# Model B: Neural Network (more flexible)
nn_model = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(64, 64, 32),
        activation='relu',
        max_iter=2000,
        random_state=42,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        verbose=False
    ))
])
nn_model.fit(X, y)
y_nn   = nn_model.predict(X)
r2_nn  = r2_score(y, y_nn)
print(f"   Neural Network R²   = {r2_nn:.5f}")

# Save both models
joblib.dump(poly_model, os.path.join(MODEL_DIR, "poly_model.pkl"))
joblib.dump(nn_model,   os.path.join(MODEL_DIR, "nn_model.pkl"))
print("   Saved: models/poly_model.pkl")
print("   Saved: models/nn_model.pkl")

# ==============================================================
# 4. TRAIN EPSC GAIN MODEL (Frequency → Gain predictor)
# ==============================================================
print("\n[4/5] EPSC Gain model train ho raha hai...")

# Log transform frequency for better fitting
X_freq = np.log10(freq_vals).reshape(-1, 1)
y_gain = np.log10(gain_vals)

gain_poly  = PolynomialFeatures(degree=4)
gain_model = Pipeline([
    ('poly',   gain_poly),
    ('scaler', StandardScaler()),
    ('reg',    LinearRegression())
])
gain_model.fit(X_freq, y_gain)
y_gain_pred = gain_model.predict(X_freq)
r2_gain = r2_score(y_gain, y_gain_pred)
print(f"   EPSC Gain model R² = {r2_gain:.5f}")

joblib.dump(gain_model, os.path.join(MODEL_DIR, "gain_model.pkl"))
print("   Saved: models/gain_model.pkl")

# Save summary
summary = {
    'poly_r2':    r2_poly,
    'nn_r2':      r2_nn,
    'gain_r2':    r2_gain,
    'vgs_min':    float(vgs_all.min()),
    'vgs_max':    float(vgs_all.max()),
    'freq_min':   float(freq_vals.min()),
    'freq_max':   float(freq_vals.max()),
    **params
}
joblib.dump(summary, os.path.join(MODEL_DIR, "training_summary.pkl"))

# ==============================================================
# 5. TRAINING PLOTS
# ==============================================================
print("\n[5/5] Training plots ban rahe hain...")

plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 10), facecolor='#08080f')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                        left=0.07, right=0.97, top=0.92, bottom=0.07)

PANEL = '#0c0c1e'
GRID  = '#1c1c3a'
C = ['#00f5d4', '#f72585', '#ffd60a', '#4cc9f0', '#7209b7']

def ax_style(ax, title, xlabel, ylabel):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=7)
    ax.set_xlabel(xlabel, color='#bbbbbb', fontsize=9)
    ax.set_ylabel(ylabel, color='#bbbbbb', fontsize=9)
    ax.tick_params(colors='#999999', labelsize=8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color('#2a2a4a')
    ax.grid(True, color=GRID, alpha=0.5, linewidth=0.5)

vgs_fine = np.linspace(vgs_all.min(), vgs_all.max(), 500).reshape(-1, 1)

# Plot 1: All CW transfer curves
ax1 = fig.add_subplot(gs[0, 0])
cm4 = plt.cm.cool(np.linspace(0.1, 0.95, 4))
for i, (nm, c) in enumerate(curves.items()):
    ax1.semilogy(c['VGS'], c['IDS']*1e9, color=cm4[i], lw=1.8, label=nm)
ax_style(ax1, 'Transfer Characteristics', 'V$_{GS}$ (V)', 'I$_{DS}$ (nA)')
ax1.legend(fontsize=8, framealpha=0.2)

# Plot 2: Poly model fit
ax2 = fig.add_subplot(gs[0, 1])
ids_poly_fine = 10 ** poly_model.predict(vgs_fine)
ax2.semilogy(vgs_all, ids_all*1e9, 'o', color=C[0], ms=2.5, alpha=0.4, label='Data')
ax2.semilogy(vgs_fine.ravel(), ids_poly_fine*1e9, color=C[1], lw=2.2,
             label=f'Poly Fit (R²={r2_poly:.4f})')
ax2.axvline(vth, color=C[2], ls='--', lw=1.5, label=f'Vth={vth:.2f}V')
ax_style(ax2, 'Polynomial Model Fit', 'V$_{GS}$ (V)', 'I$_{DS}$ (nA)')
ax2.legend(fontsize=7.5, framealpha=0.2)

# Plot 3: NN model fit
ax3 = fig.add_subplot(gs[0, 2])
ids_nn_fine = 10 ** nn_model.predict(vgs_fine)
ax3.semilogy(vgs_all, ids_all*1e9, 'o', color=C[0], ms=2.5, alpha=0.4, label='Data')
ax3.semilogy(vgs_fine.ravel(), ids_nn_fine*1e9, color=C[3], lw=2.2,
             label=f'NN Fit (R²={r2_nn:.4f})')
ax_style(ax3, 'Neural Network Model Fit', 'V$_{GS}$ (V)', 'I$_{DS}$ (nA)')
ax3.legend(fontsize=7.5, framealpha=0.2)

# Plot 4: EPSC Gain
ax4 = fig.add_subplot(gs[1, 0])
freq_fine_log = np.linspace(np.log10(freq_vals.min()), np.log10(freq_vals.max()), 200).reshape(-1,1)
gain_fine     = 10 ** gain_model.predict(freq_fine_log)
ax4.semilogy(freq_vals, gain_vals, 'D', color=C[0], ms=8, markerfacecolor=C[4], label='Measured')
ax4.semilogy(10**freq_fine_log.ravel(), gain_fine, color=C[1], lw=2,
             label=f'Model (R²={r2_gain:.4f})')
ax_style(ax4, 'EPSC Gain vs Frequency', 'Frequency (Hz)', 'EPSC Gain')
ax4.legend(fontsize=8, framealpha=0.2)

# Plot 5: Parity plot (Poly)
ax5 = fig.add_subplot(gs[1, 1])
y_pred_all = poly_model.predict(X)
ax5.scatter(y, y_pred_all, color=C[1], s=8, alpha=0.5)
lims = [y.min(), y.max()]
ax5.plot(lims, lims, 'w--', lw=1.2, alpha=0.6)
ax_style(ax5, f'Parity Plot — Poly (R²={r2_poly:.4f})',
         'Measured log₁₀(I$_{DS}$)', 'Predicted log₁₀(I$_{DS}$)')

# Plot 6: Parameter Table
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
rows = [
    ['ION',         f'{ion:.3e} A'],
    ['IOFF',        f'{ioff:.3e} A'],
    ['ION/IOFF',    f'{ratio:.2e}'],
    ['SS',          f'{ss_mv:.1f} mV/dec'],
    ['Vth',         f'{vth:.3f} V'],
    ['gm max',      f'{np.nanmax(gm):.3e} A/V'],
    ['Poly R²',     f'{r2_poly:.5f}'],
    ['NN R²',       f'{r2_nn:.5f}'],
    ['Gain R²',     f'{r2_gain:.5f}'],
]
tbl = ax6.table(cellText=rows, colLabels=['Parameter', 'Value'],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (r, cc), cell in tbl.get_celld().items():
    cell.set_edgecolor('#2a2a4a')
    cell.set_facecolor('#0c0c22' if r % 2 == 0 else '#111133')
    cell.set_text_props(color='white')
    if r == 0:
        cell.set_facecolor('#1a1a4a')
        cell.set_text_props(color=C[0], fontweight='bold')
tbl.scale(1, 1.8)
ax6.set_title('Device Parameters', color='white', fontsize=10, fontweight='bold', pad=8)

fig.suptitle('IGZO/MgO Synaptic Transistor — Training Results',
             color='white', fontsize=13, fontweight='bold', y=0.97)

plt.savefig(os.path.join(MODEL_DIR, 'training_results.png'), dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())

print("\n" + "=" * 62)
print("  TRAINING COMPLETE!")
print(f"  Polynomial Model R² : {r2_poly:.5f}")
print(f"  Neural Network R²   : {r2_nn:.5f}")
print(f"  EPSC Gain R²        : {r2_gain:.5f}")
print(f"  Models saved in     : models/")
print("=" * 62)

plt.show()
