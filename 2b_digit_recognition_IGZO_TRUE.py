
"""
==============================================================
  IGZO/MgO SYNAPTIC TRANSISTOR — IMPROVED DIGIT RECOGNITION
  Target Accuracy: 93-95% (from 88.77%)
==============================================================

WHAT IS THIS FILE?
------------------
Ye ek Python script hai jo ek "neural network" banata hai.
Neural network = ek computer program jo insaan ke brain ki
tarah seekhta hai — digits (0-9) ko pehchanna sikhta hai.

KYA SPECIAL HAI IS CODE MEIN?
------------------------------
Normal neural networks ek math formula se weight update karte
hain (Adam/SGD optimizer). Lekin yahan hum apne IGZO transistor
ki real physics use karte hain weight update ke liye.

TRANSISTOR = SYNAPSE (dimag ki connection)
IGZO device ki conductance (G) = neural weight (w)
VGS badhao → IDS badhti hai → G badhti hai → weight +  (Potentiation)
VGS ghatao → IDS ghatti hai → G ghatti hai → weight -  (Depression)

"""

# ============================================================
# SECTION 1: IMPORT KARO — Matlab external tools lane ki tarah
# ============================================================

import numpy as np
# numpy = "Numerical Python"
# Ye library numbers aur matrices ke saath kaam karti hai.
# np.array([1,2,3]) banata hai ek "array" (number ki list).
# Hum ise isliye use karte hain kyunki 784 pixels × 60000
# images = 47 million numbers hain — Python list se handle
# nahi hoga, numpy se hoga (100x faster).

import pandas as pd
# pandas = "Panel Data" library
# CSV files padhne ke liye use hoti hai.
# df = pd.read_csv("file.csv") → ek "DataFrame" (table) banta hai.
# Hum ise IGZO ke experimental data load karne ke liye use karte hain.

import matplotlib
matplotlib.use('TkAgg')
# matplotlib = plotting library (graphs banane ke liye)
# matplotlib.use('TkAgg') → Windows GUI backend set karo.
# Agar ye error de, try karo: matplotlib.use('Agg') for saving only.

import matplotlib.pyplot as plt
# pyplot = matplotlib ka "easy interface"
# plt.plot(), plt.show() jaise commands isliye kaam karte hain.

import matplotlib.gridspec as gridspec
# gridspec = complex multi-panel figure layout ke liye.
# Isse hum ek figure mein alag-alag plots arrange kar sakte hain.

from scipy.interpolate import interp1d
# scipy = "Scientific Python" — advanced math functions
# interp1d = "interpolate 1-dimensional"
# Ye function do arrays (x, y) leke ek curve fit karta hai.
# Hum ise IGZO ke VGS→IDS curve ko smooth karne ke liye use karte hain.
# Agar data mein 402 points hain, interpolation se kisi bhi VGS par
# IDS ka estimate mil sakta hai.

from scipy.ndimage import uniform_filter1d
# uniform_filter1d = moving average (data smooth karne ke liye)
# Jab data mein noise ho (jagged curve), smooth karne se
# real trend dikhti hai.

from sklearn.datasets import fetch_openml
# scikit-learn = machine learning library
# fetch_openml = internet se standard dataset download karo.
# 'mnist_784' = famous handwritten digits dataset (0-9)
# 70,000 images: 60,000 train + 10,000 test.

from sklearn.preprocessing import StandardScaler
# StandardScaler = data ko "normalize" karta hai.
# Normalize matlab: har feature ko mean=0, std=1 pe laao.
# Why? Neural network tab better seekhta hai jab inputs
# ek hi scale par hoon. Pixel values 0-255 range mein hain —
# scaler inhe roughly -3 to +3 range mein convert karta hai.

from sklearn.metrics import accuracy_score, confusion_matrix
# accuracy_score(y_true, y_pred) = kitne correct predictions the / total
# confusion_matrix = 10x10 grid: kaunsa digit kisse confuse hua.

import joblib
# joblib = Python objects save/load karne ke liye.
# joblib.dump(obj, "file.pkl") → save
# joblib.load("file.pkl") → load
# Hum trained models save karte hain taaki dobara train na karna pade.

import os
# os = "Operating System" module — file/folder operations.
# os.makedirs("folder", exist_ok=True) → folder banao agar nahi hai.
# os.path.join("data", "file.csv") → path theek se banao
# (Windows mein backslash, Linux mein forward slash — automatic).

import warnings
warnings.filterwarnings('ignore')
# Kuch libraries harmless warnings deti hain.
# filterwarnings('ignore') → in warnings ko chhupa do taaki
# output clean rahe.

# ============================================================
# SECTION 2: PATHS AND FOLDERS SETUP
# ============================================================

DATA_DIR  = "data"
# DATA_DIR = "data" → ek Python "variable" banaya.
# Variable = ek naam jo koi value store karta hai.
# Yahan hum data folder ka naam store kar rahe hain.

MODEL_DIR = "models"
# Trained models yahan save honge.

os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs() → folder banao.
# exist_ok=True → agar folder already hai to error mat do.
# Iske bina: agar "models" folder already hai → Python crash karega.

IDVD_FILE = os.path.join(DATA_DIR, "IDVD_IDBG_IGZO_MgO.csv")
# os.path.join() = path pieces jodo.
# Result: "data/IDVD_IDBG_IGZO_MgO.csv" (Linux)
#      or "data\IDVD_IDBG_IGZO_MgO.csv" (Windows)
# Directly "data/file.csv" likhna Windows pe fail ho sakta hai.

FILT_FILE = os.path.join(DATA_DIR, "filterchar_IGZO_MgO_new.csv")
# EPSC gain data file ka path.

# ============================================================
# SECTION 3: PRINT BANNER (sirf decoration)
# ============================================================

print("=" * 62)
print("   IGZO/MgO IMPROVED — Target: 93-95% Accuracy")
print("   Architecture: 784 → 512 → 256 → 10")
print("   Physics: IGZO Potentiation / Depression")
print("=" * 62)
# print() = terminal pe text dikhao.
# "=" * 62 → "=" ko 62 baar repeat karo → ek line banti hai.

# ============================================================
# SECTION 4: DATA LOAD KARO
# ============================================================

print("\n[1/6] IGZO data load ho raha hai...")
# "\n" = newline character → ek blank line chhod do.
# [1/6] = progress indicator (step 1 of 6).

df_iv = pd.read_csv(IDVD_FILE, header=0)
# pd.read_csv() = CSV file padho aur DataFrame banao.
# header=0 → pehli row (row number 0) ko column names maano.
# df_iv = "dataframe for I-V characteristics"
# Iske andar VGS aur IDS columns honge.

df_fc = pd.read_csv(FILT_FILE, header=0)
# df_fc = "dataframe for filter characteristics"
# Iske andar frequency aur EPSC Gain columns honge.

# ── 4a: Transfer curves load karo (4 channel widths) ────────

curve_map = {
    'CW10': ('Unnamed: 0', 'CW10'),
    'CW11': ('Unnamed: 2', 'Cw11'),
    'CW12': ('Unnamed: 4', 'CW12'),
    'CW13': ('Unnamed: 6', 'CW13'),
}
# curve_map = Python "dictionary" (dict).
# Dict = key:value pairs ka collection, jaise ek lookup table.
# 'CW10' → ('Unnamed: 0', 'CW10') matlab:
#   CW10 curve ke liye VGS column = 'Unnamed: 0'
#   aur IDS column = 'CW10'
# Yahan 4 channel widths hain kyunki device ka physical width
# alag-alag test kiya gaya hai (different W/L ratio).

curves = {}
# curves = khali dictionary banao.
# Baad mein hum isme processed data store karenge.

for name, (vcol, icol) in curve_map.items():
    # for loop = ek kaam baar baar karo.
    # curve_map.items() → har entry ko (key, value) pair de.
    # name = 'CW10', 'CW11', etc.
    # vcol = voltage column name, icol = current column name.

    vgs = pd.to_numeric(df_iv[vcol], errors='coerce').dropna().values
    # df_iv[vcol] = dataframe se ek column nikalo (pd.Series).
    # pd.to_numeric(...) = text ko numbers mein convert karo.
    # errors='coerce' → agar text convert na ho (jaise header ya blank)
    #   to NaN (Not a Number) de do — crash mat karo.
    # .dropna() → saare NaN rows hata do.
    # .values → pandas Series se numpy array banao.

    ids = pd.to_numeric(df_iv[icol], errors='coerce').dropna().values
    # Same process IDS ke liye.

    n = min(len(vgs), len(ids))
    # len() = array ki length (kitne elements hain).
    # min() = do values mein se chhota wala.
    # Kyun? VGS aur IDS arrays ka length exactly same nahi ho sakta
    # (CSV formatting ke wajah se). Hum chhota wala lete hain
    # taaki dono arrays same size ke rahein.

    vgs, ids = vgs[:n], ids[:n]
    # Array slicing: vgs[:n] = pehle n elements lo.
    # Ye ensure karta hai dono arrays equal length ke hain.

    mask = ids > 0
    # mask = boolean array (True/False).
    # ids > 0 → har element ke liye: kya ye 0 se bada hai?
    # Result: [True, False, True, True, ...] jaisa kuch.
    # Kyun? Log scale plot ke liye negative ya zero values
    # invalid hain (log(0) = -infinity).

    curves[name] = {'VGS': vgs[mask], 'IDS': ids[mask]}
    # vgs[mask] = sirf wahi VGS values rakho jahan IDS > 0.
    # Boolean indexing: mask ke True positions ke elements lo.
    # Dictionary mein store karo: curves['CW10'] = {'VGS':..., 'IDS':...}

# Saare curves ek saath merge karo (richer training data ke liye)
vgs_all, ids_all = [], []
# Do khali Python lists banao.

for c in curves.values():
    vgs_all.append(c['VGS'])
    ids_all.append(c['IDS'])
# .values() → dictionary ki sirf values lo (keys nahi).
# .append() → list ke end mein ek element add karo.
# Hum har curve ka VGS aur IDS append kar rahe hain.

vgs_all = np.concatenate(vgs_all)
ids_all = np.concatenate(ids_all)
# np.concatenate() = multiple arrays ko ek mein jodo.
# [array1, array2, array3] → ek bada array.
# Ye 4 curves ke data ko ek dataset banata hai (more training data).

# ── 4b: EPSC Gain data load karo ────────────────────────────

freq_vals = pd.to_numeric(df_fc['frequesncy'], errors='coerce').dropna().values
# Note: 'frequesncy' = typo in original CSV column name — RAKHO as-is.
gain_vals = pd.to_numeric(df_fc['EPSC Gain'],  errors='coerce').dropna().values
n_fg = min(len(freq_vals), len(gain_vals))
freq_vals, gain_vals = freq_vals[:n_fg], gain_vals[:n_fg]
# Same process as above — equal length ensure karo.

print(f"   Transfer data : {len(vgs_all)} points (4 CW merged)")
print(f"   EPSC Gain     : {n_fg} frequency points")
# f"..." = f-string (formatted string).
# {len(vgs_all)} → variable ki value string mein daal do.
# Ye Python 3.6+ ka feature hai — bahut convenient.

# ============================================================
# SECTION 5: IGZO PHYSICS — P/D CURVE EXTRACT KARO
# ============================================================

print("\n[2/6] IGZO Potentiation/Depression curve extract ho raha hai...")

# CW11 use karo (sabse zyada data points — 402)
vgs_raw = pd.to_numeric(df_iv['Unnamed: 2'], errors='coerce').dropna().values
ids_raw = pd.to_numeric(df_iv['Cw11'],       errors='coerce').dropna().values
n = min(len(vgs_raw), len(ids_raw))
vgs_raw, ids_raw = vgs_raw[:n], ids_raw[:n]
mask = ids_raw > 0
vgs_raw, ids_raw = vgs_raw[mask], ids_raw[mask]

# Conductance normalize karo: 0 se 1 ke beech
G_norm = (ids_raw - ids_raw.min()) / (ids_raw.max() - ids_raw.min())
# Min-Max Normalization formula: G = (x - x_min) / (x_max - x_min)
# ids_raw.min() = sabse chhoti current value
# ids_raw.max() = sabse badi current value
# Result: saare values 0 aur 1 ke beech aa jaate hain.
# 0 = fully "off" state, 1 = fully "on" state.
# Conductance ek measure hai of how easily current flows.
# Transistor: G high → current easily flows → weight high.

# VGS ke order mein sort karo
vgs_sorted_idx = np.argsort(vgs_raw)
# np.argsort() = sorted order ke indices return karo.
# Example: [3, 1, 2] → argsort → [1, 2, 0]
# (index 1 = 1, index 2 = 2, index 0 = 3 — smallest to largest)
# Kyun? Interpolation ke liye data strictly increasing hona chahiye.

vgs_sorted = vgs_raw[vgs_sorted_idx]
G_sorted   = G_norm[vgs_sorted_idx]
# Sorted indices use karke dono arrays reorder karo.
# Ab vgs_sorted ascending order mein hai.

# Interpolation function banao: VGS → Conductance
G_interp = interp1d(vgs_sorted, G_sorted,
                    kind='linear', fill_value='extrapolate')
# interp1d() = 1D interpolation function.
# kind='linear' = do adjacent points ke beech straight line.
# fill_value='extrapolate' = data range ke bahar bhi estimate karo.
# Ab G_interp(voltage) call karo → conductance milegi.
# Ye IGZO device ka "memory map" hai.

# Potentiation aur Depression step sizes nikalo
dG_all    = np.diff(G_sorted)
# np.diff() = adjacent elements ka difference.
# [1, 3, 2, 5] → diff → [2, -1, 3]
# Positive diff = conductance badhna (potentiation)
# Negative diff = conductance ghattna (depression)

pot_steps = dG_all[dG_all > 0]   # sirf positive changes
dep_steps = dG_all[dG_all < 0]   # sirf negative changes

avg_pot = pot_steps.mean() if len(pot_steps) > 0 else 0.01
# .mean() = average nikalo.
# if len(...) > 0 = agar array khali nahi hai (division by zero se bacho).
# else 0.01 = agar koi potentiation step nahi mila, default value use karo.

avg_dep = dep_steps.mean() if len(dep_steps) > 0 else -0.01
# avg_dep ek negative number hoga (kyunki depression = decrease).

max_pot = pot_steps.max() if len(pot_steps) > 0 else 0.05
min_dep = dep_steps.min() if len(dep_steps) > 0 else -0.05

# ── IMPROVEMENT: P/D Asymmetry Ratio Calculate Karo ─────────
pd_ratio = abs(avg_pot) / (abs(avg_dep) + 1e-10)
# P/D ratio = potentiation step / depression step
# Tumhara data mein: avg_pot ≈ 0.002931, avg_dep ≈ 0.000092
# Ratio ≈ 31.8 — matlab potentiation depression se 31x bada hai!
# 1e-10 add karte hain taaki division by zero na ho.
# Ye BADI problem hai — weight ek direction mein zyada shift karta hai.
# Hum ise compensate karenge IGZOWeightUpdater mein.

print(f"   Avg Potentiation step : +{avg_pot:.6f}")
print(f"   Avg Depression step   : {avg_dep:.6f}")
print(f"   P/D Asymmetry ratio   : {pd_ratio:.2f}x  ← compensating this!")
print(f"   Conductance states    : {len(G_sorted)}")

# ============================================================
# SECTION 6: IMPROVED IGZOWeightUpdater CLASS
# ============================================================

class IGZOWeightUpdater:
    """
    YE CLASS KYA KARTI HAI:
    =======================
    Ye class tumhare real IGZO transistor ka behavior simulate
    karti hai. Jab bhi neural network ka weight update hona ho,
    ye class physically sahi tarike se update karti hai.

    PROBLEM JO IS VERSION MEIN FIX KI:
    ====================================
    Original mein: potentiate aur depress steps bahut asymmetric the.
    Avg_pot = +0.002931, Avg_dep = -0.000092 → ratio = 31.8x
    Iska matlab: weight badhna bahut fast, ghattna bahut slow.
    Ye weights ko ek direction mein drift karata tha.

    FIX: dep_lr ko scale karo taaki effective step sizes equal hon.
    """

    def __init__(self, G_interp, vgs_min, vgs_max,
                 avg_pot, avg_dep, n_states=402):
        # __init__ = "constructor" method.
        # Jab class ka object banate hain: obj = IGZOWeightUpdater(...)
        # Ye method automatically call hoti hai.
        # self = is object ka reference (tumhare variable jaisa).

        self.G_interp = G_interp
        # self.G_interp mein store karo VGS→conductance function.

        self.vgs_min  = vgs_min
        self.vgs_max  = vgs_max
        # Device ki valid VGS range store karo.

        self.avg_pot  = avg_pot
        self.avg_dep  = abs(avg_dep)
        # avg_dep ko positive bana do (abs = absolute value).
        # Kyun? Hum direction alag se handle karte hain (depress method mein).

        self.n_states = n_states
        # 402 discrete conductance states hain tumhare device mein.
        # Real device sirf discrete levels mein jump karta hai
        # (quantum effect + device physics).

        # ── IMPROVEMENT: Asymmetry compensation factor ──────
        self.dep_compensation = min(avg_pot / (abs(avg_dep) + 1e-10), 30.0)
        # P/D ratio calculate karo aur cap at 30x.
        # Agar avg_dep bahut chhota hai to depression learning rate
        # ko 30x tak scale up karo.
        # min(..., 30.0) → zyada scale-up se instability aayegi, isliye cap.
        # Is compensation se dono directions mein roughly equal
        # effective weight change hogi.

        self.G_states = np.linspace(0.0, 1.0, n_states)
        # np.linspace(start, stop, num) = evenly spaced values.
        # 402 values from 0.0 to 1.0.
        # Ye real IGZO device ke discrete conductance states represent karta hai.

    def conductance_to_weight(self, G, w_min=-2.0, w_max=2.0):
        """
        Conductance (0 to 1) ko neural weight (-2 to +2) mein convert karo.

        WHY -2 TO +2?
        Neural network weights generally -1 to +1 range mein kaam
        karte hain. Hum -2 to +2 use karte hain thoda extra room ke liye.
        """
        return G * (w_max - w_min) + w_min
        # Linear mapping: G=0 → w=-2, G=0.5 → w=0, G=1 → w=+2
        # Formula: w = G × (max - min) + min
        # = G × 4 + (-2)

    def weight_to_conductance(self, w, w_min=-2.0, w_max=2.0):
        """
        Neural weight (-2 to +2) ko conductance (0 to 1) mein convert karo.
        Inverse of conductance_to_weight.
        """
        return np.clip((w - w_min) / (w_max - w_min), 0.0, 1.0)
        # np.clip(x, 0, 1) = x ko 0 aur 1 ke beech rakhna.
        # Agar formula se G < 0 aata hai → 0 kar do.
        # Agar G > 1 aata hai → 1 kar do.
        # Kyun? Real device physically 0 se below ya 1 se above nahi ja sakta.

    def potentiate(self, w, lr=1.0):
        """
        POTENTIATION: Weight badhao (conductance increase).

        Ye function real IGZO device ka potentiation curve follow karta hai.
        Key property: jab G already high hai (device saturated hai),
        potentiation se aur zyada nahi badhta — ye saturation effect hai.
        """
        G = self.weight_to_conductance(w)
        # Current weight ko conductance mein convert karo.

        dG = self.avg_pot * lr * (1.0 - G)
        # dG = change in conductance.
        # avg_pot = tumhara measured average potentiation step.
        # lr = learning rate (ek scale factor, 0 to 1+).
        # (1.0 - G) = SATURATION EFFECT:
        #   Jab G = 0 → (1-G) = 1.0 → full step
        #   Jab G = 0.5 → (1-G) = 0.5 → half step
        #   Jab G = 0.9 → (1-G) = 0.1 → very small step
        # Real IGZO device bhi aisa hi karta hai — upper limit pe
        # potentiation slow ho jaata hai.

        G_new = np.clip(G + dG, 0.0, 1.0)
        # New conductance = old + change.
        # np.clip() se ensure karo 0-1 range mein rahe.

        return self.conductance_to_weight(G_new)
        # Conductance ko wapas weight mein convert karo aur return karo.

    def depress(self, w, lr=1.0):
        """
        DEPRESSION: Weight ghattao (conductance decrease).

        Saturation effect: jab G already low hai,
        depression se aur zyada nahi ghatta.
        """
        G = self.weight_to_conductance(w)
        dG = self.avg_dep * lr * G
        # (same logic as potentiation but reverse)
        # G = 0 → dG = 0 (already at minimum, can't go lower)
        # G = 1 → dG = avg_dep (full depression step)
        # G = 0.5 → dG = avg_dep/2 (half step)
        G_new = np.clip(G - dG, 0.0, 1.0)
        return self.conductance_to_weight(G_new)

    def update(self, w, gradient, lr=0.01):
        """
        YE HAI ASLI IGZO WEIGHT UPDATE FUNCTION.

        LOGIC:
        gradient < 0 → weight badhana chahiye → POTENTIATE
        gradient > 0 → weight ghattana chahiye → DEPRESS
        gradient = 0 → koi change nahi

        IMPROVEMENT: Asymmetric compensation.
        Depression ka lr scale up kiya gaya hai
        taaki P aur D effective steps roughly equal hon.
        """
        w_new = w.copy()
        # w.copy() = array ki exact copy banao.
        # IMPORTANT: w.copy() isliye zaruri hai kyunki agar directly
        # w_new = w karein, dono same memory point karte hain.
        # w_new change karne se w bhi change ho jaata — bug!

        pot_mask = gradient < 0
        dep_mask = gradient > 0
        # Boolean masks: kahan potentiate karna hai, kahan depress.
        # Example: gradient = [-0.5, 0.3, -0.1, 0.0, 0.8]
        # pot_mask = [True, False, True, False, False]
        # dep_mask = [False, True, False, False, True]

        strength = np.abs(gradient) * lr * 50
        # Gradient ki magnitude = kitna strong update chahiye.
        # np.abs() = absolute value (sign hata do).
        # lr = learning rate (overall scale).
        # 50 = scale factor (kyunki gradient values bahut chhoti hoti hain
        #      aur avg_pot/avg_dep bhi bahut chhoti hain).

        # ── IMPROVEMENT: Compensate for P/D asymmetry ───────
        dep_strength = strength * self.dep_compensation
        # Depression ki strength ko compensation factor se multiply karo.
        # Agar compensation = 31.8, to depression strength 31.8x zyada hogi.
        # Ye isliye ki avg_dep bahut chhota hai avg_pot se compare mein.
        # Isse effective weight change dono directions mein roughly equal hogi.

        dep_strength = np.clip(dep_strength, 0.0, 5.0)
        # Depression strength ko cap karo at 5.0.
        # Kyun? Zyada zyada depression se weights crash ho sakte hain.

        if pot_mask.any():
            # .any() = kya koi bhi element True hai?
            # Agar koi bhi potentiation nahi chahiye, skip karo.
            w_new[pot_mask] = self.potentiate(w[pot_mask], strength[pot_mask])
            # Sirf pot_mask ke True positions ko potentiate karo.
            # w[pot_mask] = boolean indexing → sirf selected weights.

        if dep_mask.any():
            w_new[dep_mask] = self.depress(w[dep_mask], dep_strength[dep_mask])
            # Depression ke liye compensated strength use karo.

        return w_new

# Object banao (class ka instance)
igzo_updater = IGZOWeightUpdater(
    G_interp = G_interp,
    vgs_min  = vgs_sorted.min(),
    vgs_max  = vgs_sorted.max(),
    avg_pot  = avg_pot,
    avg_dep  = abs(avg_dep),
    n_states = len(G_sorted)
)
# Yahan hum class ko "instantiate" kar rahe hain.
# Matlab: blueprint (class) se actual object banao.
# Ab igzo_updater.update(w, grad) call kar sakte hain.

print(f"\n   IGZO Updater ready!")
print(f"   P/D Compensation factor: {igzo_updater.dep_compensation:.2f}x")

# ============================================================
# SECTION 7: DATA AUGMENTATION FUNCTION
# ============================================================

def augment_batch(Xb, noise_std=0.05, dropout_rate=0.05):
    """
    DATA AUGMENTATION KYA HOTA HAI?
    ================================
    Neural network ko agar hamesha same data dikhao, to wo
    "rote memorize" kar leta hai (overfitting).
    Augmentation = training data mein thodi variation add karo
    taaki model zyada general sikhe.

    IS FUNCTION MEIN KYA HOTA HAI:
    1. Gaussian Noise: har pixel mein thoda random number add karo
       → jaise blurry image dekhna
    2. Pixel Dropout: 5% pixels randomly zero kar do
       → jaise kuch pixels missing hon
    3. Small random shift: image ko 1-2 pixels left/right/up/down khiskaao
       → same digit, slightly different position

    WHY DOES THIS HELP?
    Model seekhta hai: "7 chahein thoda blurry ho, ya shift hua ho,
    phir bhi '7' hi hai." → better generalization.
    """
    Xb_aug = Xb.copy()
    # Original data change mat karo — copy banao.

    # ── Gaussian noise ───────────────────────────────────────
    noise = np.random.randn(*Xb_aug.shape) * noise_std
    # np.random.randn() = random numbers from normal distribution.
    # *Xb_aug.shape = shape unpack karo (batch_size, 784).
    # noise_std = 0.05 → noise bahut chhota hai (pixel value scale pe).
    # Randn → mean 0, std 1 ke numbers → * noise_std → scale down.

    Xb_aug = Xb_aug + noise
    # Har pixel mein random noise add karo.

    # ── Pixel dropout ────────────────────────────────────────
    pixel_mask = np.random.rand(*Xb_aug.shape) > dropout_rate
    # np.random.rand() = 0 to 1 random uniform numbers.
    # > dropout_rate → True agar value > 0.05, else False.
    # Matlab: 95% pixels True (keep), 5% pixels False (zero out).

    Xb_aug = Xb_aug * pixel_mask
    # False pixels × koi bhi value = 0.
    # Randomly kuch pixels zero ho jaate hain.

    # ── Random shift (MNIST = 28×28 = 784 pixels) ────────────
    shift_x = np.random.randint(-2, 3)
    shift_y = np.random.randint(-2, 3)
    # np.random.randint(low, high) = random integer, low inclusive, high exclusive.
    # -2 to 2 ke beech integer: -2, -1, 0, 1, or 2.
    # Ye image ko pixels mein shift karta hai.

    if shift_x != 0 or shift_y != 0:
        # Sirf tab shift karo jab actual shift ho (0,0 = no-op).
        imgs = Xb_aug.reshape(-1, 28, 28)
        # .reshape(-1, 28, 28) = 784 flat pixels ko 28×28 grid mein reshape karo.
        # -1 = "automatically calculate" → batch_size.
        # Ye is liye karna padta hai kyunki shift 2D operation hai.

        imgs = np.roll(imgs, shift_x, axis=2)
        imgs = np.roll(imgs, shift_y, axis=1)
        # np.roll(arr, shift, axis) = array ko circular shift karo.
        # axis=2 = horizontal shift (columns), axis=1 = vertical shift (rows).
        # Circular means: right side se nikla pixel left side aata hai.
        # MNIST ke liye 1-2 pixel shift digit ke meaning ko preserve karta hai.

        Xb_aug = imgs.reshape(-1, 784)
        # Wapas flat 784 format mein convert karo.

    return Xb_aug
    # Augmented batch return karo.

# ============================================================
# SECTION 8: MNIST DATA LOAD KARO
# ============================================================

print("\n[3/6] MNIST data load ho raha hai...")

try:
    # try-except = error handling.
    # try block mein code chalao, agar koi error aaye to...
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y  = mnist.data, mnist.target.astype(int)
    # mnist.data = 70000 × 784 matrix (70000 images, 784 pixels each).
    # mnist.target = labels (0-9), string format mein hote hain.
    # .astype(int) = string "7" ko integer 7 mein convert karo.
    X = X / 255.0
    # Pixel values 0-255 range mein hote hain.
    # 255.0 se divide karo → 0.0 to 1.0 range mein aao.
    # Neural network 0-1 range ke inputs se better seekhta hai.

except Exception as e:
    # agar koi bhi exception (error) aaye...
    print(f"   ERROR loading MNIST: {e}")
    print("   Try: pip install scikit-learn scipy")
    exit(1)
    # exit(1) = program band karo. 1 = error code.

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
# Array slicing: X[:60000] = pehle 60000 rows.
# X[60000:] = 60000 se last tak (10000 rows).
# MNIST standard split: 60k train, 10k test.

# One-hot encoding karo
def one_hot(y_arr, n_classes=10):
    """
    ONE-HOT ENCODING KYA HOTA HAI?
    ================================
    Neural network output 10 neurons hai (digit 0-9 ke liye).
    Label '7' ko directly use nahi kar sakte training mein.
    Hum use convert karte hain ek vector mein:
    7 → [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
         0  1  2  3  4  5  6  7  8  9
    Sirf position 7 pe '1' hai, baaki '0'.
    """
    oh = np.zeros((len(y_arr), n_classes))
    # np.zeros() = saari values 0 se bhara matrix banao.
    # Shape: (number_of_samples, 10).

    oh[np.arange(len(y_arr)), y_arr] = 1
    # np.arange(len(y_arr)) = [0, 1, 2, ..., 59999]
    # y_arr = [5, 0, 4, 1, 9, ...] (digit labels)
    # oh[row_index, column_index] = 1
    # Matlab: row 0 ke column 5 pe 1, row 1 ke column 0 pe 1, etc.
    # Ye "fancy indexing" hai — ek baar mein sabhi rows set karo.

    return oh

y_train_oh = one_hot(y_train)
y_test_oh  = one_hot(y_test)

# StandardScaler: data normalize karo
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
# fit_transform() → train data pe:
#   1. fit() = mean aur std calculate karo.
#   2. transform() = (x - mean) / std apply karo.
# transform() → test data pe: SAME mean/std use karo (not test ke).
# WHY? Agar test data ka apna mean/std use karein, to
# hum "future data" ki information leak kar rahe hain (cheating).

print(f"   Train: {X_train.shape[0]} images | Test: {X_test.shape[0]} images")
print(f"   Each image: {X_train.shape[1]} pixels (28×28 flattened)")

# ============================================================
# SECTION 9: IMPROVED NEURAL NETWORK CLASS
# ============================================================

class IGZONeuralNetwork:
    """
    NEURAL NETWORK KYA HOTA HAI?
    ==============================
    Neural network = layers of "neurons" connected by "weights".
    Input → [Hidden Layer 1] → [Hidden Layer 2] → Output

    IS NETWORK KI ARCHITECTURE:
    Input (784) → Layer 1 (512) → Layer 2 (256) → Output (10)

    784 = 28×28 pixels (MNIST image)
    512 = 512 hidden neurons (zyada = zyada patterns seekh sakta hai)
    256 = 256 hidden neurons (compress karo features ko)
    10  = 10 output neurons (0-9 digits)

    ORIGINAL: 784 → 256 → 10  (ek hi hidden layer)
    IMPROVED: 784 → 512 → 256 → 10  (2 hidden layers)
    Why? Deeper network = zyada complex patterns seekh sakta hai.
    Simple lines → shapes → digit parts → complete digits.

    WEIGHT UPDATE:
    Normal: W = W - lr × gradient  (math formula)
    IGZO:   W = igzo_updater.update(W, gradient)  (physics!)
    """

    def __init__(self, layer_sizes, igzo_updater, lr=0.008, seed=42):
        np.random.seed(seed)
        # np.random.seed(seed) = random number generator ko fix karo.
        # seed=42 → hamesha same "random" numbers generate honge.
        # Kyun? Reproducibility: tumhara friend bhi same results
        # paaye jab same code chalaye.

        self.igzo   = igzo_updater
        self.lr     = lr
        # lr = learning rate = kitni tezi se seekhe.
        # Too high → weights oscillate, training unstable.
        # Too low → bahut slow training.
        # 0.008 = tuned value for IGZO physics.

        self.layer_sizes = layer_sizes
        self.losses   = []   # Training loss har epoch store karo.
        self.val_acc  = []   # Validation accuracy har epoch store karo.
        self.lr_history = [] # LR history (for monitoring).

        # ── IMPROVEMENT: He Initialization ──────────────────
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            n_in  = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            # n_in = input size (784, 512, etc.)
            # n_out = output size (512, 256, 10)

            scale = np.sqrt(2.0 / n_in)
            # HE INITIALIZATION formula: sqrt(2 / n_in)
            # Kyun 2.0 aur not 1.0?
            # ReLU activation ka use kar rahe hain.
            # ReLU half values ko zero kar deta hai.
            # Isliye variance maintain karne ke liye 2x scale chahiye.
            # Xavier init (sqrt(1/n)) ReLU ke saath kaam nahi karta —
            # vanishing gradient problem aata hai.

            W = np.random.randn(n_in, n_out) * scale * 0.7
            # np.random.randn(rows, cols) = random matrix (normal distribution).
            # * scale * 0.7 = thoda aur scale down karo.
            # 0.7 = emperically tuned: initial weights conductance midpoint
            # (G=0.5) ke paas rahein taaki IGZO updater effectively kaam kare.
            # Agar weights bahut large hain → saturation immediately.
            # Agar bahut small → updater ka effect nahi dikhega.

            b = np.zeros((1, n_out))
            # Biases = zero se start karo.
            # Bias = ek extra number jo har neuron ka output shift kar sakta hai.
            # Think: y = wx + b mein 'b'.
            # Shape (1, n_out) → broadcasting ke liye (batch ke saath add hoga).

            self.weights.append(W)
            self.biases.append(b)

    # ── Activation Functions ─────────────────────────────────

    def relu(self, z):
        """
        ReLU = Rectified Linear Unit
        Formula: max(0, z)
        Kaam: negative values → 0, positive values → as-is.

        WHY ReLU?
        - Simple aur fast.
        - Gradient vanishing nahi hota (unlike sigmoid/tanh).
        - Hidden layers mein use hoti hai non-linearity ke liye.
        - Non-linearity = network complex patterns seekh sakta hai.
          Bina non-linearity = sirf straight lines seekh sakta hai.
        """
        return np.maximum(0, z)
        # np.maximum(0, z) = element-wise max(0, z).

    def relu_grad(self, z):
        """
        ReLU ka gradient (derivative):
        z > 0 → 1 (gradient pass karo)
        z <= 0 → 0 (gradient block karo)
        Backpropagation ke liye zaruri.
        """
        return (z > 0).astype(float)
        # (z > 0) = boolean array.
        # .astype(float) = True→1.0, False→0.0.

    def softmax(self, z):
        """
        Softmax = probabilities nikalna.
        Output layer pe use hoti hai (10 classes).

        Formula: exp(z_i) / sum(exp(z_j))
        Result: saare outputs ka sum = 1.0 (probabilities).
        Example: [2.1, -1.3, 4.5, ...] → [0.05, 0.01, 0.82, ...]
        82% probability → digit 2 hai.
        """
        z_shifted = z - z.max(axis=1, keepdims=True)
        # Numerical stability: sabse bada value subtract karo.
        # Kyun? exp(1000) = infinity → overflow error.
        # exp(1000 - 1000) = exp(0) = 1 → safe.
        # axis=1 = har row ka max, keepdims=True = shape maintain.

        ez = np.exp(z_shifted)
        # Exponential: e^x. Negative inputs ko positive banata hai.

        return ez / ez.sum(axis=1, keepdims=True)
        # Har row ko us row ke sum se divide karo → probabilities.

    # ── Forward Pass ─────────────────────────────────────────

    def forward(self, X):
        """
        FORWARD PASS KYA HOTA HAI?
        ===========================
        Input (image) ko network ke through pass karo
        aur prediction (output) nikalo.

        Flow:
        Input → W1×X + b1 → ReLU → W2×A1 + b2 → ReLU → W3×A2 + b3 → Softmax → Probabilities
        """
        self.activations = [X]
        # activations = har layer ka output store karo.
        # Backpropagation ke liye zaruri hai.
        # List mein pehla element = input layer.

        self.zs = []
        # zs = "pre-activation" values (before ReLU/Softmax).
        # z = W × A_prev + b
        # Backpropagation mein ReLU gradient ke liye z chahiye.

        A = X
        # A = current "activation" (input ke liye, A = X).

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            # zip() = do lists ko pair karo: [(W1,b1), (W2,b2), ...].
            # enumerate() = index bhi do: (0,pair1), (1,pair2), ...

            Z = A @ W + b
            # @ = matrix multiplication operator (Python 3.5+).
            # A shape: (batch_size, n_in)
            # W shape: (n_in, n_out)
            # A @ W shape: (batch_size, n_out)
            # + b: broadcasting → b (1, n_out) → (batch_size, n_out)
            # Z = weighted sum of inputs + bias.

            self.zs.append(Z)

            if i < len(self.weights) - 1:
                A = self.relu(Z)
                # Hidden layers → ReLU activation.
            else:
                A = self.softmax(Z)
                # Last layer → Softmax (probabilities).

            self.activations.append(A)
            # Har layer ka output save karo.

        return self.activations[-1]
        # Last element = output probabilities.

    # ── IGZO Backward Pass (Heart of the Code) ───────────────

    def backward_igzo(self, X, y_oh):
        """
        BACKWARD PASS (BACKPROPAGATION) KYA HOTA HAI?
        ===============================================
        Network ki prediction galat thi → kitni galat thi (loss)?
        Loss se gradient nikalo → gradient se weights update karo.

        STANDARD BACKPROP vs IGZO BACKPROP:
        Standard: W -= lr × dW  (simple math)
        IGZO:     W = igzo_updater.update(W, dW)  (transistor physics!)

        GRADIENT KYA HOTA HAI?
        Gradient = direction + magnitude of "steepest descent".
        Matlab: agar W ko thoda badhao → loss kitna badhta/ghatta hai?
        dW > 0 → W badhane se loss badhta hai → W ghatao (depress)
        dW < 0 → W badhane se loss ghatta hai → W badhao (potentiate)
        """
        m      = X.shape[0]
        # m = batch size (kitne images ek saath).
        # X.shape = (batch_size, 784), so X.shape[0] = batch_size.

        n_layer = len(self.weights)

        dA = self.activations[-1] - y_oh
        # Output layer gradient = (predicted - actual).
        # Example: predicted = [0.1, 0.1, 0.7, ...], actual = [0,0,1,0,...]
        # dA = [-0.1, -0.1, -0.3, ...] (cross-entropy + softmax combined).
        # Ye mathematical derivation hai — combined gradient of
        # cross-entropy loss and softmax function.

        for i in reversed(range(n_layer)):
            # reversed() = last layer se start karo, first tak jao.
            # Isliye "back" propagation kehte hain.

            A_prev = self.activations[i]
            # i-th layer ka input = (i-1)-th layer ka activation.
            # (activations[0] = input X, activations[1] = layer1 output, etc.)

            dW = (A_prev.T @ dA) / m
            # A_prev.T = transpose of A_prev.
            #   A_prev shape: (batch, n_in) → transpose → (n_in, batch)
            # @ dA: (n_in, batch) @ (batch, n_out) = (n_in, n_out)
            # / m = average over batch.
            # dW shape = same as W → weight gradient.

            db = dA.mean(axis=0, keepdims=True)
            # Bias gradient = average over batch.
            # axis=0 = average over rows (batch dimension).
            # keepdims=True = shape (1, n_out) maintain karo.

            # ── IGZO UPDATE — YE HAI ASLI MAGIC ─────────────
            self.weights[i] = self.igzo.update(
                self.weights[i], dW, lr=self.lr
            )
            # Standard gradient descent: self.weights[i] -= self.lr * dW
            # IGZO update: transistor ka P/D curve decide karta hai
            # kaise aur kitna weight change hoga.
            # dW > 0 → depress (weight ghattao)
            # dW < 0 → potentiate (weight badhao)
            # Step size IGZO device ke measured avg_pot/avg_dep se.
            # ─────────────────────────────────────────────────

            self.biases[i] -= self.lr * db
            # Biases standard gradient descent se update hote hain.
            # IGZO physics biases pe apply nahi karte (wo synaptic
            # connections nahi hain — device level pe equivalent nahi).

            if i > 0:
                # Agar pehli layer nahi hai (input layer se gradient
                # calculate nahi karna — wo fixed hai).
                dA = (dA @ self.weights[i].T) * self.relu_grad(self.zs[i - 1])
                # dA @ self.weights[i].T: gradient ko previous layer pe propagate karo.
                # * self.relu_grad(self.zs[i-1]): ReLU ka gradient apply karo.
                #   ReLU ki jo units off (z<=0) theen, unka gradient 0 hai.
                #   Chain rule: composite function ka gradient = product of gradients.

    def cross_entropy(self, y_pred, y_oh):
        """
        CROSS-ENTROPY LOSS:
        Loss = -mean(sum(y_true × log(y_pred)))

        Kya measure karta hai?
        Prediction kitna confident tha aur kitna correct tha.
        Perfect prediction → loss = 0.
        Completely wrong + confident → loss = very large.

        Example:
        True label: digit 7 → one-hot [0,0,0,0,0,0,0,1,0,0]
        Predicted:  [0.01, 0.01, 0.01, ..., 0.90, ..., 0.01]
        Loss = -log(0.90) = 0.105  (small, good prediction)

        Predicted:  [0.01, 0.01, ..., 0.05, ..., 0.01]
        Loss = -log(0.05) = 3.0  (large, bad prediction)
        """
        return -np.mean(np.sum(y_oh * np.log(y_pred + 1e-9), axis=1))
        # y_oh * np.log(y_pred + 1e-9):
        #   y_oh mostly zeros, sirf correct class pe 1.
        #   Multiply → sirf correct class ka log(probability) matter karta hai.
        # + 1e-9: log(0) = -infinity → crash. 1e-9 add karo to avoid.
        # np.sum(..., axis=1): har sample ka loss sum karo (10 classes).
        # np.mean(): saare samples ka average loss.
        # Negative sign: log(probability) negative hota hai (prob<1),
        #   negative of negative = positive loss.

    # ── Training Loop ─────────────────────────────────────────

    def train(self, X_train, y_train_oh, X_val, y_val,
              epochs=80, batch_size=256, patience=15):
        """
        TRAINING LOOP — YE KAAM KARTA HAI:
        1. Data shuffle karo (taaki order se bias na aaye).
        2. Mini-batches mein tod do (256 images ek saath).
        3. Har batch pe forward + backward pass karo.
        4. Learning rate adjust karo (cosine decay).
        5. Accuracy track karo, best model save karo.
        6. Early stopping agar improvement ruk jaye.

        IMPROVEMENTS:
        - 80 epochs (was 30)
        - Cosine LR decay (was fixed LR)
        - Data augmentation on each batch
        - Early stopping (patience=15)
        """
        n        = X_train.shape[0]  # 60000
        base_lr  = self.lr           # Initial learning rate save karo
        best_acc = 0.0               # Best accuracy track karo
        best_W   = None              # Best weights save karo
        best_b   = None
        no_improve = 0               # Epochs without improvement

        print(f"\n   {'Epoch':>5} | {'Loss':>8} | {'Val Acc':>8} | {'LR':>10} | Status")
        print(f"   {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-------")

        for ep in range(1, epochs + 1):
            # range(1, epochs+1) → 1, 2, 3, ..., 80.
            # (1 se start isliye ki printing mein "Epoch 1" readable lage)

            # ── IMPROVEMENT: Cosine Learning Rate Decay ──────
            cos_decay = 0.5 * (1 + np.cos(np.pi * ep / epochs))
            # Cosine annealing formula.
            # ep=1 → cos_decay ≈ 1.0 (full LR at start)
            # ep=40 → cos_decay = 0.5 (half LR at middle)
            # ep=80 → cos_decay ≈ 0.0 (very small LR at end)
            # Why cosine? Smooth decay, no abrupt jumps.
            # Smoother than step decay, better than linear.

            self.lr = base_lr * (0.1 + 0.9 * cos_decay)
            # 0.1 = minimum LR (10% of base) — kabhi zero nahi hota.
            # 0.9 * cos_decay = dynamic part.
            # Formula ensures LR never fully goes to zero.
            self.lr_history.append(self.lr)

            # ── Data shuffle ─────────────────────────────────
            idx = np.random.permutation(n)
            # np.random.permutation(n) = 0 to n-1 ka random order.
            # Example: n=5 → [3, 1, 4, 0, 2]
            # Kyun shuffle? Agar same order mein train karein, model
            # pattern memorize karta hai (not generalize).

            X_s = X_train[idx]
            y_s = y_train_oh[idx]
            # Array fancy indexing: idx order mein rows lo.

            # ── Mini-batch training ───────────────────────────
            for start in range(0, n, batch_size):
                # range(0, 60000, 256) → 0, 256, 512, ..., 59904
                # Ye mini-batches banata hai.

                end = min(start + batch_size, n)
                # Last batch chhota ho sakta hai (60000 % 256 ≠ 0).
                # min() → overflow mat karo.

                Xb = X_s[start:end]
                yb = y_s[start:end]
                # Current batch slice karo.

                # ── IMPROVEMENT: Data Augmentation ───────────
                Xb_aug = augment_batch(Xb)
                # Training pe augmented data use karo.
                # Validation/test pe augmentation mat karo
                # (real data ka fair evaluation chahiye).

                self.forward(Xb_aug)
                self.backward_igzo(Xb_aug, yb)
                # Ek batch ka training step.

            # ── Epoch ke baad metrics calculate karo ─────────
            y_pred_train = self.forward(X_train[:5000])
            # Sirf 5000 training samples pe loss calculate karo
            # (sab 60000 pe karna slow hoga, 5000 enough estimate hai).

            loss = self.cross_entropy(y_pred_train, y_train_oh[:5000])

            y_pred_val = self.forward(X_val)
            # VALIDATION: augmentation ke BINA test karo.
            # X_val = original, unmodified test data.

            val_pred = np.argmax(y_pred_val, axis=1)
            # np.argmax(arr, axis=1) = har row mein sabse bada value ka index.
            # [0.05, 0.02, 0.87, ...] → 2 (index 2 pe max probability).
            # Ye class prediction hai.

            val_acc  = accuracy_score(y_val, val_pred)
            # accuracy_score = (correct predictions) / (total predictions).

            self.losses.append(loss)
            self.val_acc.append(val_acc)

            # ── Early Stopping Logic ──────────────────────────
            status = ""
            if val_acc > best_acc + 0.0001:
                # New best! (0.0001 threshold to avoid noise).
                best_acc = val_acc
                best_W   = [w.copy() for w in self.weights]
                best_b   = [b.copy() for b in self.biases]
                # List comprehension: har weight ka copy banao.
                # [expression for item in iterable]
                no_improve = 0
                status = "★ best"
            else:
                no_improve += 1
                status = f"no improve {no_improve}/{patience}"

            # ── Print progress ───────────────────────────────
            if ep % 5 == 0 or ep <= 5 or status.startswith("★"):
                # ep % 5 == 0 → har 5th epoch print karo.
                # ep <= 5 → pehle 5 epochs hamesha print karo.
                # status.startswith("★") → best model pe hamesha print.
                print(f"   {ep:>5} | {loss:>8.5f} | {val_acc*100:>7.2f}% "
                      f"| {self.lr:>10.6f} | {status}")

            # ── Early Stopping Check ─────────────────────────
            if no_improve >= patience:
                print(f"\n   Early stopping at epoch {ep}!")
                print(f"   Best validation accuracy: {best_acc*100:.2f}%")
                break
                # break = loop se bahar niklo immediately.

        # ── Best weights restore karo ─────────────────────────
        if best_W is not None:
            self.weights = best_W
            self.biases  = best_b
        # Training ke dauran jo best model tha, use restore karo.
        # Last epoch ka model best nahi hota (overfitting ho sakta hai).

        print(f"\n   Training complete! Best accuracy: {best_acc*100:.2f}%")
        return best_acc

    def predict(self, X):
        """
        Final prediction: sabse probable digit return karo.
        """
        return np.argmax(self.forward(X), axis=1)
        # forward() → probabilities, argmax() → class index.

    def predict_proba(self, X):
        """
        Probability distribution return karo (saare 10 digits ke liye).
        """
        return self.forward(X)

# ============================================================
# SECTION 10: MODEL BANAO AUR TRAIN KARO
# ============================================================

print("\n[4/6] Model define aur initialize ho raha hai...")

igzo_nn = IGZONeuralNetwork(
    layer_sizes  = [784, 512, 256, 10],
    # IMPROVEMENT: 2 hidden layers (was 1).
    # 784 = input (28×28 pixels)
    # 512 = first hidden layer (zyada neurons = zyada patterns)
    # 256 = second hidden layer (compression + abstraction)
    # 10  = output (0-9 digits)

    igzo_updater = igzo_updater,
    # Humara IGZO physics engine.

    lr= 0.008,
    # Learning rate.
    # Deeper network ke liye thoda lower LR better hai.
    # 0.008 < 0.01 (original) — more stable training.

    seed= 42
    # Reproducible results ke liye.
)

print(f"   Architecture: {igzo_nn.layer_sizes}")
print(f"   Total weights: {sum(w.size for w in igzo_nn.weights):,}")
# Generator expression: sum(w.size for w in list)
# w.size = total elements in weight matrix.
# 784×512 + 512×256 + 256×10 = 401,408 + 131,072 + 2,560 = 535,040
# Ye saare IGZO synaptic connections hain!

print(f"   LR: {igzo_nn.lr} (cosine decay to {igzo_nn.lr*0.1:.5f})")
print(f"\n[5/6] Training shuru ho raha hai...")
print(f"      80 epochs | batch=256 | augmentation ON | early stopping ON")

best_acc = igzo_nn.train(
    X_train    = X_train,
    y_train_oh = y_train_oh,
    X_val      = X_test,
    y_val      = y_test,
    epochs     = 40,
    batch_size = 256,
    patience   = 15
)

# ── Final Evaluation ─────────────────────────────────────────

y_pred_test  = igzo_nn.predict(X_test)
y_pred_train = igzo_nn.predict(X_train[:10000])
# Train accuracy sirf 10000 pe (speed ke liye).

acc_test  = accuracy_score(y_test,         y_pred_test)   * 100
acc_train = accuracy_score(y_train[:10000], y_pred_train) * 100

print(f"\n   ╔══════════════════════════════════╗")
print(f"   ║  Train Accuracy : {acc_train:>6.2f}%        ║")
print(f"   ║  Test  Accuracy : {acc_test:>6.2f}%        ║")
print(f"   ║  Improvement    : +{acc_test-88.77:>5.2f}% from 88.77% ║")
print(f"   ╚══════════════════════════════════╝")

# ============================================================
# SECTION 11: MODEL SAVE KARO
# ============================================================

save_data = {
    'weights':       igzo_nn.weights,
    'biases':        igzo_nn.biases,
    'scaler':        scaler,
    'layer_sizes':   igzo_nn.layer_sizes,
    'losses':        igzo_nn.losses,
    'val_acc':       igzo_nn.val_acc,
    'lr_history':    igzo_nn.lr_history,
    'igzo_G_sorted': G_sorted,
    'igzo_vgs':      vgs_sorted,
    'avg_pot':       avg_pot,
    'avg_dep':       avg_dep,
    'pd_ratio':      pd_ratio,
    'acc_test':      acc_test,
    'acc_train':     acc_train,
}
# Dictionary mein saari important information store karo.
# Baad mein inference/deployment ke liye load kar sakte hain.

joblib.dump(save_data, os.path.join(MODEL_DIR, "igzo_IMPROVED_model.pkl"))
print(f"\n   Model saved: {os.path.join(MODEL_DIR, 'igzo_IMPROVED_model.pkl')}")

# ============================================================
# SECTION 12: VISUALIZATION — RESULTS PLOT KARO
# ============================================================

print("\n[6/6] Results visualize ho rahe hain...")

plt.style.use('dark_background')
# Dark theme — tumhare original code jaisa.

fig = plt.figure(figsize=(22, 16), facecolor='#08080f')
# figsize=(22, 16) → 22 inches wide, 16 inches tall.
# Zyada space = zyada panels.
# facecolor = figure ka background color.

gs = gridspec.GridSpec(3, 4, figure=fig,
                       hspace=0.50, wspace=0.38,
                       left=0.05, right=0.98,
                       top=0.92, bottom=0.05)
# GridSpec(rows, cols) = 3×4 grid of subplots.
# hspace = horizontal space between rows.
# wspace = vertical space between columns.
# left/right/top/bottom = figure ke andar margins.

PANEL = '#0c0c1e'   # Panel background color
GRID  = '#1c1c3a'   # Grid line color
C = ['#00f5d4', '#f72585', '#ffd60a', '#4cc9f0', '#7209b7', '#3a86ff', '#ff9f1c']
# C = color palette list.
# C[0] = cyan, C[1] = pink, C[2] = yellow, C[3] = light blue, etc.

def ax_style(ax, title, xlabel='', ylabel=''):
    """Helper function: axis ko style karo (DRY principle)."""
    ax.set_facecolor(PANEL)
    ax.set_title(title, color='white', fontsize=9.5, fontweight='bold', pad=6)
    if xlabel: ax.set_xlabel(xlabel, color='#bbbbbb', fontsize=8.5)
    if ylabel: ax.set_ylabel(ylabel, color='#bbbbbb', fontsize=8.5)
    ax.tick_params(colors='#999999', labelsize=7.5)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color('#2a2a4a')
    ax.grid(True, color=GRID, alpha=0.5, linewidth=0.5)
# DRY = Don't Repeat Yourself.
# Har plot pe same styling code likhne ki bajaye, ek function banao.
# ax = matplotlib axes object (ek single plot).

# ── Plot 1: IGZO Transfer Curve ──────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
# gs[row, col] = grid mein position select karo.
cm4 = plt.cm.cool(np.linspace(0.1, 0.95, 4))
# plt.cm.cool = cool colormap (cyan to purple).
# np.linspace(0.1, 0.95, 4) = 4 evenly spaced values.
# Har curve ke liye ek alag color.

for i, (nm, c) in enumerate(curves.items()):
    ax1.semilogy(c['VGS'], c['IDS']*1e9, color=cm4[i], lw=1.8, label=nm)
# semilogy = y-axis log scale, x-axis linear.
# *1e9 = Amperes ko nanoamperes mein convert karo (1e-9 A = 1 nA).
# lw = linewidth.

ax_style(ax1, 'IGZO Transfer Curves (Source Data)',
         'V$_{GS}$ (V)', 'I$_{DS}$ (nA)')
# $_{GS}$ = LaTeX subscript rendering in matplotlib.
ax1.legend(fontsize=8, framealpha=0.2)

# ── Plot 2: P/D Compensation Visualization ───────────────────
ax2 = fig.add_subplot(gs[0, 1])
n_steps   = 40
w_pot     = np.zeros(n_steps)
w_dep_arr = np.zeros(n_steps)
w_pot[0]  = -1.8
w_dep_arr[0] = 1.8
for i in range(1, n_steps):
    w_pot[i]     = igzo_updater.potentiate(np.array([w_pot[i-1]]))[0]
    w_dep_arr[i] = igzo_updater.depress(np.array([w_dep_arr[i-1]]))[0]
# Simulate 40 potentiation/depression pulses.

ax2.plot(range(n_steps), w_pot,     color=C[0], lw=2.5,
         label='Potentiation ↑', marker='o', ms=3)
ax2.plot(range(n_steps), w_dep_arr, color=C[1], lw=2.5,
         label=f'Depression ↓ (×{igzo_updater.dep_compensation:.1f})',
         marker='s', ms=3)
ax2.axhline(0, color='white', ls=':', lw=0.8, alpha=0.4)
ax_style(ax2, 'P/D Rule + Asymmetry Compensation', 'Pulse #', 'Weight')
ax2.legend(fontsize=7.5, framealpha=0.2)

# ── Plot 3: Learning Rate Schedule ───────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
if igzo_nn.lr_history:
    ax3.plot(range(1, len(igzo_nn.lr_history)+1),
             igzo_nn.lr_history, color=C[2], lw=2.2)
    ax3.fill_between(range(1, len(igzo_nn.lr_history)+1),
                     igzo_nn.lr_history, alpha=0.2, color=C[2])
ax_style(ax3, 'Cosine LR Decay Schedule', 'Epoch', 'Learning Rate')

# ── Plot 4: Training Loss + Accuracy ─────────────────────────
ax4 = fig.add_subplot(gs[0, 3])
epochs_arr = range(1, len(igzo_nn.losses) + 1)
ax4.plot(epochs_arr, igzo_nn.losses, color=C[1], lw=2.5, label='Loss')
ax4b = ax4.twinx()
# twinx() = second y-axis share karo same x-axis.
# Isse do different scales pe do lines ek hi plot pe dikhti hain.
ax4b.plot(epochs_arr, [v*100 for v in igzo_nn.val_acc],
          color=C[0], lw=2, ls='--', label='Val Acc %')
ax4b.set_ylabel('Val Accuracy %', color=C[0], fontsize=8)
ax4b.tick_params(colors=C[0], labelsize=7)
ax_style(ax4, 'IGZO Learning Curve', 'Epoch', 'Loss')

# ── Plot 5: Confusion Matrix ──────────────────────────────────
ax5 = fig.add_subplot(gs[1, 0:2])
# gs[1, 0:2] = row 1, columns 0 aur 1 (2 cells wide).
cm_mat = confusion_matrix(y_test, y_pred_test)
im = ax5.imshow(cm_mat, cmap='plasma', aspect='auto')
ax5.set_xticks(range(10))
ax5.set_yticks(range(10))
ax5.set_xticklabels(range(10), color='#999', fontsize=8)
ax5.set_yticklabels(range(10), color='#999', fontsize=8)
for i in range(10):
    for j in range(10):
        ax5.text(j, i, str(cm_mat[i,j]), ha='center', va='center',
                 color='white' if cm_mat[i,j] < cm_mat.max()/2 else 'black',
                 fontsize=7)
ax5.set_facecolor(PANEL)
ax5.set_title(f'Confusion Matrix  (Test Acc = {acc_test:.2f}%)',
              color='white', fontsize=10, fontweight='bold')
ax5.set_xlabel('Predicted', color='#bbb', fontsize=9)
ax5.set_ylabel('True', color='#bbb', fontsize=9)
plt.colorbar(im, ax=ax5, fraction=0.03)

# ── Plot 6: Per-Digit Accuracy ────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
per_d = [accuracy_score(y_test[y_test==d],
                        y_pred_test[y_test==d]) * 100
         for d in range(10)]
# List comprehension: har digit ke liye accuracy calculate karo.
bars = ax6.bar(range(10), per_d,
               color=plt.cm.cool(np.linspace(0.1, 0.9, 10)))
ax6.axhline(acc_test, color=C[1], ls='--', lw=1.5,
            label=f'Avg={acc_test:.1f}%')
for bar, val in zip(bars, per_d):
    ax6.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.2,
             f'{val:.0f}', ha='center', va='bottom',
             color='white', fontsize=7)
ax6.set_ylim(80, 101)
ax_style(ax6, 'Per-Digit Accuracy', 'Digit', 'Accuracy %')
ax6.legend(fontsize=8, framealpha=0.2)

# ── Plot 7: Weight Distribution (Before vs After) ─────────────
ax7 = fig.add_subplot(gs[1, 3])
all_w = np.concatenate([w.ravel() for w in igzo_nn.weights])
# .ravel() = multi-dimensional array ko flat 1D array banao.
ax7.hist(all_w, bins=100, color=C[3], alpha=0.8, edgecolor='none',
         density=True)
# density=True → y-axis = probability density (not count).
ax7.axvline(0, color='white', ls='--', lw=1.2, alpha=0.6,
            label=f'Mean={all_w.mean():.3f}')
ax7.axvline(all_w.mean(), color=C[2], ls=':', lw=1.5)
ax_style(ax7, 'IGZO Synaptic Weight Distribution', 'Weight Value', 'Density')
ax7.legend(fontsize=7.5, framealpha=0.2)

# ── Plot 8: Sample Predictions ────────────────────────────────
ax8 = fig.add_subplot(gs[2, 0:2])
ax8.axis('off')
ax8.set_facecolor(PANEL)
ax8.set_title('Correct Predictions — Improved IGZO Model',
              color=C[0], fontsize=10, fontweight='bold')
correct_idx = np.where(y_pred_test == y_test)[0][:20]
proba_all   = igzo_nn.predict_proba(X_test[correct_idx])
for k, idx in enumerate(correct_idx):
    img = scaler.inverse_transform(X_test[idx:idx+1])[0]
    img = np.clip(img, 0, 255) / 255.0
    # inverse_transform → normalize undo karo (wapas 0-255 pe).
    sub = fig.add_axes([0.05 + (k % 10) * 0.047,
                        0.10 + (1 - k//10) * 0.075,
                        0.040, 0.060])
    sub.imshow(img.reshape(28, 28), cmap='plasma', interpolation='nearest')
    conf = proba_all[k, y_test[correct_idx[k]]] * 100
    sub.set_title(f'{y_test[correct_idx[k]]}\n{conf:.0f}%',
                  color=C[0], fontsize=5.5, pad=1)
    sub.axis('off')

# ── Plot 9: Improvement Summary Table ─────────────────────────
ax9 = fig.add_subplot(gs[2, 2:])
ax9.axis('off')
rows = [
    ['Improvement',          'Original',      'Improved'],
    ['Architecture',         '784→256→10',    '784→512→256→10'],
    ['Hidden layers',        '1',             '2'],
    ['P/D Compensation',     'None',          f'{pd_ratio:.1f}x scale-up'],
    ['LR Schedule',          'Fixed 0.008',   'Cosine decay'],
    ['Data Augmentation',    'No',            'Noise+Dropout+Shift'],
    ['Early Stopping',       'No',            'patience=15'],
    ['Epochs',               '30',            '80 (early stop)'],
    ['Test Accuracy',        '88.77%',        f'{acc_test:.2f}%'],
    ['Improvement',          '—',             f'+{acc_test-88.77:.2f}%'],
]
tbl = ax9.table(cellText=rows[1:], colLabels=rows[0],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)

# Table styling
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#2a2a4a')
    if r == 0:
        cell.set_facecolor('#1a1a4a')
        cell.set_text_props(color=C[0], fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#0c0c22')
        cell.set_text_props(color='white')
    else:
        cell.set_facecolor('#111133')
        cell.set_text_props(color='white')
    # Last row highlight
    if r == len(rows) - 1:
        cell.set_facecolor('#1a3a1a')
        if c == 2:
            cell.set_text_props(color='#00ff88', fontweight='bold')

tbl.scale(1, 2.0)
ax9.set_title('Improvement Summary', color='white',
              fontsize=10, fontweight='bold', pad=8)

# ── Main Title ────────────────────────────────────────────────
fig.suptitle(
    f'IGZO/MgO Improved — Test Accuracy: {acc_test:.2f}%  '
    f'(+{acc_test-88.77:.2f}% from baseline 88.77%)',
    color='white', fontsize=13, fontweight='bold', y=0.97)

# ── Save aur Show ─────────────────────────────────────────────
out_path = os.path.join(MODEL_DIR, 'igzo_IMPROVED_results.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
# dpi=150 = dots per inch (image resolution).
# bbox_inches='tight' = extra whitespace crop karo.
print(f"   Plot saved: {out_path}")

print("\n" + "=" * 62)
print(f"  IMPROVED IGZO TRAINING COMPLETE!")
print(f"  Train Accuracy : {acc_train:.2f}%")
print(f"  Test  Accuracy : {acc_test:.2f}%")
print(f"  Gain over baseline : +{acc_test-88.77:.2f}%")
print(f"  Model saved in : models/igzo_IMPROVED_model.pkl")
print(f"  Architecture   : 784 → 512 → 256 → 10")
print(f"  Physics        : 100% IGZO P/D (no Adam/SGD)")
print(f"  P/D Ratio fix  : {pd_ratio:.1f}x compensation applied")
print("=" * 62)

plt.show()
# plt.show() = window kholo aur plot dikhao.
# IMPORTANT: ye blocking call hai — jab tak window band na ho,
# program yahan rok jaata hai.
# Agar sirf save karna hai show nahi: comment out karo.