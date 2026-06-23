# IGZO/MgO Synaptic Transistor — Complete ML Project
### Neuromorphic Digit Recognition System

---

## Project Structure
```
igzo_project/
├── data/
│   ├── IDVD_IDBG_IGZO_MgO.csv              ← Lab ka Transfer curve data
│   └── filterchar_IGZO_MgO_new.csv          ← EPSC frequency data
│
├── models/                                   ← Trained models yahan save honge
│   ├── device_params.pkl                    ← Vth, ION/IOFF, SS etc.
│   ├── poly_model.pkl                       ← Polynomial IDS predictor
│   ├── nn_model.pkl                         ← Neural Net IDS predictor
│   ├── gain_model.pkl                       ← EPSC Gain model
│   ├── igzo_digit_model.pkl                 ← Standard digit model
│   └── igzo_TRUE_digit_model.pkl            ← TRUE IGZO digit model (BEST)
│
├── 1_train_model.py                         ← STEP 1: Device model train karo
├── 2_digit_recognition.py                   ← STEP 2: Standard digit recognition
├── 2b_digit_recognition_IGZO_TRUE.py        ← STEP 2b: TRUE IGZO learning (BEST)
├── 3_app.py                                 ← STEP 3: GUI app
├── requirements.txt
└── README.md
```

---

## Konsa Model Tera Hai?

| File | Tera IGZO? | Kya karta hai |
|------|-----------|---------------|
| poly_model.pkl | 100% TERA | VGS se IDS predict karta hai |
| nn_model.pkl | 100% TERA | VGS se IDS predict karta hai |
| gain_model.pkl | 100% TERA | Frequency se EPSC Gain predict |
| igzo_digit_model.pkl | Partial | IGZO weight quantization only |
| igzo_TRUE_digit_model.pkl | 100% TERA | IGZO P/D curve se seekha (BEST) |

---

## Setup — Sirf ek baar karo

Python 3.10+ install karo: https://python.org/downloads
Install karte waqt "Add to PATH" zaroor tick karo!

CMD mein:
```
pip install -r requirements.txt
```

---

## Run Order — Ye sequence follow karo

### STEP 1
```
python 1_train_model.py
```
Device parameters + IDS predictor models ban jayenge

### STEP 2b — TRUE IGZO (Recommended)
```
python 2b_digit_recognition_IGZO_TRUE.py
```
100% tera transistor seekhega — P/D curve se weight update hoga

### STEP 2 — Standard (Optional comparison ke liye)
```
python 2_digit_recognition.py
```
Standard optimizer — zyada accuracy lekin IGZO physics kam

### STEP 3 — GUI App
```
python 3_app.py
```
Draw karo digit — model predict karega

---

## TRUE IGZO Learning Kya Hai?

Standard:   W = W - lr x gradient  (math formula)
IGZO TRUE:  W = igzo.potentiate(W) se  (teri CSV ka actual dG/dVGS)
            W = igzo.depress(W) se      (teri CSV ka actual dG/dVGS)

402 conductance states = IDVD_IDBG_IGZO_MgO.csv ke 402 data points

---

## Troubleshooting

ModuleNotFoundError:
  pip install -r requirements.txt

FileNotFoundError models:
  Pehle python 1_train_model.py chalao

App nahi khul rahi:
  pip install pillow --upgrade

MNIST slow download:
  Internet chahiye pehli baar — 11MB file hai

---

Based on paper: Low-Temperature Solution-Processed In2O3 Synaptic Transistors
IEEE Transactions on Electron Devices, Vol. 73, No. 1, Jan 2026
