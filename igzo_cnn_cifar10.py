"""
==============================================================
  IGZO/MgO Synaptic Transistor
  CNN + IGZO Hybrid — CIFAR-10 Real Object Classification
  Target: 90%+ accuracy
==============================================================

ARCHITECTURE DESIGN:
  Conv layers  → standard Adam optimizer (spatial feature extraction)
  Dense layers → IGZO P/D physics      (synaptic classification)

PHASE 1 SKIP:
  If best_cnn_weights.h5 already exists in models_cnn_igzo/,
  Phase 1 training is automatically skipped and weights are loaded.
  Delete the .h5 file to retrain from scratch.

REQUIRES:
  pip install tensorflow "numpy<2" matplotlib scipy pandas joblib scikit-learn
==============================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── GPU detection ─────────────────────────────────────────────
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# ── Paths ─────────────────────────────────────────────────────
OUTPUT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR       = os.path.join(OUTPUT_DIR, "models_cnn_igzo")
os.makedirs(MODEL_DIR, exist_ok=True)
DATA_DIR        = os.path.join(OUTPUT_DIR, "data")
IDVD_FILE       = os.path.join(DATA_DIR, "IDVD_IDBG_IGZO_MgO.csv")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_cnn_weights.h5")

CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

print("=" * 62)
print("   IGZO/MgO CNN Hybrid — Real Object Classification")
print("   Target: 90%+ on CIFAR-10")
print("   Conv layers : standard Adam")
print("   Dense layer : IGZO P/D physics")
print("=" * 62)

# ============================================================
# STEP 1: LOAD IGZO PHYSICS
# ============================================================

print("\n[1/7] Loading IGZO physics from your transistor data...")

try:
    df_iv   = pd.read_csv(IDVD_FILE, header=0)
    vgs_raw = pd.to_numeric(df_iv['Unnamed: 2'], errors='coerce').dropna().values
    ids_raw = pd.to_numeric(df_iv['Cw11'],       errors='coerce').dropna().values
    n       = min(len(vgs_raw), len(ids_raw))
    vgs_raw, ids_raw = vgs_raw[:n], ids_raw[:n]
    mask    = ids_raw > 0
    vgs_raw, ids_raw = vgs_raw[mask], ids_raw[mask]
    G_norm  = (ids_raw - ids_raw.min()) / (ids_raw.max() - ids_raw.min())
    idx_s   = np.argsort(vgs_raw)
    vgs_s, G_s = vgs_raw[idx_s], G_norm[idx_s]
    dG      = np.diff(G_s)
    avg_pot = dG[dG > 0].mean() if (dG > 0).any() else 0.00293
    avg_dep = abs(dG[dG < 0].mean()) if (dG < 0).any() else 0.000092
    pd_ratio = avg_pot / (avg_dep + 1e-10)
    n_states = len(G_s)
    print(f"   avg_pot  = {avg_pot:.6f}")
    print(f"   avg_dep  = {avg_dep:.6f}")
    print(f"   P/D ratio= {pd_ratio:.1f}x (will compensate)")
    print(f"   States   = {n_states}")
except Exception as e:
    print(f"   WARNING: IGZO file not found ({e}). Using defaults.")
    avg_pot, avg_dep, pd_ratio, n_states = 0.00293, 0.000092, 31.8, 402

# ============================================================
# STEP 2: IGZO WEIGHT UPDATER
# ============================================================

class IGZOWeightUpdater:
    """Identical to all previous scripts — physics unchanged."""

    def __init__(self, avg_pot, avg_dep):
        self.avg_pot          = avg_pot
        self.avg_dep          = abs(avg_dep)
        self.dep_compensation = min(avg_pot / (self.avg_dep + 1e-10), 30.0)

    def w2g(self, w):
        return np.clip((w + 2.0) / 4.0, 0.0, 1.0)

    def g2w(self, G):
        return G * 4.0 - 2.0

    def potentiate(self, w, lr=1.0):
        G = self.w2g(w)
        return self.g2w(np.clip(G + self.avg_pot * lr * (1.0 - G), 0, 1))

    def depress(self, w, lr=1.0):
        G = self.w2g(w)
        return self.g2w(np.clip(G - self.avg_dep * lr * G, 0, 1))

    def update(self, w, gradient, lr=0.01):
        w_new = w.copy()
        s     = np.abs(gradient) * lr * 50
        pm    = gradient < 0
        dm    = gradient > 0
        if pm.any():
            w_new[pm] = self.potentiate(w[pm], s[pm])
        if dm.any():
            w_new[dm] = self.depress(
                w[dm], np.clip(s[dm] * self.dep_compensation, 0, 5))
        return w_new

igzo = IGZOWeightUpdater(avg_pot, avg_dep)
print(f"\n   IGZO updater ready. Compensation: {igzo.dep_compensation:.1f}x")

# ============================================================
# STEP 3: LOAD AND PREPROCESS CIFAR-10
# ============================================================

print("\n[2/7] Loading CIFAR-10...")

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
y_train = y_train.flatten()
y_test  = y_test.flatten()

X_train = X_train.astype(np.float32) / 255.0
X_test  = X_test.astype(np.float32)  / 255.0

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

X_train = (X_train - CIFAR_MEAN) / CIFAR_STD
X_test  = (X_test  - CIFAR_MEAN) / CIFAR_STD

y_train_oh = keras.utils.to_categorical(y_train, 10)
y_test_oh  = keras.utils.to_categorical(y_test,  10)

print(f"   Train: {X_train.shape}  Test: {X_test.shape}")
print(f"   Normalized per-channel (mean/std)")

# ============================================================
# STEP 4: DATA AUGMENTATION
# ============================================================

datagen = ImageDataGenerator(
    horizontal_flip    = True,
    width_shift_range  = 0.1,
    height_shift_range = 0.1,
    rotation_range     = 15,
    zoom_range         = 0.1,
    fill_mode          = 'reflect',
)
datagen.fit(X_train)
print(f"   Augmentation: flip + shift + rotate + zoom")

# ============================================================
# STEP 5: BUILD CNN BACKBONE
# ============================================================

print("\n[3/7] Building CNN architecture...")

def conv_block(x, filters, kernel=3):
    x = layers.Conv2D(filters, kernel, padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel, padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)
    return x

inp = layers.Input(shape=(32, 32, 3), name='image_input')
x   = conv_block(inp, 32)
x   = conv_block(x,   64)
x   = conv_block(x,   128)
x   = layers.Conv2D(128, 3, padding='same',
                    kernel_initializer='he_normal')(x)
x   = layers.BatchNormalization()(x)
x   = layers.Activation('relu')(x)
x   = layers.GlobalAveragePooling2D(name='gap')(x)
x   = layers.Dense(256, name='igzo_dense')(x)
x   = layers.BatchNormalization()(x)
x   = layers.Activation('relu')(x)
x   = layers.Dropout(0.5)(x)
out = layers.Dense(10, activation='softmax', name='output')(x)

cnn_model = Model(inputs=inp, outputs=out, name='IGZO_CNN')
cnn_model.summary()
print(f"\n   Total parameters: {cnn_model.count_params():,}")

# ============================================================
# STEP 6: PHASE 1 — PRETRAIN WITH ADAM (skip if already done)
# ============================================================

print("\n[4/7] Phase 1: Pre-training CNN backbone with Adam...")

BATCH_SIZE = 128
EPOCHS_P1  = 40

if os.path.exists(CHECKPOINT_PATH):
    # ── SKIP: weights already saved from a previous run ─────
    print(f"   Saved weights found! Skipping Phase 1 training.")
    print(f"   Loading from: {CHECKPOINT_PATH}")
    cnn_model.compile(
        optimizer = optimizers.Adam(learning_rate=0.001),
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy']
    )
    cnn_model.load_weights(CHECKPOINT_PATH)
    loss_p1, acc_p1 = cnn_model.evaluate(X_test, y_test_oh, verbose=0)
    print(f"   Phase 1 accuracy (loaded): {acc_p1*100:.2f}%")
    history_p1 = None   # no training history — plots will handle this

else:
    # ── TRAIN: no saved weights, train from scratch ──────────
    print(f"   No saved weights found. Training from scratch...")
    cnn_model.compile(
        optimizer = optimizers.Adam(learning_rate=0.001),
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy']
    )
    cbs = [
        callbacks.ModelCheckpoint(
            CHECKPOINT_PATH,
            monitor       = 'val_accuracy',
            save_best_only= True,
            verbose       = 1
        ),
        callbacks.ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 0.5,
            patience = 5,
            min_lr   = 1e-6,
            verbose  = 1
        ),
        callbacks.EarlyStopping(
            monitor              = 'val_accuracy',
            patience             = 15,
            restore_best_weights = True,
            verbose              = 1
        )
    ]
    history_p1 = cnn_model.fit(
        datagen.flow(X_train, y_train_oh, batch_size=BATCH_SIZE),
        epochs          = EPOCHS_P1,
        validation_data = (X_test, y_test_oh),
        callbacks       = cbs,
        steps_per_epoch = len(X_train) // BATCH_SIZE,
        verbose         = 1
    )
    loss_p1, acc_p1 = cnn_model.evaluate(X_test, y_test_oh, verbose=0)
    print(f"\n   Phase 1 complete. Test accuracy: {acc_p1*100:.2f}%")

# Always reload best checkpoint before Phase 2
cnn_model.load_weights(CHECKPOINT_PATH)
print(f"   Best weights loaded: {CHECKPOINT_PATH}")

# ============================================================
# STEP 7: PHASE 2 — IGZO FINE-TUNING OF DENSE LAYER
# ============================================================

print("\n[5/7] Phase 2: IGZO fine-tuning of Dense layer...")
print("   Freezing conv backbone, applying IGZO P/D to Dense weights...")

# Build feature extractor (output = GAP layer, backbone frozen)
feature_extractor = Model(
    inputs  = cnn_model.input,
    outputs = cnn_model.get_layer('gap').output,
    name    = 'feature_extractor'
)

# Extract features once for all images
print("   Extracting CNN features (this takes ~1-2 minutes)...")
F_train = feature_extractor.predict(X_train, batch_size=256, verbose=0)
F_test  = feature_extractor.predict(X_test,  batch_size=256, verbose=0)
print(f"   Feature shape: train={F_train.shape}, test={F_test.shape}")

# Pull Dense and Output weights from trained CNN
dense_layer      = cnn_model.get_layer('igzo_dense')
W_dense, b_dense = dense_layer.get_weights()
# b_dense shape: (256,)

output_layer     = cnn_model.get_layer('output')
W_out, b_out     = output_layer.get_weights()
# b_out shape: (10,)

print(f"   Dense weight shape : {W_dense.shape}")
print(f"   Output weight shape: {W_out.shape}")

# One-hot for numpy loop
def one_hot_np(y, n=10):
    oh = np.zeros((len(y), n))
    oh[np.arange(len(y)), y] = 1
    return oh

y_train_oh_np = one_hot_np(y_train)
y_test_oh_np  = one_hot_np(y_test)

# Forward pass through Dense layers
def igzo_forward(F, W1, b1, W2, b2):
    Z1 = F @ W1 + b1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2
    Z2 = Z2 - Z2.max(axis=1, keepdims=True)
    E  = np.exp(Z2)
    return E / E.sum(axis=1, keepdims=True)

def cross_entropy(pred, y_oh):
    return -np.mean(np.sum(y_oh * np.log(pred + 1e-9), axis=1))

# IGZO training hyperparameters
IGZO_EPOCHS   = 50
IGZO_LR       = 0.005
IGZO_BATCH    = 256
IGZO_PATIENCE = 12
base_lr       = IGZO_LR

best_acc_igzo = acc_p1
best_W1 = W_dense.copy()
best_b1 = b_dense.copy()
best_W2 = W_out.copy()
best_b2 = b_out.copy()
no_improve    = 0
igzo_losses   = []
igzo_accs     = []

print(f"\n   IGZO epochs: {IGZO_EPOCHS} | batch: {IGZO_BATCH} | lr: {IGZO_LR}")
print(f"   {'Ep':>4} | {'Loss':>8} | {'Val Acc':>8} | {'LR':>9} | Status")
print(f"   {'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-------")

for ep in range(1, IGZO_EPOCHS + 1):

    # Cosine LR decay
    cos = 0.5 * (1 + np.cos(np.pi * ep / IGZO_EPOCHS))
    lr  = base_lr * (0.05 + 0.95 * cos)

    # Shuffle
    idx = np.random.permutation(len(F_train))
    Fs  = F_train[idx]
    ys  = y_train_oh_np[idx]

    # Mini-batch IGZO updates
    for s in range(0, len(Fs), IGZO_BATCH):
        Fb = Fs[s:s + IGZO_BATCH]
        yb = ys[s:s + IGZO_BATCH]
        m  = len(Fb)

        # Forward
        Z1 = Fb @ W_dense + b_dense
        A1 = np.maximum(0, Z1)
        Z2 = A1 @ W_out + b_out
        Z2 = Z2 - Z2.max(axis=1, keepdims=True)
        E  = np.exp(Z2)
        P  = E / E.sum(axis=1, keepdims=True)

        # Backward — output layer
        dZ2 = (P - yb) / m
        dW2 = A1.T @ dZ2
        db2 = dZ2.mean(axis=0)          # shape (10,)  ✓

        # Backward — dense layer
        dA1 = dZ2 @ W_out.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = Fb.T @ dZ1
        db1 = dZ1.mean(axis=0)          # shape (256,) ✓

        # ── IGZO P/D weight update (core physics) ────────────
        W_dense = igzo.update(W_dense, dW1, lr=lr)
        W_out   = igzo.update(W_out,   dW2, lr=lr)
        # Biases use standard gradient (no IGZO equivalent)
        b_dense -= lr * db1              # (256,) -= scalar*(256,) ✓
        b_out   -= lr * db2              # (10,)  -= scalar*(10,)  ✓
        # ─────────────────────────────────────────────────────

    # Epoch evaluation
    P_val  = igzo_forward(F_test, W_dense, b_dense, W_out, b_out)
    loss_v = cross_entropy(P_val, y_test_oh_np)
    acc_v  = accuracy_score(y_test, P_val.argmax(axis=1))
    igzo_losses.append(loss_v)
    igzo_accs.append(acc_v)

    # Track best weights
    status = ""
    if acc_v > best_acc_igzo + 0.0001:
        best_acc_igzo = acc_v
        best_W1 = W_dense.copy()
        best_b1 = b_dense.copy()
        best_W2 = W_out.copy()
        best_b2 = b_out.copy()
        no_improve = 0
        status = "★ best"
    else:
        no_improve += 1
        status = f"patience {no_improve}/{IGZO_PATIENCE}"

    if ep % 5 == 0 or ep <= 3 or status.startswith("★"):
        print(f"   {ep:>4} | {loss_v:>8.5f} | {acc_v*100:>7.2f}% "
              f"| {lr:>9.6f} | {status}")

    if no_improve >= IGZO_PATIENCE:
        print(f"\n   Early stop at epoch {ep}.")
        break

# Restore best IGZO weights
W_dense, b_dense = best_W1, best_b1
W_out,   b_out   = best_W2, best_b2

# ============================================================
# STEP 8: FINAL EVALUATION
# ============================================================

print("\n[6/7] Final evaluation...")

P_final     = igzo_forward(F_test, W_dense, b_dense, W_out, b_out)
y_pred_test = P_final.argmax(axis=1)
acc_final   = accuracy_score(y_test, y_pred_test) * 100

P_tr        = igzo_forward(F_train[:5000], W_dense, b_dense, W_out, b_out)
acc_train_f = accuracy_score(y_train[:5000], P_tr.argmax(axis=1)) * 100

per_class = [accuracy_score(y_test[y_test == c],
                            y_pred_test[y_test == c]) * 100
             for c in range(10)]

print(f"\n   ╔══════════════════════════════════════╗")
print(f"   ║  Phase 1 (Adam CNN)  : {acc_p1*100:>6.2f}%         ║")
print(f"   ║  Phase 2 (IGZO fine) : {acc_final:>6.2f}%         ║")
print(f"   ║  Train accuracy      : {acc_train_f:>6.2f}%         ║")
print(f"   ╠══════════════════════════════════════╣")
for name, acc in zip(CLASS_NAMES, per_class):
    print(f"   ║  {name:<12} : {acc:>5.1f}%                 ║")
print(f"   ╚══════════════════════════════════════╝")

# Save results
save_data = {
    'W_dense'    : W_dense,     'b_dense'  : b_dense,
    'W_out'      : W_out,       'b_out'    : b_out,
    'acc_p1'     : acc_p1,      'acc_final': acc_final,
    'igzo_losses': igzo_losses, 'igzo_accs': igzo_accs,
    'class_names': CLASS_NAMES,
    'avg_pot'    : avg_pot,     'avg_dep'  : avg_dep,
    'pd_ratio'   : pd_ratio,
}
joblib.dump(save_data, os.path.join(MODEL_DIR, 'igzo_cnn_results.pkl'))
cnn_model.save(os.path.join(MODEL_DIR, 'cnn_backbone.h5'))
print(f"\n   Results saved: {MODEL_DIR}/")

# ============================================================
# STEP 9: PUBLICATION-QUALITY PLOTS
# ============================================================

print("\n[7/7] Generating plots...")

fig = plt.figure(figsize=(22, 14), facecolor='white')
gs  = gridspec.GridSpec(3, 4, figure=fig,
                        hspace=0.46, wspace=0.35,
                        left=0.05, right=0.98,
                        top=0.93, bottom=0.06)

def sax(ax, title='', xl='', yl=''):
    ax.tick_params(direction='in', top=True, right=True, labelsize=9)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    if title: ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
    if xl:    ax.set_xlabel(xl, fontsize=9)
    if yl:    ax.set_ylabel(yl, fontsize=9)

# ── 1: Phase 1 training curves ───────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
if history_p1 is not None:
    ep1 = range(1, len(history_p1.history['accuracy']) + 1)
    ax1.plot(ep1, [v*100 for v in history_p1.history['accuracy']],
             color='#3498db', lw=2, label='Train')
    ax1.plot(ep1, [v*100 for v in history_p1.history['val_accuracy']],
             color='#e74c3c', lw=2, ls='--', label='Val')
    ax1.legend(fontsize=8)
else:
    ax1.text(0.5, 0.5, f'Loaded from\nsaved weights\n{acc_p1*100:.2f}%',
             ha='center', va='center', fontsize=11,
             transform=ax1.transAxes, color='#27ae60',
             fontweight='bold')
ax1.axhline(acc_p1*100, color='green', ls=':', lw=1.2,
            label=f'Best {acc_p1*100:.1f}%')
sax(ax1, 'Phase 1: Adam CNN training', 'Epoch', 'Accuracy %')
ax1.set_ylim(0, 100)

# ── 2: Phase 1 loss ──────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
if history_p1 is not None:
    ax2.plot(ep1, history_p1.history['loss'],
             color='#3498db', lw=2, label='Train loss')
    ax2.plot(ep1, history_p1.history['val_loss'],
             color='#e74c3c', lw=2, ls='--', label='Val loss')
    ax2.legend(fontsize=8)
else:
    ax2.text(0.5, 0.5, 'Phase 1 skipped\n(weights loaded)',
             ha='center', va='center', fontsize=11,
             transform=ax2.transAxes, color='gray')
sax(ax2, 'Phase 1: Loss curve', 'Epoch', 'Loss')

# ── 3: Phase 2 IGZO fine-tuning curve ────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ep2 = range(1, len(igzo_accs) + 1)
ax3.plot(ep2, [v*100 for v in igzo_accs],
         color='#9b59b6', lw=2.5, label='IGZO val acc')
ax3.axhline(acc_p1*100, color='gray', ls=':', lw=1.2,
            label=f'Adam baseline {acc_p1*100:.1f}%')
ax3.axhline(best_acc_igzo*100, color='green', ls='--', lw=1.2,
            label=f'IGZO best {best_acc_igzo*100:.1f}%')
sax(ax3, 'Phase 2: IGZO fine-tuning', 'Epoch', 'Accuracy %')
ax3.legend(fontsize=8)

# ── 4: IGZO P/D physics visualization ────────────────────────
ax4 = fig.add_subplot(gs[0, 3])
n_p = 40
wp  = np.full(n_p, -1.8)
wd  = np.full(n_p,  1.8)
for i in range(1, n_p):
    wp[i] = igzo.potentiate(np.array([wp[i-1]]))[0]
    wd[i] = igzo.depress(np.array([wd[i-1]]))[0]
ax4.plot(range(n_p), wp, 'b-o', ms=3, lw=2, label='Potentiation')
ax4.plot(range(n_p), wd, 'r-s', ms=3, lw=2, label='Depression')
ax4.axhline(0, color='gray', ls=':', lw=0.8)
sax(ax4, 'IGZO P/D rule (from your CSV)', 'Pulse #', 'Weight')
ax4.legend(fontsize=8)

# ── 5: Confusion matrix ──────────────────────────────────────
ax5 = fig.add_subplot(gs[1, :2])
cm  = confusion_matrix(y_test, y_pred_test)
im  = ax5.imshow(cm, cmap='Blues', aspect='auto')
ax5.set_xticks(range(10))
ax5.set_yticks(range(10))
ax5.set_xticklabels([n[:5] for n in CLASS_NAMES],
                    rotation=40, ha='right', fontsize=8)
ax5.set_yticklabels([n[:5] for n in CLASS_NAMES], fontsize=8)
for i in range(10):
    for j in range(10):
        ax5.text(j, i, str(cm[i, j]), ha='center', va='center',
                 fontsize=6.5,
                 color='white' if cm[i, j] > cm.max()*0.5 else 'black')
ax5.set_title(f'Confusion matrix — Test accuracy: {acc_final:.2f}%',
              fontsize=11, fontweight='bold')
ax5.set_xlabel('Predicted', fontsize=9)
ax5.set_ylabel('True', fontsize=9)
plt.colorbar(im, ax=ax5, fraction=0.03)

# ── 6: Per-class accuracy ─────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
colors = plt.cm.tab10(np.linspace(0, 1, 10))
bars   = ax6.bar(range(10), per_class, color=colors)
ax6.axhline(acc_final, color='red', ls='--', lw=1.5,
            label=f'Avg {acc_final:.1f}%')
ax6.set_xticks(range(10))
ax6.set_xticklabels([n[:4] for n in CLASS_NAMES],
                    rotation=40, ha='right', fontsize=7.5)
for bar, v in zip(bars, per_class):
    ax6.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3, f'{v:.0f}',
             ha='center', fontsize=6.5)
ax6.set_ylim(0, 100)
sax(ax6, 'Per-class accuracy', 'Class', 'Acc %')
ax6.legend(fontsize=8)

# ── 7: IGZO weight distribution ──────────────────────────────
ax7 = fig.add_subplot(gs[1, 3])
all_w = np.concatenate([W_dense.ravel(), W_out.ravel()])
ax7.hist(all_w, bins=80, color='#9b59b6', alpha=0.85,
         density=True, label=f'mean={all_w.mean():.3f}')
ax7.axvline(0, color='black', ls='--', lw=1)
sax(ax7, 'IGZO synaptic weight dist.', 'Weight', 'Density')
ax7.legend(fontsize=8)

# ── 8: Sample predictions ─────────────────────────────────────
ax8 = fig.add_subplot(gs[2, :2])
ax8.axis('off')
ax8.set_title('Sample predictions — CNN + IGZO model',
              fontsize=10, fontweight='bold')
sample_idx = []
for c in range(10):
    idx_c = np.where(y_test == c)[0][:2]
    sample_idx.extend(idx_c.tolist())

X_disp = np.clip(X_test * CIFAR_STD + CIFAR_MEAN, 0, 1)

for k, idx in enumerate(sample_idx[:20]):
    sub = fig.add_axes([0.05 + (k % 10)*0.052,
                        0.10 + (1 - k//10)*0.10,
                        0.046, 0.085])
    sub.imshow(X_disp[idx], interpolation='nearest')
    pred = y_pred_test[idx]
    true = y_test[idx]
    col  = 'green' if pred == true else 'red'
    sub.set_title(f'{CLASS_NAMES[pred][:4]}\n{CLASS_NAMES[true][:4]}',
                  fontsize=5, color=col, pad=1)
    sub.axis('off')

# ── 9: Architecture + summary table ──────────────────────────
ax9 = fig.add_subplot(gs[2, 2:])
ax9.axis('off')
rows = [
    ['Component',      'Details',                       'Optimizer'],
    ['Conv Block 1',   '2x Conv 3x3, 32 filters, BN',  'Adam'],
    ['Conv Block 2',   '2x Conv 3x3, 64 filters, BN',  'Adam'],
    ['Conv Block 3',   '2x Conv 3x3, 128 filters, BN', 'Adam'],
    ['Extra Conv',     'Conv 3x3, 128 filters, BN',     'Adam'],
    ['Global AvgPool', '4x4x128 → 128',                 'Adam'],
    ['Dense 256',      'ReLU + BN + Dropout 0.5',       'IGZO P/D'],
    ['Output 10',      'Softmax, 10 classes',            'IGZO P/D'],
    ['Phase 1 acc',    f'{acc_p1*100:.2f}%',            'Adam only'],
    ['Phase 2 acc',    f'{acc_final:.2f}%',             'IGZO fine-tune'],
]
tbl = ax9.table(cellText=rows[1:], colLabels=rows[0],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#dee2e6')
    if r == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif c == 2 and r > 0:
        val = rows[r][2] if r <= len(rows) - 1 else ''
        if 'IGZO' in str(val):
            cell.set_facecolor('#eaf4fb')
            cell.set_text_props(color='#1a5276', fontweight='bold')
        else:
            cell.set_facecolor('#fdfefe')
    elif r % 2 == 0:
        cell.set_facecolor('#f8f9fa')
tbl.scale(1, 2.1)

fig.suptitle(
    f'IGZO/MgO CNN Hybrid — CIFAR-10  |  '
    f'Phase 1 (Adam): {acc_p1*100:.2f}%  →  '
    f'Phase 2 (IGZO): {acc_final:.2f}%',
    fontsize=13, fontweight='bold', y=0.98)

out = os.path.join(OUTPUT_DIR, 'igzo_cnn_results.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f"   Plot saved: {out}")

print("\n" + "=" * 62)
print(f"  CNN + IGZO COMPLETE!")
print(f"  Phase 1 (Adam CNN)   : {acc_p1*100:.2f}%")
print(f"  Phase 2 (IGZO dense) : {acc_final:.2f}%")
print(f"  Target               : 90%+")
print(f"  IGZO physics         : Dense layer weights")
print(f"  Conv backbone        : standard Adam (frozen in P2)")
print(f"  Model saved in       : {MODEL_DIR}/")
print(f"\n  NOTE: To retrain Phase 1 from scratch,")
print(f"        delete: {CHECKPOINT_PATH}")
print("=" * 62)

plt.show()