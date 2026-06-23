"""
==============================================================
STEP 2: IGZO-Inspired Digit Recognition on MNIST
==============================================================
Ye script:
  - IGZO device physics se inspired ANN banati hai
  - Synaptic weights = IGZO conductance states
  - MNIST dataset pe train & test karti hai
  - ~95% accuracy achieve karti hai (paper jaise!)

Run karo: python 2_digit_recognition.py
==============================================================
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 62)
print("   IGZO-Inspired Digit Recognition — MNIST")
print("=" * 62)

# ==============================================================
# 1. IGZO SYNAPTIC WEIGHT MAPPER
# ==============================================================
# IGZO device ka conductance multiple discrete states mein hota hai
# Hum yahi use karenge as weight quantization

class IGZOSynapticLayer:
    """
    IGZO device physics se inspired weight mapper.
    Conductance states simulate karta hai.
    """
    def __init__(self, n_states=32):
        self.n_states = n_states
        # IGZO-like potentiation curve (paper Fig 4 se inspired)
        t = np.linspace(0, 1, n_states)
        self.G_states = 0.08 + 0.92 * (1 - np.exp(-t * 4.5))  # normalized

    def quantize_weights(self, weights):
        """Weights ko IGZO conductance states pe snap karo"""
        w_min, w_max = weights.min(), weights.max()
        if w_max == w_min:
            return weights
        w_norm = (weights - w_min) / (w_max - w_min)
        # Find nearest conductance state
        indices = np.argmin(
            np.abs(w_norm.ravel()[:, None] - self.G_states[None, :]),
            axis=1
        )
        w_quantized = self.G_states[indices].reshape(weights.shape)
        # Rescale back
        return w_quantized * (w_max - w_min) + w_min

    def get_conductance_info(self):
        return {
            'n_states': self.n_states,
            'G_min': self.G_states.min(),
            'G_max': self.G_states.max(),
            'G_range': self.G_states.max() - self.G_states.min()
        }

igzo_layer = IGZOSynapticLayer(n_states=32)
print(f"\n[INFO] IGZO Synaptic Layer initialized")
print(f"       Conductance states : {igzo_layer.n_states}")
info = igzo_layer.get_conductance_info()
print(f"       G range            : {info['G_min']:.3f} to {info['G_max']:.3f}")

# ==============================================================
# 2. LOAD MNIST DATA
# ==============================================================
print("\n[1/4] MNIST data load ho raha hai...")
print("      (Pehli baar internet se download hoga ~11MB, patience rakho)")

try:
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    print(f"   OK Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
except Exception as e:
    print(f"   ERROR: {e}")
    print("   Internet connection check karo aur dobara try karo")
    exit(1)

# Normalize pixels 0-255 -> 0-1
X = X / 255.0

# Split: 60000 train, 10000 test (MNIST standard)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

print(f"   Train : {X_train.shape[0]} samples")
print(f"   Test  : {X_test.shape[0]} samples")

# ==============================================================
# 3. BUILD & TRAIN IGZO-INSPIRED ANN
# ==============================================================
print("\n[2/4] IGZO-Inspired ANN train ho rahi hai...")
print("      (3-5 minutes lagenge, coffee pi lo!)")
print()

# Architecture inspired from paper Fig 7:
# 784 input -> 300 hidden -> 10 output
# Paper mein bhi 28x28 pixel = 784 input neurons use hue hain

igzo_ann = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(300, 100),   # paper-inspired architecture
        activation='relu',
        solver='adam',
        alpha=1e-4,                       # regularization
        batch_size=200,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=50,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=True,                     # progress dikhao
        tol=1e-4,
    ))
])

igzo_ann.fit(X_train, y_train)

# Apply IGZO weight quantization
mlp = igzo_ann.named_steps['mlp']
original_weights = [w.copy() for w in mlp.coefs_]

print("\n   Applying IGZO synaptic weight quantization...")
for i in range(len(mlp.coefs_)):
    mlp.coefs_[i] = igzo_layer.quantize_weights(mlp.coefs_[i])
print("   Weight quantization applied!")

# ==============================================================
# 4. EVALUATE
# ==============================================================
print("\n[3/4] Model evaluate ho raha hai...")

y_pred_train = igzo_ann.predict(X_train)
y_pred_test  = igzo_ann.predict(X_test)

acc_train = accuracy_score(y_train, y_pred_train) * 100
acc_test  = accuracy_score(y_test,  y_pred_test)  * 100

print(f"\n   Train Accuracy : {acc_train:.2f}%")
print(f"   Test  Accuracy : {acc_test:.2f}%")
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred_test, digits=4))

# Save model
joblib.dump(igzo_ann, os.path.join(MODEL_DIR, "igzo_digit_model.pkl"))
print(f"   Model saved: models/igzo_digit_model.pkl")

# ==============================================================
# 5. VISUALIZATIONS
# ==============================================================
print("\n[4/4] Results visualize ho rahe hain...")

plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 11), facecolor='#08080f')
gs  = gridspec.GridSpec(2, 4, figure=fig,
                        hspace=0.45, wspace=0.38,
                        left=0.06, right=0.98,
                        top=0.92, bottom=0.06)

PANEL = '#0c0c1e'
GRID  = '#1c1c3a'
C = ['#00f5d4', '#f72585', '#ffd60a', '#4cc9f0', '#7209b7']

def ax_style(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=7)
    if xlabel: ax.set_xlabel(xlabel, color='#bbbbbb', fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color='#bbbbbb', fontsize=9)
    ax.tick_params(colors='#999999', labelsize=8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color('#2a2a4a')

# Plot 1: Sample predictions (correct)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(PANEL)
ax1.axis('off')
ax1.set_title('Sample Predictions', color='white', fontsize=10, fontweight='bold')
correct_idx = np.where(y_pred_test == y_test)[0][:16]
for k, idx in enumerate(correct_idx):
    ax_sub = fig.add_axes([
        0.06 + (k % 4) * 0.055,
        0.58 - (k // 4) * 0.075,
        0.048, 0.065
    ])
    ax_sub.imshow(X_test[idx].reshape(28, 28), cmap='plasma', interpolation='nearest')
    ax_sub.set_title(f'{y_pred_test[idx]}', color=C[0], fontsize=7, pad=1)
    ax_sub.axis('off')

# Plot 2: Confusion Matrix
ax2 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_test, y_pred_test)
im = ax2.imshow(cm, cmap='plasma', aspect='auto')
ax2.set_xticks(range(10))
ax2.set_yticks(range(10))
ax2.set_xticklabels(range(10), color='#999999', fontsize=8)
ax2.set_yticklabels(range(10), color='#999999', fontsize=8)
for i in range(10):
    for j in range(10):
        ax2.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color='white' if cm[i, j] < cm.max()/2 else 'black', fontsize=6)
ax_style(ax2, f'Confusion Matrix (Acc={acc_test:.1f}%)',
         'Predicted', 'True Label')
plt.colorbar(im, ax=ax2, fraction=0.04)

# Plot 3: Per-digit accuracy
ax3 = fig.add_subplot(gs[0, 2])
per_digit = [accuracy_score(y_test[y_test == d], y_pred_test[y_test == d]) * 100
             for d in range(10)]
bars = ax3.bar(range(10), per_digit, color=plt.cm.cool(np.linspace(0.1, 0.9, 10)))
ax3.axhline(acc_test, color=C[1], ls='--', lw=1.5, label=f'Overall={acc_test:.1f}%')
ax3.set_ylim(80, 101)
for i, (bar, val) in enumerate(zip(bars, per_digit)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}', ha='center', va='bottom', color='white', fontsize=6.5)
ax_style(ax3, 'Per-Digit Accuracy', 'Digit (0-9)', 'Accuracy (%)')
ax3.set_facecolor(PANEL)
ax3.grid(True, color=GRID, alpha=0.5)
ax3.legend(fontsize=8)

# Plot 4: IGZO conductance states (weight distribution)
ax4 = fig.add_subplot(gs[0, 3])
all_weights = np.concatenate([w.ravel() for w in mlp.coefs_])
ax4.hist(all_weights, bins=80, color=C[3], alpha=0.8, edgecolor='none')
ax4.axvline(0, color='white', ls='--', lw=1, alpha=0.5)
ax_style(ax4, 'IGZO Synaptic Weight Distribution',
         'Weight Value', 'Count')
ax4.set_facecolor(PANEL)
ax4.grid(True, color=GRID, alpha=0.5)

# Plot 5: Wrong predictions
ax5 = fig.add_subplot(gs[1, 0])
ax5.set_facecolor(PANEL)
ax5.axis('off')
ax5.set_title('Wrong Predictions', color=C[1], fontsize=10, fontweight='bold')
wrong_idx = np.where(y_pred_test != y_test)[0][:16]
for k, idx in enumerate(wrong_idx):
    ax_sub2 = fig.add_axes([
        0.06 + (k % 4) * 0.055,
        0.08 + (1 - k // 4) * 0.075 - 0.05,
        0.048, 0.065
    ])
    ax_sub2.imshow(X_test[idx].reshape(28, 28), cmap='hot', interpolation='nearest')
    ax_sub2.set_title(f'T:{y_test[idx]} P:{y_pred_test[idx]}',
                      color=C[1], fontsize=5.5, pad=1)
    ax_sub2.axis('off')

# Plot 6: Learning curve
ax6 = fig.add_subplot(gs[1, 1])
loss_curve = mlp.loss_curve_
val_scores  = mlp.validation_scores_ if hasattr(mlp, 'validation_scores_') else None
ax6.plot(loss_curve, color=C[0], lw=2, label='Train Loss')
if val_scores is not None:
    ax6b = ax6.twinx()
    ax6b.plot(val_scores, color=C[2], lw=2, ls='--', label='Val Acc')
    ax6b.set_ylabel('Val Accuracy', color=C[2], fontsize=9)
    ax6b.tick_params(colors=C[2])
ax_style(ax6, 'Learning Curve', 'Epoch', 'Loss')
ax6.set_facecolor(PANEL)
ax6.grid(True, color=GRID, alpha=0.5)
ax6.legend(fontsize=8, loc='upper right')

# Plot 7: IGZO conductance states
ax7 = fig.add_subplot(gs[1, 2])
n_states_arr = np.arange(igzo_layer.n_states)
ax7.step(n_states_arr, igzo_layer.G_states, color=C[0], lw=2, where='mid')
ax7.fill_between(n_states_arr, igzo_layer.G_states, alpha=0.2, color=C[0], step='mid')
ax_style(ax7, 'IGZO Conductance States',
         'State Index', 'Normalized Conductance')
ax7.set_facecolor(PANEL)
ax7.grid(True, color=GRID, alpha=0.5)

# Plot 8: Summary
ax8 = fig.add_subplot(gs[1, 3])
ax8.axis('off')
rows = [
    ['Train Accuracy',  f'{acc_train:.2f}%'],
    ['Test  Accuracy',  f'{acc_test:.2f}%'],
    ['Architecture',    '784→300→100→10'],
    ['Synaptic States', '32 (IGZO-based)'],
    ['Training Epochs', str(mlp.n_iter_)],
    ['Total Params',    f'{sum(w.size for w in mlp.coefs_):,}'],
    ['Best Digit',      f'{np.argmax(per_digit)} ({max(per_digit):.1f}%)'],
    ['Worst Digit',     f'{np.argmin(per_digit)} ({min(per_digit):.1f}%)'],
]
tbl = ax8.table(cellText=rows, colLabels=['Metric', 'Value'],
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
tbl.scale(1, 1.85)
ax8.set_title('Model Summary', color='white', fontsize=10, fontweight='bold', pad=8)

fig.suptitle('IGZO-Inspired Synaptic ANN — MNIST Digit Recognition',
             color='white', fontsize=13, fontweight='bold', y=0.97)

plt.savefig(os.path.join(MODEL_DIR, 'digit_recognition_results.png'),
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())

print("\n" + "=" * 62)
print(f"  DIGIT RECOGNITION COMPLETE!")
print(f"  Train Accuracy : {acc_train:.2f}%")
print(f"  Test  Accuracy : {acc_test:.2f}%")
print(f"  Model saved    : models/igzo_digit_model.pkl")
print("=" * 62)

plt.show()
