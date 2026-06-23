"""
==============================================================
STEP 3: Interactive Digit Recognition GUI App
==============================================================
Ye script ek interactive app banati hai jisme:
  - Canvas pe digit draw karo mouse se
  - IGZO-trained model digit predict karega
  - Confidence scores dikhayega
  - IGZO device response simulate karega

Run karo: python 3_app.py
(Pehle 1_train_model.py aur 2_digit_recognition.py chalao!)
==============================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw, ImageFilter
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = "models"

# ==============================================================
# Load Models
# ==============================================================
def load_models():
    models = {}
    try:
        # TRUE IGZO model prefer karo — agar nahi mila to standard try karo
        true_model_path = os.path.join(MODEL_DIR, "igzo_TRUE_digit_model.pkl")
        std_model_path  = os.path.join(MODEL_DIR, "igzo_digit_model.pkl")

        if os.path.exists(true_model_path):
            raw = joblib.load(true_model_path)
            # TRUE model ek dict hai — wrap karo
            models['digit']      = raw
            models['digit_type'] = 'TRUE_IGZO'
            print("[OK] TRUE IGZO model loaded! (100% tera transistor)")
        elif os.path.exists(std_model_path):
            models['digit']      = joblib.load(std_model_path)
            models['digit_type'] = 'STANDARD'
            print("[OK] Standard digit model loaded!")
        else:
            raise FileNotFoundError("Koi bhi digit model nahi mila!")

        models['poly']   = joblib.load(os.path.join(MODEL_DIR, "poly_model.pkl"))
        models['gain']   = joblib.load(os.path.join(MODEL_DIR, "gain_model.pkl"))
        models['params'] = joblib.load(os.path.join(MODEL_DIR, "device_params.pkl"))
        print("[OK] All models loaded!")
        return models
    except FileNotFoundError as e:
        messagebox.showerror("Error",
            f"Model file nahi mili: {e}\n\n"
            "Pehle ye run karo:\n"
            "  python 1_train_model.py\n"
            "  python 2b_digit_recognition_IGZO_TRUE.py")
        return None


# ==============================================================
# Main App
# ==============================================================
class IGZOApp:
    def __init__(self, root, models):
        self.root    = root
        self.models  = models
        self.root.title("IGZO Synaptic Transistor — TRUE IGZO Digit Recognition")
        self.root.configure(bg='#0a0a1a')
        self.root.geometry("1100x720")
        self.root.resizable(True, True)

        # Drawing state
        self.drawing    = False
        self.last_x     = None
        self.last_y     = None
        self.canvas_size = 280
        self.brush_size  = 18

        # PIL image for processing
        self.pil_image  = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.pil_draw   = ImageDraw.Draw(self.pil_image)

        self._build_ui()
        self._update_device_info()

    # ── UI Construction ────────────────────────────────────────
    def _build_ui(self):
        # ── Header ──
        hdr = tk.Label(self.root,
            text="IGZO/MgO Synaptic Transistor  ·  Neuromorphic Digit Recognition",
            font=("Consolas", 13, "bold"), fg="#00f5d4", bg="#0a0a1a")
        hdr.pack(pady=(10, 4))

        sub = tk.Label(self.root,
            text="Draw a digit (0–9) on the canvas below",
            font=("Consolas", 9), fg="#888888", bg="#0a0a1a")
        sub.pack(pady=(0, 8))

        # ── Main frame ──
        main = tk.Frame(self.root, bg="#0a0a1a")
        main.pack(fill="both", expand=True, padx=12, pady=4)

        # Left: canvas + controls
        left = tk.Frame(main, bg="#0a0a1a")
        left.pack(side="left", fill="y", padx=(0, 10))

        canvas_label = tk.Label(left, text="✏  Draw Here",
            font=("Consolas", 10, "bold"), fg="#ffd60a", bg="#0a0a1a")
        canvas_label.pack(pady=(0, 4))

        self.canvas = tk.Canvas(left,
            width=self.canvas_size, height=self.canvas_size,
            bg="#111111", cursor="crosshair",
            highlightthickness=2, highlightbackground="#00f5d4")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>",   self._start_draw)
        self.canvas.bind("<B1-Motion>",       self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_draw)

        # Brush size
        bs_frame = tk.Frame(left, bg="#0a0a1a")
        bs_frame.pack(pady=6, fill="x")
        tk.Label(bs_frame, text="Brush:", fg="#aaaaaa", bg="#0a0a1a",
                 font=("Consolas", 8)).pack(side="left")
        self.brush_var = tk.IntVar(value=self.brush_size)
        bs_scale = ttk.Scale(bs_frame, from_=8, to=30,
                             variable=self.brush_var, orient="horizontal",
                             length=120, command=self._update_brush)
        bs_scale.pack(side="left", padx=6)
        self.brush_lbl = tk.Label(bs_frame, text=f"{self.brush_size}px",
                                   fg=C0, bg="#0a0a1a", font=("Consolas", 8))
        self.brush_lbl.pack(side="left")

        # Buttons
        btn_frame = tk.Frame(left, bg="#0a0a1a")
        btn_frame.pack(pady=6)

        self._btn(btn_frame, "⚡  PREDICT", self._predict,
                  bg="#00f5d4", fg="#0a0a1a").pack(side="left", padx=4)
        self._btn(btn_frame, "🗑  CLEAR",   self._clear,
                  bg="#f72585", fg="white").pack(side="left", padx=4)

        # Prediction display
        self.pred_label = tk.Label(left,
            text="?", font=("Consolas", 72, "bold"),
            fg="#ffd60a", bg="#0a0a1a")
        self.pred_label.pack(pady=4)

        self.conf_label = tk.Label(left,
            text="Draw a digit and press PREDICT",
            font=("Consolas", 9), fg="#888888", bg="#0a0a1a",
            wraplength=280)
        self.conf_label.pack()

        # Right: matplotlib plots
        right = tk.Frame(main, bg="#0a0a1a")
        right.pack(side="left", fill="both", expand=True)

        self.fig, self.axes = plt.subplots(2, 2, figsize=(7, 5.5),
                                            facecolor="#08080f")
        self.fig.subplots_adjust(hspace=0.45, wspace=0.38,
                                  left=0.1, right=0.97,
                                  top=0.92, bottom=0.1)

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.mpl_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._init_plots()

        # Device info bar
        info_frame = tk.Frame(self.root, bg="#111133")
        info_frame.pack(fill="x", padx=12, pady=(4, 8))
        self.info_lbl = tk.Label(info_frame, text="", font=("Consolas", 8),
                                  fg="#00f5d4", bg="#111133")
        self.info_lbl.pack(pady=3)

    def _btn(self, parent, text, cmd, bg, fg):
        return tk.Button(parent, text=text, command=cmd,
                         font=("Consolas", 9, "bold"),
                         bg=bg, fg=fg, activebackground=bg,
                         relief="flat", padx=12, pady=6, cursor="hand2")

    # ── Drawing ───────────────────────────────────────────────
    def _start_draw(self, event):
        self.drawing = True
        self.last_x  = event.x
        self.last_y  = event.y

    def _draw(self, event):
        if not self.drawing:
            return
        x, y = event.x, event.y
        r = self.brush_var.get()
        self.canvas.create_oval(x-r, y-r, x+r, y+r,
                                 fill="white", outline="white")
        if self.last_x is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                     fill="white", width=r*2, capstyle="round")
        self.pil_draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
        if self.last_x is not None:
            self.pil_draw.line([self.last_x, self.last_y, x, y],
                                fill=255, width=r*2)
        self.last_x, self.last_y = x, y

    def _stop_draw(self, event):
        self.drawing = False
        self.last_x  = None
        self.last_y  = None

    def _update_brush(self, val):
        self.brush_lbl.config(text=f"{int(float(val))}px")

    def _clear(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.pil_draw  = ImageDraw.Draw(self.pil_image)
        self.pred_label.config(text="?", fg="#ffd60a")
        self.conf_label.config(text="Draw a digit and press PREDICT")
        self._init_plots()

    # ── Preprocessing ─────────────────────────────────────────
    def _preprocess(self):
        """Canvas image ko MNIST format mein convert karo"""
        img = self.pil_image.copy()
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        img = img.resize((28, 28), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    # ── Prediction ────────────────────────────────────────────
    def _predict(self):
        arr  = self._preprocess()
        flat = arr.ravel().reshape(1, -1)

        # Digit prediction — TRUE IGZO ya Standard dono handle karo
        if self.models.get('digit_type') == 'TRUE_IGZO':
            # TRUE IGZO model dict se predict karo
            d    = self.models['digit']
            A    = d['scaler'].transform(flat)
            for i, (W, b) in enumerate(zip(d['weights'], d['biases'])):
                Z = A @ W + b
                if i < len(d['weights']) - 1:
                    A = np.maximum(0, Z)           # ReLU
                else:
                    Z  = Z - Z.max(axis=1, keepdims=True)
                    ez = np.exp(Z)
                    A  = ez / ez.sum(axis=1, keepdims=True)  # Softmax
            proba = A[0]
        else:
            proba = self.models['digit'].predict_proba(flat)[0]

        pred = np.argmax(proba)
        conf = proba[pred] * 100

        # Simulate IGZO response
        vgs_range = np.linspace(-1, 5, 100)
        ids_pred  = 10 ** self.models['poly'].predict(vgs_range.reshape(-1,1))

        # Update UI
        color = "#00f5d4" if conf > 70 else "#ffd60a" if conf > 40 else "#f72585"
        self.pred_label.config(text=str(pred), fg=color)
        top3 = np.argsort(proba)[::-1][:3]
        top3_str = "  |  ".join([f"{i}: {proba[i]*100:.1f}%" for i in top3])
        self.conf_label.config(
            text=f"Confidence: {conf:.1f}%\nTop 3: {top3_str}",
            fg=color)

        self._update_plots(arr, proba, pred, vgs_range, ids_pred)

    # ── Plots ─────────────────────────────────────────────────
    def _init_plots(self):
        for ax in self.axes.ravel():
            ax.clear()
            ax.set_facecolor("#0c0c1e")
            for sp in ['top','right']: ax.spines[sp].set_visible(False)
            for sp in ['left','bottom']: ax.spines[sp].set_color('#2a2a4a')
            ax.tick_params(colors='#777777', labelsize=7)
        self.axes[0,0].set_title("Drawn Digit", color="white", fontsize=9, fontweight='bold')
        self.axes[0,0].text(0.5, 0.5, "Draw digit\nthen PREDICT",
                             ha='center', va='center', color='#555555',
                             fontsize=9, transform=self.axes[0,0].transAxes)
        self.axes[0,1].set_title("Confidence Scores", color="white", fontsize=9, fontweight='bold')
        self.axes[1,0].set_title("IGZO Transfer Curve", color="white", fontsize=9, fontweight='bold')
        self.axes[1,1].set_title("IGZO EPSC Response", color="white", fontsize=9, fontweight='bold')
        self.fig.suptitle("IGZO Device Response", color="white", fontsize=10, fontweight='bold')
        self.mpl_canvas.draw()

    def _update_plots(self, img_arr, proba, pred, vgs, ids):
        for ax in self.axes.ravel():
            ax.clear()
            ax.set_facecolor("#0c0c1e")
            for sp in ['top','right']: ax.spines[sp].set_visible(False)
            for sp in ['left','bottom']: ax.spines[sp].set_color('#2a2a4a')
            ax.tick_params(colors='#777777', labelsize=7)

        # Plot 1: Drawn digit
        ax = self.axes[0, 0]
        ax.imshow(img_arr, cmap='plasma', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f"Input Digit → Predicted: {pred}", color="#ffd60a",
                     fontsize=9, fontweight='bold')
        ax.axis('off')

        # Plot 2: Confidence bar chart
        ax = self.axes[0, 1]
        colors = ['#f72585' if i == pred else '#1a1a4a' for i in range(10)]
        bars = ax.bar(range(10), proba * 100, color=colors, edgecolor='#333355')
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(0, 105)
        ax.set_xticks(range(10))
        ax.set_xlabel("Digit", color='#aaaaaa', fontsize=8)
        ax.set_ylabel("Confidence %", color='#aaaaaa', fontsize=8)
        ax.set_title(f"Confidence Scores (Pred={pred}, {proba[pred]*100:.1f}%)",
                     color="white", fontsize=9, fontweight='bold')
        ax.grid(True, color='#1c1c3a', alpha=0.5, axis='y', linewidth=0.5)
        ax.axhline(50, color='white', ls=':', lw=0.8, alpha=0.3)

        # Plot 3: IGZO Transfer Curve
        ax = self.axes[1, 0]
        ax.semilogy(vgs, ids * 1e9, color='#00f5d4', lw=2)
        # Mark operating point based on confidence
        vop = -1 + (pred / 9.0) * 6
        ids_op = 10 ** self.models['poly'].predict([[vop]])[0] * 1e9
        ax.semilogy(vop, ids_op, 'o', color='#f72585', ms=10,
                    label=f'V_op={vop:.1f}V', zorder=5)
        ax.set_xlabel("V_GS (V)", color='#aaaaaa', fontsize=8)
        ax.set_ylabel("I_DS (nA)", color='#aaaaaa', fontsize=8)
        ax.set_title("IGZO Transfer Curve", color="white", fontsize=9, fontweight='bold')
        ax.grid(True, color='#1c1c3a', alpha=0.5, linewidth=0.5)
        ax.legend(fontsize=7, framealpha=0.2)

        # Plot 4: Simulated EPSC
        ax = self.axes[1, 1]
        t  = np.linspace(0, 50, 500)
        A  = proba[pred] * 10
        tau_d = 15 + pred * 1.2
        tau_r = 0.5
        epsc = A * (np.exp(-t/tau_d) - np.exp(-t/tau_r))
        epsc = np.clip(epsc, 0, None)
        ax.plot(t, epsc, color='#ffd60a', lw=2)
        ax.fill_between(t, epsc, alpha=0.2, color='#ffd60a')
        ax.set_xlabel("Time (ms)", color='#aaaaaa', fontsize=8)
        ax.set_ylabel("EPSC (nA)", color='#aaaaaa', fontsize=8)
        ax.set_title(f"Simulated Synaptic Response (Digit={pred})",
                     color="white", fontsize=9, fontweight='bold')
        ax.grid(True, color='#1c1c3a', alpha=0.5, linewidth=0.5)

        self.fig.suptitle(f"IGZO Device Analysis  —  Predicted: {pred}  ({proba[pred]*100:.1f}%)",
                          color="#00f5d4", fontsize=10, fontweight='bold')
        self.mpl_canvas.draw()

    # ── Device Info Bar ───────────────────────────────────────
    def _update_device_info(self):
        p = self.models['params']
        txt = (f"Device: IGZO/MgO TFT  |  "
               f"ION={p['ION']:.2e}A  |  "
               f"IOFF={p['IOFF']:.2e}A  |  "
               f"ION/IOFF={p['ION_IOFF']:.1e}  |  "
               f"Vth={p['Vth_V']:.2f}V  |  "
               f"SS={p['SS_mVdec']:.0f} mV/dec")
        self.info_lbl.config(text=txt)


# ── Color constants ────────────────────────────────────────────
C0 = "#00f5d4"

# ==============================================================
# Run App
# ==============================================================
if __name__ == "__main__":
    print("=" * 62)
    print("   IGZO Synaptic App — Starting...")
    print("=" * 62)

    models = load_models()
    if models is None:
        exit(1)

    root = tk.Tk()

    # Style ttk
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure("TScale", troughcolor="#1a1a3a", background="#0a0a1a")

    app = IGZOApp(root, models)
    print("\n[OK] App launched! Digit draw karo aur PREDICT dabao.")
    root.mainloop()
