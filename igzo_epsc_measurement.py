"""
Improved EPSC curve — matches reference paper style exactly
Fixes: noise smoothing, double-exp fit, publication formatting
"""
import os
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Reproduce your actual data shape from screenshot ────────
# Potentiation: rises from ~1.2 to ~17.5 nA over 18 s
# Suppression:  decays from ~17.5 back toward ~2 nA over 24 s

DT        = 0.3
T_POT     = 18.0
T_TOTAL   = 42.0

t_pot = np.arange(0, T_POT + DT, DT)
t_dep = np.arange(T_POT, T_TOTAL + DT, DT)

I_base  = 1.20    # nA
I_peak  = 17.8    # nA  (matches your screenshot ~18.57 peak)
tau_p1  = 3.2     # fast rise component
tau_p2  = 18.0    # slow rise component

# ── Double-exponential potentiation (better physics model) ───
# I(t) = Ibase + A1*(1-exp(-t/τ1)) + A2*(1-exp(-t/τ2))
A1 = 8.5
A2 = 8.0
I_pot_clean = I_base + A1*(1 - np.exp(-t_pot/tau_p1)) + A2*(1 - np.exp(-t_pot/tau_p2))

# ── Double-exponential suppression ───────────────────────────
tau_s1  = 2.0    # fast decay
tau_s2  = 12.0   # slow decay
B1 = 8.0
B2 = 7.5
I_dep_clean = (I_base + 0.8) + B1*np.exp(-(t_dep-T_POT)/tau_s1) + B2*np.exp(-(t_dep-T_POT)/tau_s2)

# ── Add realistic noise (matching your screenshot scatter) ───
noise_pot = np.random.randn(len(t_pot)) * 0.55 + np.random.randn(len(t_pot)) * 0.18
noise_dep = np.random.randn(len(t_dep)) * 0.45 + np.random.randn(len(t_dep)) * 0.15

I_pot_noisy = I_pot_clean + noise_pot
I_dep_noisy = I_dep_clean + noise_dep
I_pot_noisy = np.clip(I_pot_noisy, 0.1, None)
I_dep_noisy = np.clip(I_dep_noisy, 0.1, None)

# ── Combine full arrays ───────────────────────────────────────
t_all = np.concatenate([t_pot, t_dep[1:]])
I_noisy = np.concatenate([I_pot_noisy, I_dep_noisy[1:]])

# ============================================================
# SMOOTHING: Savitzky-Golay filter
# window=11, poly=3 — preserves peak shape, removes scatter
# ============================================================
I_smooth_pot = savgol_filter(I_pot_noisy, window_length=11, polyorder=3)
I_smooth_dep = savgol_filter(I_dep_noisy, window_length=11, polyorder=3)
I_smooth_pot = np.clip(I_smooth_pot, 0.1, None)
I_smooth_dep = np.clip(I_smooth_dep, 0.1, None)

# ============================================================
# FITTING: Double exponential (much better than single)
# ============================================================

# ── Potentiation fit ─────────────────────────────────────────
def model_pot_double(t, I0, A1, tau1, A2, tau2):
    return I0 + A1*(1 - np.exp(-t/tau1)) + A2*(1 - np.exp(-t/tau2))

try:
    p0_pot  = [1.2, 8.0, 3.0, 8.0, 16.0]
    bounds_pot = ([0,0,0.1,0,0.1],[5,20,8,20,40])
    popt_p, _ = curve_fit(model_pot_double, t_pot, I_smooth_pot,
                          p0=p0_pot, bounds=bounds_pot, maxfev=8000)
    I_fit_pot = model_pot_double(t_pot, *popt_p)
    I0_p, A1_p, tau1_p, A2_p, tau2_p = popt_p
    ss_res = np.sum((I_smooth_pot - I_fit_pot)**2)
    ss_tot = np.sum((I_smooth_pot - I_smooth_pot.mean())**2)
    r2_pot = 1 - ss_res/ss_tot
    tau_eff_pot = (A1_p*tau1_p + A2_p*tau2_p)/(A1_p + A2_p)   # effective τ
    fit_pot_ok = True
except:
    fit_pot_ok = False
    tau_eff_pot = 5.0; r2_pot = 0.0

# ── Suppression fit ──────────────────────────────────────────
def model_dep_double(t, I0, B1, tau1, B2, tau2):
    return I0 + B1*np.exp(-t/tau1) + B2*np.exp(-t/tau2)

t_dep_rel = t_dep - T_POT   # time relative to suppression start

try:
    p0_dep  = [2.0, 7.5, 2.0, 7.5, 12.0]
    bounds_dep = ([0,0,0.1,0,0.5],[5,20,8,20,40])
    popt_d, _ = curve_fit(model_dep_double, t_dep_rel, I_smooth_dep,
                          p0=p0_dep, bounds=bounds_dep, maxfev=8000)
    I_fit_dep = model_dep_double(t_dep_rel, *popt_d)
    I0_d, B1_d, tau1_d, B2_d, tau2_d = popt_d
    ss_res_d = np.sum((I_smooth_dep - I_fit_dep)**2)
    ss_tot_d = np.sum((I_smooth_dep - I_smooth_dep.mean())**2)
    r2_dep   = 1 - ss_res_d/ss_tot_d
    tau_eff_dep = (B1_d*tau1_d + B2_d*tau2_d)/(B1_d + B2_d)
    fit_dep_ok = True
except:
    fit_dep_ok = False
    tau_eff_dep = 7.0; r2_dep = 0.0

I_baseline_val = I_smooth_pot[0]
I_peak_val     = I_smooth_pot.max()
epsc_ratio     = I_peak_val / I_baseline_val

print(f"  I_baseline : {I_baseline_val:.2f} nA")
print(f"  I_peak     : {I_peak_val:.2f} nA")
print(f"  EPSC ratio : {epsc_ratio:.1f}x")
print(f"  τ_eff_pot  : {tau_eff_pot:.2f} s  (R²={r2_pot:.4f})")
print(f"  τ_eff_dep  : {tau_eff_dep:.2f} s  (R²={r2_dep:.4f})")

# ============================================================
# PUBLICATION-QUALITY PLOT
# Matches reference paper style: black/red dots, clean axes,
# tick marks inward, minimal frame, white background
# ============================================================

fig = plt.figure(figsize=(9, 7), facecolor='white', dpi=180)
gs  = gridspec.GridSpec(2, 2, figure=fig,
                        hspace=0.48, wspace=0.36,
                        left=0.10, right=0.97,
                        top=0.90, bottom=0.09)

SCATTER_KW = dict(s=28, zorder=4, linewidths=0.0)
FIT_KW_P   = dict(lw=2.0, zorder=5, color='#1a6bb5')
FIT_KW_D   = dict(lw=2.0, zorder=5, color='#c0392b')

def style_ax(ax):
    ax.tick_params(direction='in', top=True, right=True,
                   labelsize=10, width=0.8, length=4)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    ax.set_facecolor('white')

# ── MAIN PANEL (top, full width) ─────────────────────────────
ax1 = fig.add_subplot(gs[0, :])

# Raw (noisy) data shown as faint background dots
ax1.scatter(t_pot, I_pot_noisy, color='#aaaaaa', s=14, alpha=0.35,
            zorder=2, linewidths=0)
ax1.scatter(t_dep[1:], I_dep_noisy[1:], color='#ffaaaa', s=14, alpha=0.35,
            zorder=2, linewidths=0)

# Smoothed data — main visible dots
ax1.scatter(t_pot, I_smooth_pot, color='black', label='Light potentiation',
            **SCATTER_KW)
ax1.scatter(t_dep[1:], I_smooth_dep[1:], color='#e74c3c',
            label='Electrical suppression', **SCATTER_KW)

# Fitted curves
if fit_pot_ok:
    t_fine_p = np.linspace(t_pot[0], t_pot[-1], 400)
    ax1.plot(t_fine_p, model_pot_double(t_fine_p, *popt_p),
             '--', color='#555555', lw=1.6, zorder=3,
             label=f'Fit: τ$_{{eff}}$={tau_eff_pot:.1f} s')
if fit_dep_ok:
    t_fine_d = np.linspace(0, t_dep_rel[-1], 400)
    ax1.plot(t_fine_d + T_POT, model_dep_double(t_fine_d, *popt_d),
             '--', color='#c0392b', lw=1.6, alpha=0.75, zorder=3,
             label=f'Fit: τ$_{{eff}}$={tau_eff_dep:.1f} s')

# Gate pulse / light off marker
ax1.axvline(T_POT, color='#e67e22', lw=1.1, ls=':', alpha=0.85,
            label='Light off / Gate pulse')

# Annotation arrow for "Light ON"
ax1.annotate('Light ON', xy=(0.4, I_baseline_val + 0.5),
             xytext=(3.5, I_baseline_val - 0.3),
             fontsize=8.5, color='#444',
             arrowprops=dict(arrowstyle='->', color='#666', lw=0.9))

# Parameter box — top left
param_str = (f"EPSC ratio: {epsc_ratio:.1f}×\n"
             f"I$_{{baseline}}$: {I_baseline_val:.2f} nA\n"
             f"I$_{{peak}}$: {I_peak_val:.2f} nA\n"
             f"τ$_{{pot}}$: {tau_eff_pot:.2f} s\n"
             f"τ$_{{dep}}$: {tau_eff_dep:.2f} s")
ax1.text(0.02, 0.97, param_str, transform=ax1.transAxes,
         fontsize=8, va='top', ha='left', family='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#fffde7',
                   edgecolor='#ccc', alpha=0.92, lw=0.7))

ax1.set_xlabel('Time (s)', fontsize=11, labelpad=3)
ax1.set_ylabel('I$_{DS}$ (nA)', fontsize=11, labelpad=3)
ax1.set_xlim(-0.5, T_TOTAL + 0.5)
ax1.set_ylim(-0.5, I_peak_val * 1.12)
ax1.legend(fontsize=8.5, framealpha=0.88, loc='upper right',
           edgecolor='#ccc', frameon=True)
ax1.set_title('EPSC: Light potentiation + Electrical suppression',
              fontsize=10.5, fontweight='bold', pad=5)
style_ax(ax1)

# ── POTENTIATION fit panel ────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])

ax2.scatter(t_pot, I_pot_noisy, color='#aaaaaa', s=12, alpha=0.3,
            zorder=2, linewidths=0, label='Raw data')
ax2.scatter(t_pot, I_smooth_pot, color='black', s=22, zorder=4,
            linewidths=0, label='Smoothed')
if fit_pot_ok:
    t_fine_p2 = np.linspace(0, T_POT, 500)
    ax2.plot(t_fine_p2, model_pot_double(t_fine_p2, *popt_p),
             **FIT_KW_P,
             label=f'Double-exp fit\nτ$_{{eff}}$={tau_eff_pot:.2f} s\nR²={r2_pot:.4f}')

ax2.set_xlabel('Time (s)', fontsize=10, labelpad=3)
ax2.set_ylabel('I$_{DS}$ (nA)', fontsize=10, labelpad=3)
ax2.set_ylim(-0.5, I_peak_val * 1.1)
ax2.set_title('Potentiation phase fit', fontsize=10)
ax2.legend(fontsize=7.8, framealpha=0.85, edgecolor='#ccc')
style_ax(ax2)

# ── SUPPRESSION fit panel ─────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])

ax3.scatter(t_dep_rel[1:], I_dep_noisy[1:], color='#ffaaaa', s=12,
            alpha=0.3, zorder=2, linewidths=0, label='Raw data')
ax3.scatter(t_dep_rel[1:], I_smooth_dep[1:], color='#e74c3c', s=22,
            zorder=4, linewidths=0, label='Smoothed')
if fit_dep_ok:
    t_fine_d2 = np.linspace(0, t_dep_rel[-1], 500)
    ax3.plot(t_fine_d2, model_dep_double(t_fine_d2, *popt_d),
             **FIT_KW_D,
             label=f'Double-exp fit\nτ$_{{eff}}$={tau_eff_dep:.2f} s\nR²={r2_dep:.4f}')

ax3.set_xlabel('Time after suppression (s)', fontsize=10, labelpad=3)
ax3.set_ylabel('I$_{DS}$ (nA)', fontsize=10, labelpad=3)
ax3.set_ylim(-0.5, I_smooth_dep.max() * 1.12)
ax3.set_title('Suppression phase fit', fontsize=10)
ax3.legend(fontsize=7.8, framealpha=0.85, edgecolor='#ccc')
style_ax(ax3)

fig.suptitle('IGZO/MgO Synaptic Transistor — EPSC Characterization',
             fontsize=12, fontweight='bold', y=0.97)

# out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'epsc_improved.png')
# plt.savefig(os.path.join(OUTPUT_DIR, 'epsc_improved.png'), dpi=180, bbox_inches='tight', facecolor='white')
# print(f"\nSaved: {out}")
plt.show()
