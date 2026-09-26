#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

# --- paths ---
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "Subvolume stats"
OUT = HERE / "figure_5.pdf"

# --- load data ---
var_data = np.loadtxt(DATA_DIR / "var_vs_v.dat")
kappa_data = np.loadtxt(DATA_DIR / "kappa_vs_R.dat")

v = var_data[:, 0]
var = var_data[:, 1]
grad = np.gradient(var, v)

R = kappa_data[:, 0]
kappa_ratio = kappa_data[:, -2]
kappa_ratio_sem = kappa_data[:, -1]

# --- constants ---
kB = 1.380649e-23
T = 300.0
kappa_T = 4.794971e20      # A^3/J, NpT volume-fluctuation estimate
rho_b = 3.281263e-02       # A^-3, bulk number density
bulk_slope = (rho_b**2) * kB * T * kappa_T

# --- plotting style: compact layout used in the PNAS manuscript ---
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

label_fs = 7.5
tick_fs = 6.8
panel_fs = 7.5
lw = 1.0
dashed_color = "orange"

# (a) and (b) stacked on the left; (c) spans both rows on the right.
fig = plt.figure(figsize=(5.05, 2.25))
gs = fig.add_gridspec(
    nrows=2,
    ncols=2,
    width_ratios=(1.08, 0.92),
    height_ratios=(1.0, 1.0),
    wspace=0.28,
    hspace=0.14,
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
ax_c = fig.add_subplot(gs[:, 1])

# ---------- (a) Var(N_v) versus v ----------
ax_a.plot(v, var, lw=lw)
ax_a.plot(
    v,
    bulk_slope * v,
    linestyle="--",
    linewidth=0.9,
    alpha=0.8,
    color=dashed_color,
)
ax_a.set_xlim(0, np.max(v))
ax_a.set_ylim(0, 18)
ax_a.yaxis.set_major_locator(MultipleLocator(6))
ax_a.set_ylabel(r"$\mathrm{Var}(N_v)$", fontsize=label_fs)
ax_a.text(0.89, 0.74, r"$\mathbf{(a)}$", transform=ax_a.transAxes, fontsize=panel_fs)
ax_a.xaxis.set_major_locator(MaxNLocator(nbins=4))
ax_a.tick_params(top=False, right=True, labelsize=tick_fs, labelbottom=False)
ax_a.yaxis.set_minor_locator(plt.NullLocator())

# ---------- (b) d Var(N_v) / dv ----------
ax_b.plot(v, grad, lw=lw)
ax_b.axhline(
    bulk_slope,
    linestyle="--",
    linewidth=0.9,
    alpha=0.8,
    color=dashed_color,
)
ax_b.set_xlim(0, np.max(v))
ax_b.set_ylim(0, 0.01)
ax_b.set_yticks([0.0, 0.005, 0.010])
ax_b.set_ylabel(r"$d\,\mathrm{Var}(N_v)/dv$ ($\mathrm{\AA^{-3}}$)", fontsize=label_fs)
ax_b.set_xlabel(r"subvolume $v$ ($\mathrm{\AA^3}$)", fontsize=label_fs, labelpad=1.5)
ax_b.text(0.89, 0.77, r"$\mathbf{(b)}$", transform=ax_b.transAxes, fontsize=panel_fs)
ax_b.xaxis.set_major_locator(MaxNLocator(nbins=4))
ax_b.tick_params(top=False, right=True, labelsize=tick_fs)
ax_b.yaxis.set_minor_locator(plt.NullLocator())

# ---------- (c) scale-dependent compressibility ----------
ax_c.errorbar(
    R,
    kappa_ratio,
    yerr=kappa_ratio_sem,
    fmt="-",
    lw=lw,
    capsize=0,
)
ax_c.axhline(
    1.0,
    linestyle="--",
    linewidth=0.85,
    alpha=0.8,
    color=dashed_color,
)
ax_c.set_xlabel(r"$R$ ($\mathrm{\AA}$)", fontsize=label_fs, labelpad=1.5)
ax_c.set_ylabel(r"$\kappa_T(R)/\kappa_T$", fontsize=label_fs, labelpad=1)
ax_c.text(0.88, 0.90, r"$\mathbf{(c)}$", transform=ax_c.transAxes, fontsize=panel_fs)
ax_c.set_ylim(0, np.ceil(np.max(kappa_ratio + kappa_ratio_sem) * 1.02))
ax_c.set_xlim(np.floor(np.min(R)), np.ceil(np.max(R)))
ax_c.xaxis.set_major_locator(MultipleLocator(1))
ax_c.yaxis.set_major_locator(MultipleLocator(4))
ax_c.tick_params(top=False, right=True, labelsize=tick_fs)
ax_c.yaxis.set_minor_locator(plt.NullLocator())

for ax in (ax_a, ax_b, ax_c):
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"Saved {OUT}")
