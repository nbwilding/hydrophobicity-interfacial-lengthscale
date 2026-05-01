#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

# --- load data ---
var_data = np.loadtxt("var_vs_v.dat")
kappa_data = np.loadtxt("kappa_vs_R.dat")

v = var_data[:, 0]
var = var_data[:, 1]
grad = np.gradient(var, v)

R = kappa_data[:, 0]
kappa_ratio = kappa_data[:, -2]
kappa_ratio_sem = kappa_data[:, -1]

# --- constants ---
kB = 1.380649e-23
T = 300.0
kappa_T = 4.79e20          # A^3/J
rho_b = 3.2823e-02         # A^-3
bulk_slope = (rho_b**2) * kB * T * kappa_T  # A^-3

# --- plotting style ---
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

label_fs = 9
tick_fs = 8
panel_fs = 9
lw = 1.2

# PNAS one-column style
fig, axes = plt.subplots(3, 1, figsize=(3.35, 5.6), sharex=False)

# ---------- (a) Var vs v ----------
ax = axes[0]
ax.plot(v, var, lw=lw)
ax.set_ylabel(r"$\mathrm{Var}(N_v)$", fontsize=label_fs)
ax.set_xlabel(r"subvolume $v$ ($\mathrm{\AA^3}$)", fontsize=label_fs)
ax.text(0.9, 0.8, r"$\mathbf{(a)}$", transform=ax.transAxes, fontsize=panel_fs)

ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.tick_params(top=True, right=True, labelsize=tick_fs)
ax.yaxis.set_minor_locator(plt.NullLocator())

# ---------- (b) dVar/dv ----------
ax = axes[1]
ax.plot(v, grad, lw=lw)
ax.axhline(bulk_slope, linestyle="--", linewidth=lw)

ax.set_ylabel(
    r"$d\,\mathrm{Var}(N_v)/dv$ ($\mathrm{\AA^{-3}}$)",
    fontsize=label_fs,
)
ax.set_xlabel(r"subvolume $v$ ($\mathrm{\AA^3}$)", fontsize=label_fs)
ax.set_ylim(0, 0.012)
ax.set_yticks([0.0, 0.005, 0.010])
ax.text(0.9, 0.8, r"$\mathbf{(b)}$", transform=ax.transAxes, fontsize=panel_fs)

ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.tick_params(top=True, right=True, labelsize=tick_fs)
ax.yaxis.set_minor_locator(plt.NullLocator())

# ---------- (c) kappa vs R ----------
ax = axes[2]
ax.errorbar(
    R,
    kappa_ratio,
    yerr=kappa_ratio_sem,
    fmt="-",
    lw=lw,
    capsize=0,
)
ax.axhline(1.0, linestyle="--", linewidth=lw)

ax.set_xlabel(r"$R$ ($\mathrm{\AA}$)", fontsize=label_fs)
ax.set_ylabel(r"$\kappa_T(R)/\kappa_T$", fontsize=label_fs)
ax.text(0.9, 0.8, r"$\mathbf{(c)}$", transform=ax.transAxes, fontsize=panel_fs)

ax.set_ylim(0, np.ceil(np.max(kappa_ratio + kappa_ratio_sem) * 1.05))
ax.set_xlim(np.floor(np.min(R)), np.ceil(np.max(R)))

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.tick_params(top=True, right=True, labelsize=tick_fs)
ax.yaxis.set_minor_locator(plt.NullLocator())

# --- box spines ---
for ax in axes:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

# --- tighter PNAS-style layout ---
plt.subplots_adjust(
    hspace=0.28,
    left=0.24,
    right=0.98,
    bottom=0.08,
    top=0.98,
)

# --- save tightly cropped files ---
plt.savefig("three_panel_variance_kappa.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
plt.savefig("three_panel_variance_kappa.pdf", bbox_inches="tight", pad_inches=0.02)

plt.show()
