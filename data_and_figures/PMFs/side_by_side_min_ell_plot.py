#!/usr/bin/env python3

import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def extract_rs(name: str):
    m = re.search(r"Rs([0-9]+(?:\.[0-9]+)?)", name)
    return float(m.group(1)) if m else None


# --- read PMF data ---
data_dir = Path(".")
pmf_files = sorted(
    data_dir.glob("pmf_T300.0_P0.0_Rs*_D00.01.dat"),
    key=lambda p: extract_rs(p.name)
)

Rs_vals = []
depths = []

for f in pmf_files:
    Rs = extract_rs(f.name)
    data = np.loadtxt(f, comments="#")

    x = data[:, 0]
    W = data[:, 1]

    # --- find first local minimum ---
    first_min = None
    for i in range(1, len(W) - 1):
        if W[i] < W[i - 1] and W[i] < W[i + 1]:
            first_min = W[i]
            break

    # fallback if no local minimum found
    if first_min is None:
        first_min = np.min(W)

    Rs_vals.append(Rs)
    depths.append(abs(first_min))  # magnitude

Rs_vals = np.array(Rs_vals)
depths = np.array(depths)

# --- read ell data ---
ell_candidates = [Path("ell_vals.dat"), Path("ell_vals")]
ell_file = next((p for p in ell_candidates if p.exists()), None)
if ell_file is None:
    raise FileNotFoundError("Could not find ell_vals.dat or ell_vals in the working directory.")

ell_data = np.loadtxt(ell_file, comments="#")
if ell_data.ndim == 1:
    ell_data = ell_data.reshape(1, -1)

ell_Rs = ell_data[:, 0]
ell_vals = ell_data[:, 1]
ell_errs = ell_data[:, 2] if ell_data.shape[1] > 2 else None

# --- plotting style ---
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
})

# --- side-by-side plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.0), sharex=False)

# Left panel: PMF first minimum depth

ax1.plot(Rs_vals, depths, marker='o', linestyle='none')
ax1.set_xlabel(r"$R_s\;(\mathrm{\AA})$", fontsize=14)
ax1.set_ylabel(r"$\left|W_{\min}\right|/k_{\mathrm{B}}T$", fontsize=14)
ax1.tick_params(labelsize=11)
ax1.text(0.04, 0.95, r"$\mathbf{(b)}$", transform=ax1.transAxes, fontsize=15, va="top")

# Right panel: ell vs Rs with error bars
ax2.errorbar(ell_Rs, ell_vals, yerr=ell_errs, marker='o',  linestyle='none', capsize=3)
ax2.set_xlabel(r"$R_s\;(\mathrm{\AA})$", fontsize=14)
ax2.set_ylabel(r"$\ell\;(\mathrm{\AA})$", fontsize=14)
ax2.tick_params(labelsize=11)
ax2.text(0.04, 0.95, r"$\mathbf{(c)}$", transform=ax2.transAxes, fontsize=15, va="top")

plt.tight_layout(w_pad=2.0)

# --- save ---
plt.savefig("pmf_first_min_and_ell_vs_Rs.png", dpi=300, bbox_inches="tight")
plt.savefig("pmf_first_min_and_ell_vs_Rs.pdf", bbox_inches="tight")

plt.show()
