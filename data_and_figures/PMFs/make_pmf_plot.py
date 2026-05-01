#!/usr/bin/env python3

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.style.use('tableau-colorblind10')

def extract_rs(filename):
    m = re.search(r"Rs([0-9]+(?:\.[0-9]+)?)", filename)
    return float(m.group(1))


# --- Find files in current directory ---
files = sorted(
    Path(".").glob("pmf_T300.0_P0.0_Rs*_D00.01.dat"),
    key=lambda p: extract_rs(p.name)
)

if not files:
    raise RuntimeError("No PMF files found in current directory.")

# --- Matplotlib style (PNAS-like) ---
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, ax = plt.subplots(figsize=(7.2, 5.0))

small_rs = {1.4, 1.7, 2.0, 2.5, 3.0}
inset_data = []

# --- Main plot ---
for path in files:
    rs = extract_rs(path.name)
    data = np.loadtxt(path, comments="#")
    x, y = data[:, 0], data[:, 1]

    ax.plot(x, y, lw=1.8)

    if any(abs(rs - val) < 1e-6 for val in small_rs):
        inset_data.append((rs, x, y))

# --- Axes ---
ax.text(
    0.94, 0.98, r"$\mathbf{(a)}$",
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=15
)


ax.set_xlabel(r"$r - 2R_s\;(\mathrm{\AA})$", fontsize=16)
ax.set_ylabel(r"$W(r)/k_{\mathrm{B}}T$", fontsize=16)
ax.set_xlim(-1.0, 6.0)
ax.set_ylim(-50.0, 8.0)
ax.tick_params(labelsize=12)

# --- Inset ---
axins = inset_axes(
    ax,
    width="60%", height="60%",
    bbox_to_anchor=(0.59, 0.06, 0.60, 0.60),
    bbox_transform=ax.transAxes,
    loc="lower left"
)

for rs, x, y in inset_data:
    axins.plot(x, y, lw=1.5)

axins.set_xlim(-1.0, 6.0)
axins.set_ylim(-3.0, 1.0)
axins.tick_params(labelsize=9)

# --- Save ---
fig.tight_layout()
fig.savefig("pmf_figure.png", dpi=300)
fig.savefig("pmf_figure.pdf")

plt.show()
