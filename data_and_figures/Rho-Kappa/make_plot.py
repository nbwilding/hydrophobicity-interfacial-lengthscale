#!/usr/bin/env python3

import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.style.use("tableau-colorblind10")


def extract_rs(name):
    m = re.search(r"Rs([0-9]+(?:\.[0-9]+)?)", name)
    return float(m.group(1)) if m else 0.0


# --- read data from current directory ---
data_dir = Path(".")

rho_files = sorted(
    [f for f in data_dir.glob("rho_profile_*.dat") if "D00.01" in f.name],
    key=lambda p: extract_rs(p.name)
)

kappa_files = sorted(
    [f for f in data_dir.glob("kappa_profile_*.dat") if "D00.01" in f.name],
    key=lambda p: extract_rs(p.name)
)

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.2), sharex=True)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

handles = []
labels = []

# ===== (a) rho =====
for i, f in enumerate(rho_files):
    Rs = extract_rs(f.name)
    data = np.loadtxt(f, comments="#")
    x, y = data[:, 0], data[:, 1]

    line, = ax1.plot(
        x, y,
        lw=2.0,
        color=colors[i % len(colors)],
        label=rf"$R_s={Rs}\,\mathrm{{\AA}}$"
    )

    handles.append(line)
    labels.append(rf"$R_s={Rs}\,\mathrm{{\AA}}$")

ax1.set_xlim(0, 10)
ax1.set_xlabel(r"$r-(R_s+\sigma_{\mathrm{mw}}/2)\;(\mathrm{\AA})$", fontsize=15)
ax1.set_ylabel(r"$\rho(r)/\rho_b$", fontsize=15)
ax1.tick_params(axis='both', which='both',
                direction='in', top=True, right=True, labelsize=13)

# ===== (b) kappa =====
for i, f in enumerate(kappa_files):
    data = np.loadtxt(f, comments="#")
    x, y = data[:, 0], data[:, 1]

    ax2.plot(
        x, y,
        lw=2.0,
        color=colors[i % len(colors)]
    )

ax2.set_xlim(0, 10)
ax2.set_ylim(None, 40)
ax2.set_xlabel(r"$r-(R_s+\sigma_{\mathrm{mw}}/2)\;(\mathrm{\AA})$", fontsize=15)
ax2.set_ylabel(r"$\kappa(r)/\kappa_T$", fontsize=15)

ax1.text(0.04, 0.97, r"$\mathbf{(a)}$", transform=ax1.transAxes,
         fontsize=14, va='top', ha='left')
ax2.text(0.04, 0.97, r"$\mathbf{(b)}$", transform=ax2.transAxes,
         fontsize=14, va='top', ha='left')

# consistent ticks
for ax in (ax1, ax2):
    ax.xaxis.set_major_locator(MultipleLocator(2))

# shared legend above panels (no overlap)
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=min(len(labels), 4),
    frameon=False,
    fontsize=11,
    bbox_to_anchor=(0.5, 1.02)
)

plt.tight_layout(rect=[0, 0, 1, 0.92])

plt.savefig("rho_kappa_D0_0.01.png", dpi=600, bbox_inches="tight")
plt.savefig("rho_kappa_D0_0.01.pdf")

plt.show()