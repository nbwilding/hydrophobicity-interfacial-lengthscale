#!/usr/bin/env python3

import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def extract_pressure(name):
    m = re.search(r"P([0-9.]+)", name)
    return float(m.group(1)) if m else 0.0


def find_pmf_files(data_dir, rs):
    pattern = f"pmf_T300.0_P*_Rs{rs:.1f}_D00.01.dat"
    return sorted(data_dir.glob(pattern), key=lambda p: extract_pressure(p.name))


# --- read data ---
data_dir = Path(".")
rs_values = [3.0, 10.0]
pmf_files_by_rs = {rs: find_pmf_files(data_dir, rs) for rs in rs_values}

if not any(pmf_files_by_rs.values()):
    raise FileNotFoundError(
        "No PMF files found for either Rs=3.0 or Rs=10.0 in the current directory."
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

# --- plot ---
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharex=False, sharey=False)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for ax, rs in zip(axes, rs_values):
    pmf_files = pmf_files_by_rs[rs]

    if not pmf_files:
        ax.text(
            0.5, 0.5,
            rf"No files found for $R_s={int(rs)}\,\mathrm{{\AA}}$",
            transform=ax.transAxes,
            ha="center", va="center", fontsize=10
        )
        ax.set_title(rf"$R_s={int(rs)}\,\mathrm{{\AA}}$", fontsize=14)
        continue

    for i, f in enumerate(pmf_files):
        data = np.loadtxt(f, comments="#")
        x, W = data[:, 0], data[:, 1]

        ax.plot(
            x, W,
            lw=2.0,
            color=colors[i % len(colors)],
            label=rf"$P={int(extract_pressure(f.name))}$ atm"
        )
    ax.axhline(0.0, linestyle='--', linewidth=1.2, color='k', alpha=0.7)
    ax.set_title(rf"$R_s={int(rs)}\,\mathrm{{\AA}}$", fontsize=14)

    if rs == 3.0:
        ax.set_xlim(-0.5, 3)
        ax.set_ylim(-4, 3)
    elif rs == 10.0:
        ax.set_xlim(-1, 6)
        ax.set_ylim(-50, 10)

    ax.set_xlabel(r"$r-2R_s\;(\mathrm{\AA})$", fontsize=15)
    ax.tick_params(axis='both', which='both',
                   direction='in', top=True, right=True, labelsize=13)

axes[0].set_ylabel(r"$W(r)/k_{\mathrm{B}}T$", fontsize=15)

axes[0].xaxis.set_major_locator(MultipleLocator(1))
axes[0].xaxis.set_major_formatter(FormatStrFormatter('%d'))

# single legend
handles, labels = [], []
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        break
if handles:
    fig.legend(handles, labels,
               loc="upper center",
               ncol=min(len(labels), 5),
               frameon=False,
               fontsize=11,
               bbox_to_anchor=(0.5, 1.02))

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("pmf_pressure.png", dpi=600, bbox_inches="tight")
plt.savefig("pmf_pressure.pdf")
plt.show()


