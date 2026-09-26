#!/usr/bin/env python3

# SI Figure 4: LJ pressure dependence

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.style.use('tableau-colorblind10')


def extract_pressure(name):
    m = re.search(r"P([0-9.]+)", name)
    return float(m.group(1)) if m else 0.0


def find_pmf_files(data_dir, rs, T=0.775, eps=0.01):
    patterns = [
        f"pmf_T{T:.3f}_P*_Rs{rs:.1f}_D0{eps:05.2f}.dat",
        f"pmf_T{T:.3f}_P*_Rs{rs:.1f}_D00.01.dat",
    ]
    files = []
    seen = set()
    for pattern in patterns:
        for path in data_dir.glob(pattern):
            if path not in seen:
                files.append(path)
                seen.add(path)
    return sorted(files, key=lambda p: extract_pressure(p.name))


# --- settings ---
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
data_dir = ROOT / 'LJ PMF_data'
rs_values = [1.0, 4.0]
T_target = 0.775
eps_target = 0.01
output_pdf = HERE / 'SI_figure_4.pdf'

# Re-zero each PMF using a far-field plateau.
baseline_windows = {
    1.0: (3.5, 5.0),
    4.0: (8.5, 9.5),
}

# Plot limits tuned to the available LJ data.
xlims = {
    1.0: (-0.5, 5.0),
    4.0: (-0.5, 5.0),
}
ylims = {
    1.0: (-6.0, 1.0),
    4.0: (-56.0, 4.0),
}

pmf_files_by_rs = {rs: find_pmf_files(data_dir, rs, T=T_target, eps=eps_target)
                   for rs in rs_values}

if not any(pmf_files_by_rs.values()):
    raise FileNotFoundError(
        f'No files found in {data_dir} for T={T_target}, eps_sw={eps_target}.'
    )


# --- plotting style ---
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
})


# --- plot ---
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharex=False, sharey=False)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
panel_labels = ['(a)', '(b)']

for ax, rs, panel in zip(axes, rs_values, panel_labels):
    pmf_files = pmf_files_by_rs[rs]

    if not pmf_files:
        ax.text(
            0.5, 0.5,
            rf'No files found for $R_s={rs:g}\sigma$',
            transform=ax.transAxes,
            ha='center', va='center', fontsize=10
        )
        ax.set_title(rf'$R_s={rs:g}\sigma$', fontsize=14)
        continue

    for i, f in enumerate(pmf_files):
        P = extract_pressure(f.name)
        data = np.loadtxt(f, comments='#')
        x, W = data[:, 0], data[:, 1]

        # Re-zero from a far-field baseline if possible.
        lo, hi = baseline_windows[rs]
        mask = np.isfinite(x) & np.isfinite(W) & (x >= lo) & (x <= hi)
        if np.count_nonzero(mask) >= 5:
            W = W - np.mean(W[mask])

        ax.plot(
            x, W,
            lw=2.0,
            color=colors[i % len(colors)],
            label=rf'$P={P:g}$'
        )

    ax.axhline(0.0, linestyle='--', linewidth=1.2, color='k', alpha=0.7)
    ax.set_title(rf'$R_s={rs:g}\sigma$', fontsize=14, pad=6)
    ax.set_xlim(*xlims[rs])
    ax.set_ylim(*ylims[rs])
    ax.set_xlabel(r'$(r-2R_s)/\sigma$', fontsize=15)
    ax.tick_params(axis='both', which='both', direction='in', top=False, right=True, labelsize=13)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.minorticks_on()
    ax.text(-0.10, 1.04, panel, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=14, clip_on=False)

    # Keep legend clear of the data.
    if rs == 1.0:
        ax.legend(loc='lower right', frameon=False, fontsize=11, handlelength=2.2)
    else:
        ax.legend(loc='center right', frameon=False, fontsize=11, handlelength=2.2)

axes[0].set_ylabel(r'$\beta W(r)$', fontsize=15)

plt.tight_layout()
plt.savefig(output_pdf, bbox_inches='tight')
plt.close(fig)