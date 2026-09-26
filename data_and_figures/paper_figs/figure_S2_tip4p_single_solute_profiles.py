#!/usr/bin/env python3

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.style.use('tableau-colorblind10')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / 'TIP4P_Rho_Kappa_data'
OUT_PDF = HERE / 'SI_figure_2.pdf'
SIGMA = 3.1589  # Angstrom, oxygen LJ sigma in TIP4P/2005
T_TARGET = 298.15
P_TARGET = 1.0
RS_VALUES = [3.0, 6.0]
EPS_VALUES = [0.01, 0.40]

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Latin Modern Roman'],
    'text.latex.preamble': r'\usepackage{lmodern}\usepackage[T1]{fontenc}',
    'mathtext.fontset': 'cm',
    'axes.linewidth': 1.0,
    'axes.labelsize': 15,
    'font.size': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
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
    'legend.frameon': False,
    'lines.linewidth': 2.0,
})


def load_txt(subdir, basename):
    path = DATA_DIR / basename
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find required TIP4P profile file:\n  {path}"
        )
    return np.loadtxt(path, comments='#')


def density_name(rs, eps):
    return f'density-T{T_TARGET:.2f}-P{P_TARGET:.1f}-dr0.2-D0{eps:.2f}-Rs{rs:.1f}.txt'


def kappa_name(rs, eps):
    return f'kappa-T{T_TARGET:.2f}-P{P_TARGET:.1f}-dr0.2-D0{eps:.2f}-Rs{rs:.1f}.txt'


def bulk_density_from_tail(rho):
    # estimate from the last 10 bins
    return np.mean(rho[-10:])


fig, ax = plt.subplots(2, 2, figsize=(7.1, 5.6), sharex=False, sharey='row',
                       gridspec_kw={'hspace': 0.16, 'wspace': 0.12})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_map = {0.01: colors[0], 0.40: colors[1]}
panel_labels = [['(a)', '(b)'], ['(c)', '(d)']]

for j, rs in enumerate(RS_VALUES):
    xshift = rs + 0.5 * SIGMA

    # density row
    a = ax[0, j]
    for eps in EPS_VALUES:
        arr = load_txt('TIP4P_Rho_Kappa_data', density_name(rs, eps))
        r = arr[:, 0]
        rho = arr[:, 1]
        rho_std = arr[:, 2]
        rho_b = bulk_density_from_tail(rho)
        x = r - xshift
        y = rho / rho_b
        mask = np.isfinite(x) & np.isfinite(y) & (x >= -0.5) & (x <= 7.0)
        a.plot(x[mask], y[mask], color=color_map[eps],
               label=rf'$\epsilon_{{sw}}={eps:.2f}$')
    a.axhline(1.0, linestyle='--', linewidth=1.1, color='0.35')
    a.set_xlim(-0.5, 7.0)
    a.set_ylim(-0.05, 3.2)
    a.set_title(rf'$R_s={rs:g}\,\mathrm{{\AA}}$', fontsize=15, pad=6)
    a.tick_params(axis='both', which='both', top=False, right=True)
    a.text(0.12, 0.96, panel_labels[0][j], transform=a.transAxes,
           ha='left', va='top', fontsize=14)
    a.legend(loc='upper right', handlelength=2.0)

    # kappa row
    a = ax[1, j]
    for eps in EPS_VALUES:
        arr = load_txt('TIP4P_Rho_Kappa_data', kappa_name(rs, eps))
        r = arr[:, 0]
        kap = arr[:, 1]
        kap_std = arr[:, 2]
        x = r - xshift
        mask = np.isfinite(x) & np.isfinite(kap) & (x >= -0.5) & (x <= 7.0)
        a.plot(x[mask], kap[mask], color=color_map[eps])
    a.axhline(1.0, linestyle='--', linewidth=1.1, color='0.35')
    a.set_xlim(-0.5, 7.0)
    a.set_ylim(-4.0, 10.0)
    a.tick_params(axis='both', which='both', top=False, right=True)
    a.text(0.12, 0.96, panel_labels[1][j], transform=a.transAxes,
           ha='left', va='top', fontsize=14)
    a.set_xlabel(r'$r-(R_s+\sigma/2)\ (\mathrm{\AA})$')

ax[0, 0].set_ylabel(r'$\varrho(r)/\rho_b$')
ax[1, 0].set_ylabel(r'$\kappa(r)/\kappa_T$')

for i in range(2):
    for j in range(2):
        ax[i, j].xaxis.set_major_locator(MultipleLocator(1.0))
        if i == 0:
            ax[i, j].yaxis.set_major_locator(MultipleLocator(1.0))
        else:
            ax[i, j].yaxis.set_major_locator(MultipleLocator(2.0))

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches='tight')
plt.close(fig)
print(f'Saved {OUT_PDF}')
