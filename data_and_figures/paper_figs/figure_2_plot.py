#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ----------------------------------------------------------------------
# Revised PNAS Fig. 2
#
# Panels:
# (a) Rs = 3 A,  eps_sw = 0.01 kcal/mol, pressure dependence
# (b) Rs = 10 A, eps_sw = 0.01 kcal/mol, pressure dependence
# (c) Rs = 3 A,  P = 0 atm, affinity dependence
# (d) Rs = 3 A,  eps_sw = 0.20 kcal/mol, pressure dependence
#
# Output:
#     figure_2.pdf
# ----------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "mW PMF_data"
OUTFILE = HERE / "figure_2.pdf"

plt.style.use("tableau-colorblind10")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "text.latex.preamble": r"\usepackage{lmodern}\usepackage[T1]{fontenc}",
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 9.5,
})

# Ordered colour mapping used consistently across all pressure panels.
# Low pressure starts cool; higher pressure moves to warmer colours.
pressure_colors = {
    0:    "#4E79A7",  # blue
    500:  "#76B7B2",  # teal
    1000: "#F28E2B",  # orange
    2000: "#E15759",  # red
}

# Ordered mapping for affinity panel.
affinity_colors = {
    0.01: "#4E79A7",  # blue
    0.20: "#F28E2B",  # orange
    0.40: "#E15759",  # red
}



def load_pmf(filename):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required PMF file not found:\n  {path}\n"
            "Check that the script is in the PMFs directory and that "
            "../mW PMF_data contains the required data files."
        )
    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, 1]


def pressure_filename(Rs, P, eps):
    return f"pmf_T300.0_P{P:.1f}_Rs{Rs:.1f}_D0{eps:0.2f}.dat"


fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.1))
ax1, ax2, ax3, ax4 = axs.ravel()

for ax in axs.ravel():
    ax.axhline(0.0, ls="--", lw=1.1, color="0.35")
    ax.tick_params(axis="both", which="both", top=True, right=True)


# ----------------------------------------------------------------------
# (a) Rs = 3 A, eps_sw = 0.01, pressure dependence
# ----------------------------------------------------------------------
pressures = [0, 500, 1000, 2000]

for P in pressures:
    x, y = load_pmf(pressure_filename(3.0, P, 0.01))
    ax1.plot(x, y, lw=2.0, color=pressure_colors[P], label=rf"$P={P}$ atm")

ax1.set_xlim(-0.5, 3.0)
ax1.set_ylim(-4, 3)
ax1.set_title(r"$R_s=3\,\mathrm{\AA}$", fontsize=17, pad=8)
ax1.text(0.10, 0.95, "(a)", transform=ax1.transAxes,
         ha="left", va="top", fontsize=14)
ax1.text(0.96, 0.92,
         r"$\epsilon_{sw}=0.01\ \mathrm{kcal/mol}$",
         transform=ax1.transAxes, ha="right", va="top", fontsize=10.5)
ax1.legend(frameon=False, loc="lower right", handlelength=2.3)
ax1.set_ylabel(r"$W(r)/k_{\mathrm{B}}T$")
ax1.xaxis.set_major_locator(MultipleLocator(1))


# ----------------------------------------------------------------------
# (b) Rs = 10 A, eps_sw = 0.01, same pressures
# ----------------------------------------------------------------------
for P in pressures:
    x, y = load_pmf(pressure_filename(10.0, P, 0.01))
    ax2.plot(x, y, lw=2.0, color=pressure_colors[P], label=rf"$P={P}$ atm")

ax2.set_xlim(-1.0, 6.0)
ax2.set_ylim(-50, 15)
ax2.set_title(r"$R_s=10\,\mathrm{\AA}$", fontsize=17, pad=8)
ax2.text(0.10, 0.95, "(b)", transform=ax2.transAxes,
         ha="left", va="top", fontsize=14)
ax2.text(0.96, 0.94,
         r"$\epsilon_{sw}=0.01\ \mathrm{kcal/mol}$",
         transform=ax2.transAxes, ha="right", va="top", fontsize=10.5)
ax2.legend(frameon=False, loc="lower right", handlelength=2.3)
ax2.xaxis.set_major_locator(MultipleLocator(2))


# ----------------------------------------------------------------------
# (c) Rs = 3 A, P = 0 atm, affinity dependence
# eps_sw = 0.60 deliberately omitted
# ----------------------------------------------------------------------
affinities = [0.01, 0.20, 0.40]

for eps in affinities:
    filename = pressure_filename(3.0, 0, eps)
    x, y = load_pmf(filename)
    ax3.plot(x, y, lw=2.0, color=affinity_colors[eps],
             label=rf"$\epsilon_{{sw}}={eps:.2f}$")

ax3.set_xlim(-0.5, 3.0)
ax3.set_ylim(-4, 3)
ax3.set_title(r"$R_s=3\,\mathrm{\AA}$", fontsize=17, pad=8)
ax3.text(0.10, 0.95, "(c)", transform=ax3.transAxes,
         ha="left", va="top", fontsize=14)
ax3.text(0.96, 0.92, r"$P=0\ \mathrm{atm}$",
         transform=ax3.transAxes, ha="right", va="top", fontsize=10.5)
ax3.legend(frameon=False, loc="lower right", handlelength=2.3)
ax3.set_ylabel(r"$W(r)/k_{\mathrm{B}}T$")
ax3.set_xlabel(r"$r-2R_s\ (\mathrm{\AA})$")
ax3.xaxis.set_major_locator(MultipleLocator(1))


# ----------------------------------------------------------------------
# (d) Rs = 3 A, eps_sw = 0.20, pressure dependence
# P = 500 atm deliberately omitted, but the remaining pressures retain the
# same colours as in panels (a) and (b).
# ----------------------------------------------------------------------
pressures_d = [0, 1000, 2000]
for P in pressures_d:
    x, y = load_pmf(pressure_filename(3.0, P, 0.20))
    ax4.plot(x, y, lw=2.0, color=pressure_colors[P], label=rf"$P={P}$ atm")

ax4.set_xlim(-0.5, 3.0)
ax4.set_ylim(-4, 3)
ax4.set_title(r"$R_s=3\,\mathrm{\AA}$", fontsize=17, pad=8)
ax4.text(0.10, 0.95, "(d)", transform=ax4.transAxes,
         ha="left", va="top", fontsize=14)
ax4.text(0.96, 0.92,
         r"$\epsilon_{sw}=0.20\ \mathrm{kcal/mol}$",
         transform=ax4.transAxes, ha="right", va="top", fontsize=10.5)
ax4.legend(frameon=False, loc="lower right", handlelength=2.3)
ax4.set_xlabel(r"$r-2R_s\ (\mathrm{\AA})$")
ax4.xaxis.set_major_locator(MultipleLocator(1))


fig.tight_layout(h_pad=2.0, w_pad=1.8)
fig.savefig(OUTFILE, bbox_inches="tight")
plt.close(fig)

print(f"Wrote {OUTFILE}")
