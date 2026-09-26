#!/usr/bin/env python3

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# SI Figure 1: TIP4P/2005 PMFs showing size, affinity and pressure dependence.
# Full available x-range is shown over the chosen common plotting window.

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "TIP4P PMF_data"
OUTFILE = HERE / "SI_figure_1.pdf"

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
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 9.0,
})
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

pattern = re.compile(
    r"pmf_T(?P<T>[0-9.]+)_P(?P<P>[0-9.]+)_R(?P<R>[0-9.]+)_D0(?P<eps>[0-9.]+)\.dat$"
)

class DataSource:
    def __init__(self):
        self.data_dir = DATA_DIR
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Cannot find TIP4P PMF data directory:\n  {self.data_dir}"
            )

    def names(self):
        return [p.name for p in self.data_dir.glob("*.dat")]

    def find_name(self, T, P, R, eps):
        hits = []
        for name in self.names():
            m = pattern.match(name)
            if m is None:
                continue
            g = {k: float(v) for k, v in m.groupdict().items()}
            if (np.isclose(g["T"], T) and np.isclose(g["P"], P)
                    and np.isclose(g["R"], R) and np.isclose(g["eps"], eps)):
                hits.append(name)
        if len(hits) != 1:
            raise FileNotFoundError(
                f"Expected one match in {self.data_dir} for "
                f"T={T}, P={P}, R={R}, eps={eps}; got {hits}"
            )
        return hits[0]

    def load(self, T, P, R, eps):
        name = self.find_name(T, P, R, eps)
        data = np.loadtxt(self.data_dir / name, comments="#")
        finite = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
        return data[finite, 0], data[finite, 1]

source = DataSource()
T0 = 298.15
P0 = 1.0
size_series = [1.6, 1.8, 2.0, 3.0, 6.0]
affinities = [0.01, 0.20, 0.60]
pressures = [1.0, 500.0, 1000.0, 2000.0]

fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.1))
ax1, ax2, ax3, ax4 = axs.ravel()
for ax in axs.ravel():
    ax.axhline(0.0, ls="--", lw=1.1, color="0.35")
    ax.tick_params(axis="both", which="both", top=True, right=True)

# (a) size dependence
for i, R in enumerate(size_series):
    x, y = source.load(T0, P0, R, 0.01)
    ax1.plot(x, y, lw=1.8, color=colors[i % len(colors)],
             label=rf"$R_s={R:g}\,\mathrm{{\AA}}$")
ax1.set_title(r"TIP4P/2005", fontsize=17, pad=8)
ax1.text(0.12, 0.96, "(a)", transform=ax1.transAxes, ha="left", va="top", fontsize=14)
ax1.text(0.97, 0.94, r"$P=1\ \mathrm{atm},\ \epsilon_{sw}=0.01\ \mathrm{kcal/mol}$",
         transform=ax1.transAxes, ha="right", va="top", fontsize=9.5)
ax1.legend(frameon=False, loc="lower right", ncol=2,
           handlelength=2.0, columnspacing=0.8)
ax1.set_ylabel(r"$W(r)/k_{\mathrm{B}}T$")

# (b) affinity dependence, Rs=6
for i, eps in enumerate(affinities):
    x, y = source.load(T0, P0, 6.0, eps)
    ax2.plot(x, y, lw=1.8, color=colors[i], label=rf"$\epsilon_{{sw}}={eps:.2f}$")
ax2.set_title(r"$R_s=6\,\mathrm{\AA}$", fontsize=17, pad=8)
ax2.text(0.12, 0.96, "(b)", transform=ax2.transAxes, ha="left", va="top", fontsize=14)
ax2.text(0.97, 0.94, r"$P=1\ \mathrm{atm}$",
         transform=ax2.transAxes, ha="right", va="top", fontsize=10.5)
ax2.legend(frameon=False, loc="lower right", handlelength=2.0)

# (c) pressure dependence, Rs=3. Omit P=1000 atm from this panel
# because the stored dataset is anomalous/truncated and does not display
# meaningfully over the plotting window.
pressures_panel_c = [1.0, 500.0, 2000.0]
for i, P in enumerate(pressures_panel_c):
    x, y = source.load(T0, P, 3.0, 0.01)
    plab = f"{int(P)}" if P >= 10 else f"{P:g}"
    color_index = pressures.index(P)
    ax3.plot(x, y, lw=1.8, color=colors[color_index], label=rf"$P={plab}\ \mathrm{{atm}}$")
ax3.set_title(r"$R_s=3\,\mathrm{\AA}$", fontsize=17, pad=8)
ax3.text(0.12, 0.96, "(c)", transform=ax3.transAxes, ha="left", va="top", fontsize=14)
ax3.text(0.97, 0.94, r"$\epsilon_{sw}=0.01\ \mathrm{kcal/mol}$",
         transform=ax3.transAxes, ha="right", va="top", fontsize=10.0)
ax3.legend(frameon=False, loc="lower right", handlelength=2.0)
ax3.set_ylabel(r"$W(r)/k_{\mathrm{B}}T$")
ax3.set_xlabel(r"$r-2R_s\ (\mathrm{\AA})$")

# (d) pressure dependence, Rs=6
for i, P in enumerate(pressures):
    x, y = source.load(T0, P, 6.0, 0.01)
    plab = f"{int(P)}" if P >= 10 else f"{P:g}"
    ax4.plot(x, y, lw=1.8, color=colors[i], label=rf"$P={plab}\ \mathrm{{atm}}$")
ax4.set_title(r"$R_s=6\,\mathrm{\AA}$", fontsize=17, pad=8)
ax4.text(0.12, 0.96, "(d)", transform=ax4.transAxes, ha="left", va="top", fontsize=14)
ax4.text(0.97, 0.94, r"$\epsilon_{sw}=0.01\ \mathrm{kcal/mol}$",
         transform=ax4.transAxes, ha="right", va="top", fontsize=10.0)
ax4.legend(frameon=False, loc="lower right", handlelength=2.0)
ax4.set_xlabel(r"$r-2R_s\ (\mathrm{\AA})$")

# Use a common displayed x range.
for ax in axs.ravel():
    ax.set_xlim(-0.7, 7.0)
    ax.xaxis.set_major_locator(MultipleLocator(2.0))
    ax.margins(y=0.05)

# Requested y ranges.
for ax in (ax1, ax2, ax4):
    ax.set_ylim(-20, 5)
ax3.set_ylim(top=5)

fig.tight_layout(h_pad=2.0, w_pad=1.8)
fig.savefig(OUTFILE, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {OUTFILE}")
