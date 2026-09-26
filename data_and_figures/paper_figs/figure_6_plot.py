#!/usr/bin/env python3
"""Plot LJ-solvent PMFs versus solute size and pressure.

The script expects the supplied ``pmf_*.dat`` files in the same directory.
Each PMF is defined only up to an additive constant, so the mean over a
long-range plateau is subtracted before plotting.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "LJ PMF_data"
OUTPUT_PDF = HERE / "figure_6.pdf"

TARGET_T = 0.775
TARGET_EPSILON_SW = 0.01
SIZE_PRESSURE = 0.0
PRESSURE_RADIUS = 2.0
PRESSURES = (0.0, 0.5, 1.0)

# The supplied files do not all have the same usable far-field extent. In
# particular, the Rs=0.5 and 0.7 data jump to a spurious second plateau beyond
# about 5.5 sigma, whereas the Rs=4 PMF has a genuine large additive offset and
# reaches its far-field plateau later. These windows select the stable region
# before the boundary artefacts in each data set.
ZERO_WINDOWS = {
    0.5: (4.0, 5.0),
    0.6: (4.0, 5.0),
    0.7: (4.0, 5.0),
    1.0: (4.0, 5.0),
    2.0: (6.0, 7.0),
    3.0: (4.0, 5.0),
    4.0: (5.0, 7.0),
}

FILENAME = re.compile(
    r"^pmf_T(?P<T>[-+]?\d*\.?\d+)"
    r"_P(?P<P>[-+]?\d*\.?\d+)"
    r"_Rs(?P<Rs>[-+]?\d*\.?\d+)"
    r"_D0(?P<eps>[-+]?\d*\.?\d+)\.dat$"
)


def load_curves():
    """Return all finite PMFs at the target temperature and affinity."""
    curves = []
    for path in DATA_DIR.glob("pmf_*.dat"):
        match = FILENAME.match(path.name)
        if match is None:
            continue

        values = {key: float(value) for key, value in match.groupdict().items()}
        if not (
            np.isclose(values["T"], TARGET_T)
            and np.isclose(values["eps"], TARGET_EPSILON_SW)
        ):
            continue

        data = np.loadtxt(path, comments="#")
        finite = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
        data = data[finite, :2]
        if len(data) == 0:
            raise ValueError(f"No finite data in {path.name}")

        curves.append((values["P"], values["Rs"], data, path.name))
    return curves


def set_zero_at_long_range(data, radius):
    """Subtract the arbitrary PMF constant using a stable far-field window.

    The radius-dependent windows avoid the nonphysical jumps and drifts in the
    final part of several supplied files.
    """
    x, y = data[:, 0], data[:, 1]
    matched_radius = next(
        (target for target in ZERO_WINDOWS if np.isclose(radius, target)),
        None,
    )
    if matched_radius is None:
        raise ValueError(f"No normalization window specified for Rs={radius:g}")
    lower, upper = ZERO_WINDOWS[matched_radius]
    plateau = (x >= lower) & (x <= upper)
    if plateau.sum() < 5:
        raise ValueError("Too few points in the long-range normalization window")
    offset = float(y[plateau].mean())
    return np.column_stack((x, y - offset))


curves = load_curves()

size_curves = sorted(
    (
        (radius, set_zero_at_long_range(data, radius), name)
        for pressure, radius, data, name in curves
        if np.isclose(pressure, SIZE_PRESSURE)
    ),
    key=lambda item: item[0],
)

pressure_curves = {}
for pressure, radius, data, name in curves:
    if np.isclose(radius, PRESSURE_RADIUS) and any(
        np.isclose(pressure, target) for target in PRESSURES
    ):
        pressure_curves[pressure] = (set_zero_at_long_range(data, radius), name)

if not size_curves:
    raise FileNotFoundError("No P=0 size-series PMFs were found")
for pressure in PRESSURES:
    if not any(np.isclose(pressure, found) for found in pressure_curves):
        raise FileNotFoundError(f"Missing Rs=2 PMF at P={pressure:g}")


# Tableau's colourblind palette gives distinct traces in print and on screen.
plt.style.use("tableau-colorblind10")
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 13,
        "axes.labelsize": 15,
        "legend.fontsize": 11,
        "legend.title_fontsize": 11,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), constrained_layout=True)

# (a) Smooth growth of the attraction and its range with solute radius.
ax = axes[0]
for radius, data, _ in size_curves:
    ax.plot(data[:, 0], data[:, 1], label=rf"${radius:g}$")
ax.axhline(0.0, color="black", linewidth=0.7, zorder=0)
ax.set_xlim(-0.5, 5.0)
ax.set_ylim(-57.0, 4.0)
ax.set_xlabel(r"$(r-2R_s)/\sigma$")
ax.set_ylabel(r"$W(r)/(k_{\mathrm{B}}T)$")
ax.set_title(r"$\mathbf{(a)}\quad P=0$", loc="left", fontsize=14, pad=7)
# Magnify the molecular-solute PMFs, whose structure is compressed by the
# much deeper minima for the larger radii in the main panel.
axins = inset_axes(
    ax,
    width="44%",
    height="38%",
    loc="lower right",
    bbox_to_anchor=(-0.015, 0.07, 1.0, 1.0),
    bbox_transform=ax.transAxes,
    borderpad=0.8,
)
for radius, data, _ in size_curves:
    if any(np.isclose(radius, target) for target in (0.5, 0.6, 0.7)):
        axins.plot(data[:, 0], data[:, 1], linewidth=1.5)
axins.axhline(0.0, color="black", linewidth=0.55, zorder=0)
axins.set_xlim(-0.35, 2.0)
axins.set_ylim(-3.1, 0.8)
axins.set_xticks((0, 1, 2))
axins.set_yticks((-3, -2, -1, 0))
axins.tick_params(direction="in", top=False, right=True, labelsize=9, pad=1.5)

# (b) Pressure weakens and shortens the solvent-mediated attraction.
ax = axes[1]
for target in PRESSURES:
    pressure = next(found for found in pressure_curves if np.isclose(found, target))
    data, _ = pressure_curves[pressure]
    ax.plot(data[:, 0], data[:, 1], label=rf"${pressure:g}$")
ax.axhline(0.0, color="black", linewidth=0.7, zorder=0)
ax.set_xlim(-0.5, 5.0)
ax.set_ylim(-19.5, 2.0)
ax.set_xlabel(r"$(r-2R_s)/\sigma$")
ax.set_title(r"$\mathbf{(b)}\quad R_s=2\sigma$", loc="left", fontsize=14, pad=7)
ax.legend(
    title=r"$P$",
    frameon=False,
    loc="lower right",
    handlelength=2.0,
    labelspacing=0.3,
    borderaxespad=0.55,
)

for ax in axes:
    ax.tick_params(direction="in", top=False, right=True)

fig.savefig(OUTPUT_PDF, bbox_inches="tight")
plt.close(fig)

print(f"Saved {OUTPUT_PDF}")
print("Size-series files:")
for radius, _, name in size_curves:
    print(f"  Rs={radius:g}: {name}")
print("Pressure-series files:")
for pressure in PRESSURES:
    found = next(value for value in pressure_curves if np.isclose(value, pressure))
    print(f"  P={pressure:g}: {pressure_curves[found][1]}")
