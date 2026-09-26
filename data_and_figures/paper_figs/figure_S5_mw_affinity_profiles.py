#!/usr/bin/env python3

# SI Figure 5: mW affinity-dependent single-solute density and local-compressibility profiles
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


# Plot finite-affinity single-solute density and local-compressibility
# profiles for R_s = 3 and 10 Angstrom. Run this script in the directory
# containing the matching rho_profile_*.dat and kappa_profile_*.dat files.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
input_directory = ROOT / "mW_Rho-Kappa_data"
output_pdf = HERE / "SI_figure_5.pdf"

T_target = 300.0
P_target = 0.0

D0_targets = {
    3.0: (0.01, 0.20, 0.40, 0.60),
    10.0: (0.01, 0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40),
}
all_D0_values = sorted({D0 for values in D0_targets.values() for D0 in values})

filename_pattern = re.compile(
    r"^(?P<quantity>rho|kappa)_profile"
    r"_T(?P<T>[-+]?\d*\.?\d+)"
    r"_P(?P<P>[-+]?\d*\.?\d+)"
    r"_Rs(?P<Rs>[-+]?\d*\.?\d+)"
    r"_D0(?P<D0>[-+]?\d*\.?\d+)\.dat$"
)

profiles = {
    Rs: {"rho": {}, "kappa": {}}
    for Rs in D0_targets
}

for file_path in input_directory.glob("*_profile_*.dat"):
    match = filename_pattern.match(file_path.name)
    if match is None:
        continue

    T = float(match.group("T"))
    P = float(match.group("P"))
    Rs = float(match.group("Rs"))
    D0 = float(match.group("D0"))

    Rs_key = next(
        (target for target in D0_targets if np.isclose(Rs, target)),
        None,
    )
    if Rs_key is None or not (
        np.isclose(T, T_target)
        and np.isclose(P, P_target)
        and any(np.isclose(D0, target) for target in D0_targets[Rs_key])
    ):
        continue

    data = np.loadtxt(file_path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{file_path.name} must contain at least two columns")

    finite = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
    D0_key = next(
        target for target in D0_targets[Rs_key] if np.isclose(D0, target)
    )
    profiles[Rs_key][match.group("quantity")][D0_key] = data[finite, :2]

for Rs, quantities in profiles.items():
    for quantity, available in quantities.items():
        missing = [D0 for D0 in D0_targets[Rs] if D0 not in available]
        if missing:
            raise FileNotFoundError(
                f"Missing {quantity} profiles for R_s={Rs:g} and "
                f"epsilon_sw={missing}"
            )

plt.style.use("tableau-colorblind10")
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],
        "text.latex.preamble": r"\usepackage{lmodern}\usepackage[T1]{fontenc}",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
    }
)

palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colour_for_D0 = {
    D0: palette[index % len(palette)]
    for index, D0 in enumerate(all_D0_values)
}

fig, axes = plt.subplots(
    2,
    2,
    figsize=(7.0, 4.65),
    sharex="col",
    gridspec_kw={"hspace": 0.08, "wspace": 0.24},
)

panel_labels = (("(a)", "(b)"), ("(c)", "(d)"))

for column, Rs in enumerate(D0_targets):
    ax_rho = axes[0, column]
    ax_kappa = axes[1, column]

    for D0 in D0_targets[Rs]:
        rho = profiles[Rs]["rho"][D0]
        kappa = profiles[Rs]["kappa"][D0]

        # Omit bins whose centres lie inside the nominal solvent-accessible
        # surface; these are especially sensitive to radial binning.
        rho = rho[rho[:, 0] >= 0]
        kappa = kappa[kappa[:, 0] >= 0]

        label = rf"$\epsilon_{{sw}}={D0:.2f}$"
        colour = colour_for_D0[D0]
        ax_rho.plot(rho[:, 0], rho[:, 1], color=colour, label=label)
        ax_kappa.plot(kappa[:, 0], kappa[:, 1], color=colour)

    for row, ax in enumerate((ax_rho, ax_kappa)):
        ax.axhline(1.0, color="black", linewidth=0.75, zorder=0)
        ax.set_xlim(0, 6)
        ax.set_xticks(np.arange(0, 7, 2))
        ax.tick_params(direction="in", top=True, right=True)
        ax.text(0.035, 0.92, panel_labels[row][column], transform=ax.transAxes)

    ax_rho.set_title(rf"$R_s={Rs:g}\,\mathrm{{\AA}}$")
    ax_kappa.set_xlabel(
        r"$r-(R_s+\sigma_{\mathrm{mW}}/2)\;[\mathrm{\AA}]$"
    )
    ax_rho.legend(
        frameon=False,
        ncol=1 if Rs == 3.0 else 2,
        loc="upper right",
        columnspacing=0.8,
        handlelength=2.2,
    )

axes[0, 0].set_ylim(0, 4.05)
axes[1, 0].set_ylim(-6, 11)
axes[0, 1].set_ylim(0, 4.25)
axes[1, 1].set_ylim(-11, 40)

axes[0, 0].set_ylabel(r"$\varrho(r)/\rho_b$")
axes[1, 0].set_ylabel(r"$\kappa(r)/\kappa_T$")

fig.savefig(output_pdf, bbox_inches="tight")
plt.close(fig)

print(f"Saved {output_pdf}")
for Rs, values in D0_targets.items():
    print(
        f"R_s={Rs:g}; epsilon_sw plotted: "
        + ", ".join(f"{value:.2f}" for value in values)
    )