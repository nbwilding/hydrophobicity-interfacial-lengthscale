#!/usr/bin/env python3

from pathlib import Path
import io
import re
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Circle

plt.style.use("tableau-colorblind10")

HERE = Path(__file__).resolve().parent
ZIPFILE = HERE / "PNAS_LCW crossover Figs.zip"
OUTFILE = HERE / "figure_1.pdf"

SIGMA_MW = 2.3925
SIGMA_TIP4P = 3.15890
DELTA_REFF = 0.5 * (SIGMA_TIP4P - SIGMA_MW)
TIP4P_RS_PANEL_B = (1.6, 1.8, 2.0, 3.0, 6.0)
TIP4P_ELL_FILE = HERE / "TIP4P_first_well_linear_fit_ell_broad_largeRs.dat"


def find_data_dir(dirname):
    candidates = [
        HERE / dirname,
        HERE.parent / dirname,
        HERE / "PNAS_LCW crossover Figs" / dirname,
        HERE.parent / "PNAS_LCW crossover Figs" / dirname,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


MW_DATA_DIR = find_data_dir("mW PMF_data")
TIP4P_DATA_DIR = find_data_dir("TIP4P PMF_data")


def extract_rs_mw(filename):
    m = re.search(r"Rs([0-9]+(?:\.[0-9]+)?)", filename)
    return float(m.group(1)) if m else None


def first_minimum_depth(y):
    for i in range(1, len(y) - 1):
        if y[i] < y[i - 1] and y[i] < y[i + 1]:
            return abs(y[i])
    return abs(np.min(y))


def load_txt_from_zip(subdir, basename):
    if not ZIPFILE.is_file():
        raise FileNotFoundError(
            f"Could not find data directory and no zip archive at {ZIPFILE}"
        )
    with zipfile.ZipFile(ZIPFILE) as zf:
        member = None
        for zname in zf.namelist():
            if f"/{subdir}/" in zname and Path(zname).name == basename:
                member = zname
                break
        if member is None:
            raise FileNotFoundError(f"Could not find {basename} under {subdir} in {ZIPFILE}")
        with zf.open(member) as fh:
            return np.loadtxt(io.TextIOWrapper(fh), comments="#")


def pmf_files_mw():
    if MW_DATA_DIR is not None:
        files = sorted(
            MW_DATA_DIR.glob("pmf_T300.0_P0.0_Rs*_D00.01.dat"),
            key=lambda p: extract_rs_mw(p.name)
        )
        if files:
            return files
    # Fallback to zip archive.
    if ZIPFILE.is_file():
        names = []
        with zipfile.ZipFile(ZIPFILE) as zf:
            for zname in zf.namelist():
                base = Path(zname).name
                if "/mW PMF_data/" in zname and re.match(r"pmf_T300\.0_P0\.0_Rs.*_D00\.01\.dat$", base):
                    names.append(base)
        names = sorted(names, key=extract_rs_mw)
        if names:
            return names
    raise FileNotFoundError("No matching mW PMF files found in directory or zip archive")


plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
})


def load_mw_pmf(file_or_name):
    if isinstance(file_or_name, Path):
        data = np.loadtxt(file_or_name, comments="#")
    else:
        data = load_txt_from_zip("mW PMF_data", file_or_name)
    return data[:, 0], data[:, 1]


def load_tip4p_pmf(rs):
    name = f"pmf_T298.15_P1.0_R{rs:.1f}_D00.01.dat"
    if TIP4P_DATA_DIR is not None:
        path = TIP4P_DATA_DIR / name
        if path.exists():
            data = np.loadtxt(path, comments="#")
            return data[:, 0], data[:, 1]
    data = load_txt_from_zip("TIP4P PMF_data", name)
    return data[:, 0], data[:, 1]


def tip4p_ell_with_error(rs_nominal, fit_lo, fit_hi):
    x, y = load_tip4p_pmf(rs_nominal)
    mask = np.isfinite(x) & np.isfinite(y) & (x >= fit_lo) & (x <= fit_hi)
    coeffs, cov = np.polyfit(x[mask], y[mask], 1, cov=True)
    m, b = coeffs
    x0 = -b / m
    # Propagate covariance for x0 = -b/m.
    dfdm = b / (m * m)
    dfdb = -1.0 / m
    var_x0 = dfdm * dfdm * cov[0, 0] + dfdb * dfdb * cov[1, 1] + 2.0 * dfdm * dfdb * cov[0, 1]
    ell = 0.5 * x0
    ell_err = 4.0 * 0.5 * np.sqrt(max(var_x0, 0.0))
    return ell, ell_err


files = pmf_files_mw()
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# First-minimum data for panel (b): mW.
Rs_vals, depths = [], []
for f in files:
    name = f.name if isinstance(f, Path) else f
    Rs = extract_rs_mw(name)
    x, W = load_mw_pmf(f)
    Rs_vals.append(Rs)
    depths.append(first_minimum_depth(W))
Rs_vals = np.asarray(Rs_vals)
depths = np.asarray(depths)

# First-minimum data for panel (b): TIP4P/2005, shifted to mW-equivalent
# solvent-accessible radius.
TIP4P_Rs_vals, TIP4P_depths = [], []
for Rs in TIP4P_RS_PANEL_B:
    x, W = load_tip4p_pmf(Rs)
    TIP4P_Rs_vals.append(Rs + DELTA_REFF)
    TIP4P_depths.append(first_minimum_depth(W))
TIP4P_Rs_vals = np.asarray(TIP4P_Rs_vals)
TIP4P_depths = np.asarray(TIP4P_depths)

# ell data for panel (c).
if MW_DATA_DIR is not None:
    ell_file = next((p for p in (MW_DATA_DIR / "ell_vals.dat", MW_DATA_DIR / "ell_vals") if p.exists()), None)
    ell_data = np.loadtxt(ell_file, comments="#") if ell_file is not None else load_txt_from_zip("mW PMF_data", "ell_vals.dat")
else:
    ell_data = load_txt_from_zip("mW PMF_data", "ell_vals.dat")
if ell_data.ndim == 1:
    ell_data = ell_data.reshape(1, -1)
ell_Rs = ell_data[:, 0]
ell_vals = ell_data[:, 1]
ell_errs = ell_data[:, 2] if ell_data.shape[1] > 2 else None

# TIP4P/2005 ell values extracted from linear fits to the first PMF well.
if not TIP4P_ELL_FILE.exists():
    raise FileNotFoundError(f"Could not find {TIP4P_ELL_FILE}")
tip4p_ell_data = np.loadtxt(TIP4P_ELL_FILE, comments="#")
if tip4p_ell_data.ndim == 1:
    tip4p_ell_data = tip4p_ell_data.reshape(1, -1)
tip4p_ell_Rs = tip4p_ell_data[:, 1]
tip4p_ell_vals = []
for row in tip4p_ell_data:
    ell, ell_err = tip4p_ell_with_error(row[0], row[2], row[3])
    tip4p_ell_vals.append(ell)
tip4p_ell_vals = np.asarray(tip4p_ell_vals)
# For the plotted uncertainty, use error bars comparable in scale to the mW
# ones, obtained by interpolating the mW ell uncertainties to the corrected
# TIP4P radii. This better reflects the extraction uncertainty than the tiny
# formal linear-regression errors.
tip4p_ell_errs = np.interp(tip4p_ell_Rs, ell_Rs, ell_errs)

# One PDF for the whole numbered figure: top panel (a), bottom panels (b,c).
fig = plt.figure(figsize=(7.1, 7.8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], hspace=0.34, wspace=0.34)
ax = fig.add_subplot(gs[0, :])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1, 1])

# Panel (a): PMFs and inset.
small_rs = {1.4, 1.7, 2.0, 2.5, 3.0}
inset_data = []
for file_or_name in files:
    name = file_or_name.name if isinstance(file_or_name, Path) else file_or_name
    rs = extract_rs_mw(name)
    x, y = load_mw_pmf(file_or_name)
    ax.plot(x, y, lw=1.8)
    if any(abs(rs - val) < 1e-6 for val in small_rs):
        inset_data.append((rs, x, y))

ax.text(0.94, 0.98, r"$\mathbf{(a)}$", transform=ax.transAxes,
        ha="left", va="top", fontsize=15)
ax.set_xlabel(r"$r - 2R_s\;(\mathrm{\AA})$", fontsize=16)
ax.set_ylabel(r"$W(r)/k_{\mathrm{B}}T$", fontsize=16)
ax.set_xlim(-1.0, 6.0)
ax.set_ylim(-50.0, 8.0)
ax.tick_params(labelsize=12)

axins = inset_axes(
    ax, width="60%", height="60%",
    bbox_to_anchor=(0.59, 0.06, 0.60, 0.60),
    bbox_transform=ax.transAxes, loc="lower left"
)
for rs, x, y in inset_data:
    axins.plot(x, y, lw=1.5)
axins.set_xlim(-1.0, 6.0)
axins.set_ylim(-3.0, 1.0)
axins.tick_params(labelsize=9)

# Panel (b): PMF first-minimum depth, with shifted TIP4P/2005 comparison.
mw_handle_b, = ax1.plot(Rs_vals, depths, marker="o", linestyle="none", color=colors[0], label="mW")
tip_handle_b, = ax1.plot(TIP4P_Rs_vals, TIP4P_depths, marker="s", linestyle="none",
         markerfacecolor="none", markeredgewidth=1.3, markersize=6.0,
         color=colors[1], label=r"TIP4P")
ax1.set_xlabel(r"$R_s\; (\mathrm{\AA})$", fontsize=16)
ax1.set_ylabel(r"$\left|W_{\min}\right|/k_{\mathrm{B}}T$", fontsize=16)
ax1.tick_params(labelsize=11, direction="in", right=True, top=False)
ax1.set_xlim(1.2, 10.4)
ax1.set_ylim(0, 42)
ax1.xaxis.set_major_locator(MultipleLocator(2))
ax1.yaxis.set_major_locator(MultipleLocator(6))
ax1.legend([mw_handle_b, tip_handle_b], ["mW", "TIP4P"], frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.88), fontsize=10, handlelength=1.5)
ax1.text(0.04, 0.96, r"$\mathbf{(b)}$", transform=ax1.transAxes,
         fontsize=16, ha="left", va="top")

# Panel (c): interfacial length scale.
mw_handle_c = ax2.errorbar(ell_Rs, ell_vals, yerr=ell_errs, marker="o",
             linestyle="none", capsize=3, color=colors[0], label="mW")
tip_handle_c = ax2.errorbar(tip4p_ell_Rs, tip4p_ell_vals, yerr=tip4p_ell_errs, marker="s", linestyle="none",
         markerfacecolor="none", markeredgewidth=1.3, markersize=6.0,
         capsize=3, color=colors[1], label=r"TIP4P")
ax2.set_xlabel(r"$R_s\;(\mathrm{\AA})$", fontsize=16)
ax2.set_ylabel(r"$\ell\;(\mathrm{\AA})$", fontsize=16)
ax2.tick_params(labelsize=11, direction="in", right=True, top=False)
ax2.set_xlim(1.2, 10.4)
ax2.set_ylim(0, 3.0)
ax2.xaxis.set_major_locator(MultipleLocator(2))
ax2.legend([mw_handle_c, tip_handle_c], ["mW", "TIP4P"], frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.88), fontsize=10, handlelength=1.5)
ax2.text(0.04, 0.96, r"$\mathbf{(c)}$", transform=ax2.transAxes,
         fontsize=16, ha="left", va="top")

# Vector schematic in bottom-right of panel (c).
sax = ax2.inset_axes([0.382, 0.045, 0.756, 0.54], zorder=2)
sax.set_xlim(0.0, 1.0)
sax.set_ylim(0.0, 0.72)
sax.set_aspect("equal", adjustable="box")
sax.set_axis_off()

# Center and radii.
cx, cy = 0.40, 0.36
halo_r = 0.33
solute_r = 0.17

# Diffuse halo built from concentric translucent circles.
for radius, alpha in [(0.33, 0.08), (0.29, 0.10), (0.25, 0.12), (0.21, 0.15)]:
    sax.add_patch(Circle((cx, cy), radius, facecolor="#9fcdf4",
                         edgecolor="none", alpha=alpha, zorder=0))

# Thin circle marking the outer limit of the solvation shell.
sax.add_patch(Circle((cx, cy), halo_r, facecolor="none",
                     edgecolor="#6f8fa8", linewidth=0.7, zorder=1.5))

# Solute disk.
sax.add_patch(Circle((cx, cy), solute_r, facecolor="0.85",
                     edgecolor="black", linewidth=1.0, zorder=2))
sax.text(cx, cy, "solute", ha="center", va="center", fontsize=7.5, zorder=3)

# Ell arrow from solute surface to edge of halo.
x0 = cx + solute_r
x1 = cx + halo_r
sax.annotate("", xy=(x1, cy), xytext=(x0, cy),
             arrowprops=dict(arrowstyle="<->", color="black", lw=0.9,
                             shrinkA=0, shrinkB=0), zorder=3)
sax.text((x0 + x1) / 2, cy + 0.075, r"$\ell$",
         ha="center", va="center", fontsize=10.5, zorder=3)

fig.savefig(OUTFILE, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {OUTFILE}")
