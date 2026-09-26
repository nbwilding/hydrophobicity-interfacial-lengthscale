# SI Figure 3: LJ affinity dependence and single-solute response
from pathlib import Path
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.style.use('tableau-colorblind10')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PMF_DIR = ROOT / 'LJ PMF_data'
PROFILE_DIR = ROOT / 'LJ_Rho_Kappa_data'
OUT_PDF = HERE / 'SI_figure_3.pdf'

T_TARGET = 0.775
P_TARGET = 0.0
RS_VALUES = (1.0, 4.0)

PMF_EPS = {
    1.0: (0.01, 0.10, 0.20, 0.40, 0.60),
    4.0: (0.01, 0.20, 0.40, 0.60, 0.80, 0.90),
}

PROFILE_EPS = {
    1.0: (0.01, 0.60),
    4.0: (0.01, 0.40, 0.80, 1.20),
}

PMF_BASELINE = {
    (1.0, 0.01): (3.5, 5.0),
    (1.0, 0.10): (3.5, 5.0),
    (1.0, 0.20): (3.5, 5.0),
    (1.0, 0.40): (3.5, 5.0),
    (1.0, 0.60): (3.5, 5.0),
    # For the larger solute, use the broad pre-noise plateau rather than
    # the extreme large-r tail, which develops an artificial upturn.
    (4.0, 0.01): (5.0, 7.0),
    (4.0, 0.20): (5.0, 7.0),
    (4.0, 0.40): (5.0, 7.0),
    (4.0, 0.60): (5.0, 7.0),
    (4.0, 0.80): (4.5, 6.5),
    (4.0, 0.90): (4.8, 6.8),
}

pmf_pattern = re.compile(
    r"pmf_T(?P<T>[0-9.]+)_P(?P<P>[0-9.]+)_Rs(?P<Rs>[0-9.]+)_D0(?P<eps>[0-9.]+)\.dat$"
)
profile_pattern = re.compile(
    r"(?P<kind>density|kappa)-T(?P<T>[0-9.]+)-P(?P<P>[0-9.]+)-dr(?P<dr>[0-9.]+)-D0(?P<eps>[0-9.]+)-Rs(?P<Rs>[0-9.]+)\.dat$"
)


def find_match(pattern, rs, eps, kind=None):
    matches = []
    search_dir = PROFILE_DIR if kind is not None else PMF_DIR
    for path in search_dir.glob('*.dat'):
        m = pattern.match(path.name)
        if m is None:
            continue
        g = m.groupdict()
        if kind is not None and g.get('kind') != kind:
            continue
        if (np.isclose(float(g['T']), T_TARGET)
                and np.isclose(float(g['P']), P_TARGET)
                and np.isclose(float(g['Rs']), rs)
                and np.isclose(float(g['eps']), eps)):
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(
            f'Expected exactly one file for kind={kind}, Rs={rs}, eps={eps}; found {len(matches)}: {matches}'
        )
    return matches[0]


def find_match_optional(pattern, rs, eps, kind=None):
    try:
        return find_match(pattern, rs, eps, kind)
    except FileNotFoundError:
        return None


mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.linewidth': 1.0,
    'axes.labelsize': 15,
    'font.size': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
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
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colour_maps = {}
for rs in RS_VALUES:
    union = sorted(set(PMF_EPS[rs]) | set(PROFILE_EPS[rs]))
    colour_maps[rs] = {eps: colors[i % len(colors)] for i, eps in enumerate(union)}

fig, ax = plt.subplots(3, 2, figsize=(7.1, 8.5), sharex=False, sharey=False,
                       gridspec_kw={'hspace': 0.36, 'wspace': 0.28})

panel_labels = [['(a)', '(b)'], ['(c)', '(d)'], ['(e)', '(f)']]

# PMFs
for j, rs in enumerate(RS_VALUES):
    a = ax[0, j]
    plotted_eps = []
    for eps in PMF_EPS[rs]:
        path = find_match_optional(pmf_pattern, rs, eps)
        if path is None:
            print(f'Warning: missing PMF file for Rs={rs}, eps={eps}; skipping this curve.')
            continue
        data = np.loadtxt(path, comments='#')
        x, w = data[:, 0], data[:, 1]
        finite = np.isfinite(x) & np.isfinite(w)
        lo, hi = PMF_BASELINE.get((rs, eps), (max(x) - 1.0, max(x)))
        base = finite & (x >= lo) & (x <= hi)
        if not np.any(base):
            base = finite & (x >= np.nanmax(x) - 1.0)
        w = w - np.mean(w[base])
        visible = finite & (x >= -0.5) & (x <= 5.0)
        a.plot(x[visible], w[visible], color=colour_maps[rs][eps],
               label=rf'$\epsilon_{{sf}}={eps:g}$')
        plotted_eps.append(eps)
    a.axhline(0.0, linestyle='--', linewidth=1.2, color='k', alpha=0.7)
    a.set_xlim(-0.5, 5.0)
    a.set_xlabel(r'$(r-2R_s)/\sigma$', fontsize=15)
    a.tick_params(axis='both', which='both', direction='in', top=False, right=True, labelsize=13)
    a.minorticks_on()
    a.legend(loc='lower right', handlelength=2.1)
    a.set_title(rf'$R_s={rs:g}\sigma$', fontsize=15, pad=6)

ax[0, 0].set_ylabel(r'$\beta W(r)$', fontsize=15)
ax[0, 0].set_ylim(-5.8, 0.8)
ax[0, 1].set_ylim(-55.5, 4.0)

# Density and local compressibility
for j, rs in enumerate(RS_VALUES):
    r_contact = rs + 0.5
    for eps in PROFILE_EPS[rs]:
        rho_path = find_match(profile_pattern, rs, eps, 'density')
        kap_path = find_match(profile_pattern, rs, eps, 'kappa')
        rho = np.loadtxt(rho_path, comments='#')
        kap = np.loadtxt(kap_path, comments='#')

        xr = rho[:, 0] - r_contact
        xk = kap[:, 0] - r_contact
        cr = colour_maps[rs][eps]

        mr = np.isfinite(xr) & np.isfinite(rho[:, 1]) & (xr >= -0.5) & (xr <= 5.0)
        mk = np.isfinite(xk) & np.isfinite(kap[:, 1]) & (xk >= -0.2) & (xk <= 5.0)

        ax[1, j].plot(xr[mr], rho[mr, 1], lw=2.0, color=cr,
                      label=rf'$\epsilon_{{sf}}={eps:g}$')
        ax[2, j].plot(xk[mk], kap[mk, 1], lw=2.0, color=cr,
                      label=rf'$\epsilon_{{sf}}={eps:g}$')

    for i in (1, 2):
        a = ax[i, j]
        a.axhline(1.0, linestyle='--', linewidth=1.2, color='k', alpha=0.7)
        a.set_xlim(-0.5, 5.0)
        a.tick_params(axis='both', which='both', direction='in', top=False, right=True, labelsize=13)
        a.minorticks_on()

    ax[1, j].legend(loc='upper right', handlelength=2.1)
    ax[2, j].legend(loc='upper right', handlelength=2.1)
    ax[2, j].set_xlabel(r'$[r-(R_s+\sigma/2)]/\sigma$', fontsize=15)

ax[1, 0].set_ylabel(r'$\varrho(r)/\rho_b$', fontsize=15)
ax[2, 0].set_ylabel(r'$\kappa(r)/\kappa_T$', fontsize=15)

ax[1, 0].set_ylim(-0.05, 2.70)
ax[1, 1].set_ylim(-0.05, 1.90)
ax[2, 0].set_ylim(-0.6, 9.8)
ax[2, 1].set_ylim(-0.8, 30.0)

# x tick spacing like other figures
for row in range(3):
    for j in range(2):
        ax[row, j].xaxis.set_major_locator(MultipleLocator(1.0))

# panel labels outside axes to avoid overlap with data
for i in range(3):
    for j in range(2):
        ax[i, j].text(-0.10, 1.04, panel_labels[i][j], transform=ax[i, j].transAxes,
                      ha='left', va='bottom', fontsize=14, clip_on=False)

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches='tight')
plt.close(fig)
print(f'Saved {OUT_PDF}')
