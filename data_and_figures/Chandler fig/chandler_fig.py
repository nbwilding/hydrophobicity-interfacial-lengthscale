import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# LaTeX-like styling
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 13,
    "axes.linewidth": 1.1,
})

# Digitised data from the original schematic
R = np.array([0.160, 0.252, 0.364, 0.496, 0.646, 0.793,
              0.951, 1.119, 1.344, 1.602, 1.870])

DG_area = np.array([13.4, 24.6, 33.9, 44.6, 51.4, 55.4,
                    57.8, 59.5, 61.8, 64.0, 64.6])

# Manual guide line, chosen to match the original Chandler-style schematic
gamma = 70.0

idx_anchor = 4
slope = DG_area[idx_anchor] / R[idx_anchor]

R_solid = np.linspace(0.0, R[idx_anchor], 200)
R_dash = np.linspace(R[idx_anchor], 1.18, 100)


fig, ax = plt.subplots(figsize=(5.8, 4.0))

# Data
ax.scatter(R, DG_area, s=105, color="black", zorder=3)

# Linear guide and dashed continuation
ax.plot(R_solid, slope * R_solid, color="black", lw=1.8)
ax.plot(R_dash, slope * R_dash, color="black", lw=1.8, ls="--")

# Large-R interfacial limit
ax.axhline(gamma, color="black", lw=1.2, ls="--")

# Gamma label
ax.annotate(
    r"$\gamma_{lv}$",
    xy=(1.9, gamma),          # arrow points to line
    xytext=(2.05, gamma+5),   # move label up/right
    ha="left",
    va="bottom",
    arrowprops=dict(arrowstyle="->", lw=1.0, color="black"),
)



# Labels
ax.set_xlabel(r"$R\ \mathrm{(nm)}$", fontsize=15)
ax.set_ylabel(r"$\Delta G/4\pi R^2\ \mathrm{(mJ\,m^{-2})}$", fontsize=15)

# Limits and ticks
ax.set_xlim(0.0, 2.2)
ax.set_ylim(0.0, 90.0)
ax.set_xticks(np.arange(0.0, 2.21, 0.4))
ax.set_yticks(np.arange(0.0, 91.0, 20.0))

# PNAS/JCP-style axes
ax.tick_params(direction="in", top=True, right=True, length=5, width=1.0)
ax.tick_params(which="minor", direction="in", top=True, right=True)
ax.minorticks_off()

plt.tight_layout()

# Save vector + raster versions
plt.savefig("chandler_style_schematic.pdf", bbox_inches="tight")
plt.savefig("chandler_style_schematic.png", dpi=600, bbox_inches="tight")

plt.show()
