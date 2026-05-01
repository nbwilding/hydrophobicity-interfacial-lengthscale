#!/usr/bin/env python3
"""
Estimate the isothermal compressibility from a fluctuating-volume LAMMPS dump in two ways:

1) NpT route from whole-box volume fluctuations:
      kappa_T = Var(V) / (k_B T <V>)
   with a simple propagated uncertainty estimate from the framewise statistics of V

2) Local / grand-canonical-like route from number fluctuations in spherical subvolumes:
      kappa_T(R) = Var(N_v) / (k_B T rho_b <N_v>)
where rho_b is the bulk density of the whole system, with its SEM estimated from frame-to-frame fluctuations of N/V.

Conventions used here:
- trajectory coordinates and box lengths are assumed to be in Angstrom
- temperature is in Kelvin
- k_B is taken in SI units (J/K) by default
- therefore kappa_T is reported in Angstrom^3 / J

The dump format is the same orthorhombic custom dump used previously:
    ITEM: TIMESTEP
    ITEM: NUMBER OF ATOMS
    ITEM: BOX BOUNDS ...
    ITEM: ATOMS ... x y z

Periodic boundaries are handled frame-by-frame using the instantaneous box lengths,
so fluctuating volume in an NpT trajectory is treated correctly for orthorhombic boxes.

Outputs:
- kappa_vs_R.dat
- kappa_vs_R.png
- var_vs_v.dat
- var_vs_v.png          (combined two-panel figure: (a) Var(N_v) vs v, (b) gradient vs v)
- P_Nv.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt


KB_SI = 1.380649e-23  # J/K


@dataclass
class Frame:
    timestep: int
    box: np.ndarray    # shape (3,), Angstrom
    pos: np.ndarray    # shape (N,3), shifted into [0,L), Angstrom


def minimum_image_displacement(dr: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Apply minimum-image convention for an orthorhombic periodic box."""
    return dr - box * np.round(dr / box)


def read_lammpstrj(filename: str | Path) -> List[Frame]:
    """
    Read a LAMMPS custom dump with columns including x, y, z.

    Supports orthorhombic boxes written as three lines of low/high bounds.
    Positions are shifted into [0, L) for convenience.
    """
    frames: List[Frame] = []

    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            if not line.startswith("ITEM: TIMESTEP"):
                raise ValueError("Unexpected dump format: expected 'ITEM: TIMESTEP'")
            timestep = int(f.readline().strip())

            line = f.readline().strip()
            if line != "ITEM: NUMBER OF ATOMS":
                raise ValueError("Unexpected dump format: expected 'ITEM: NUMBER OF ATOMS'")
            n_atoms = int(f.readline().strip())

            line = f.readline().strip()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Unexpected dump format: expected 'ITEM: BOX BOUNDS'")
            if any(tok in line.lower() for tok in ("xy", "xz", "yz")):
                raise ValueError("This script currently supports orthorhombic boxes only, not triclinic/tilted boxes.")

            bounds = []
            for _ in range(3):
                lo, hi = map(float, f.readline().split()[:2])
                bounds.append((lo, hi))
            bounds = np.array(bounds, dtype=float)
            box_lo = bounds[:, 0]
            box_hi = bounds[:, 1]
            box = box_hi - box_lo

            line = f.readline().strip()
            if not line.startswith("ITEM: ATOMS"):
                raise ValueError("Unexpected dump format: expected 'ITEM: ATOMS'")
            headers = line.split()[2:]
            col_index = {name: i for i, name in enumerate(headers)}
            for name in ("x", "y", "z"):
                if name not in col_index:
                    raise ValueError(f"Dump is missing required column '{name}'")

            pos = np.empty((n_atoms, 3), dtype=float)
            for i in range(n_atoms):
                parts = f.readline().split()
                pos[i, 0] = float(parts[col_index["x"]])
                pos[i, 1] = float(parts[col_index["y"]])
                pos[i, 2] = float(parts[col_index["z"]])

            pos -= box_lo
            pos %= box
            frames.append(Frame(timestep=timestep, box=box, pos=pos))

    if not frames:
        raise ValueError("No frames found in dump file.")
    return frames


def count_particles_in_spheres_multiR(
    pos: np.ndarray,
    box: np.ndarray,
    radii: np.ndarray,
    n_centers: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    For one frame, choose random sphere centres and count how many particle centres
    lie within each radius in 'radii'.

    Returns an array of shape (n_centers, len(radii)).
    """
    radii = np.asarray(radii, dtype=float)
    counts = np.empty((n_centers, len(radii)), dtype=int)

    for j in range(n_centers):
        r0 = rng.random(3) * box
        dr = pos - r0
        dr = minimum_image_displacement(dr, box)
        d2 = np.sum(dr * dr, axis=1)
        d2_sorted = np.sort(d2)
        counts[j, :] = np.searchsorted(d2_sorted, radii**2, side="right")
    return counts


def compute_npt_kappa(frames: List[Frame], T: float, kB: float) -> Dict[str, Any]:
    """Whole-system NpT compressibility from volume fluctuations, with simple framewise uncertainty estimates."""
    volumes = np.array([np.prod(fr.box) for fr in frames], dtype=float)
    n_frames = len(volumes)
    v_mean = float(np.mean(volumes))
    v_var = float(np.var(volumes, ddof=1)) if n_frames > 1 else 0.0
    if v_mean <= 0:
        raise ValueError("Mean volume is non-positive; check the trajectory.")
    kappa = v_var / (kB * T * v_mean)

    if n_frames > 1:
        v_mean_sem = float(np.std(volumes, ddof=1) / np.sqrt(n_frames))
    else:
        v_mean_sem = np.nan

    # Delta-method uncertainty for the sample variance and propagated uncertainty for kappa.
    # This assumes effectively independent frames and should therefore be interpreted cautiously
    # if the trajectory is strongly time-correlated.
    if n_frames > 3 and v_var > 0.0:
        mu4 = float(np.mean((volumes - v_mean) ** 4))
        var_of_sample_variance = (mu4 - ((n_frames - 3) / (n_frames - 1)) * (v_var ** 2)) / n_frames
        var_of_sample_variance = max(var_of_sample_variance, 0.0)
        v_var_sem = float(np.sqrt(var_of_sample_variance))
    elif n_frames > 1:
        v_var_sem = np.nan
    else:
        v_var_sem = np.nan

    if np.isfinite(v_mean_sem) and np.isfinite(v_var_sem) and v_mean > 0.0 and v_var >= 0.0:
        rel_var_term = 0.0 if v_var == 0.0 else (v_var_sem / v_var) ** 2
        rel_mean_term = (v_mean_sem / v_mean) ** 2
        kappa_sem = float(abs(kappa) * np.sqrt(rel_var_term + rel_mean_term))
    else:
        kappa_sem = np.nan

    return {
        "volumes": volumes,
        "V_mean": v_mean,
        "V_mean_sem": v_mean_sem,
        "V_var": v_var,
        "V_var_sem": v_var_sem,
        "kappa": kappa,
        "kappa_sem": kappa_sem,
        "n_frames": n_frames,
    }


def compute_bulk_density(frames: List[Frame]) -> Dict[str, Any]:
    """
    Compute the bulk density of the whole system from the trajectory.
    For fixed N and fluctuating V, we use <N/V> and report the SEM across frames.
    """
    rho_frames = np.array([fr.pos.shape[0] / np.prod(fr.box) for fr in frames], dtype=float)
    rho_mean = float(np.mean(rho_frames))
    if len(rho_frames) > 1:
        rho_sem = float(np.std(rho_frames, ddof=1) / np.sqrt(len(rho_frames)))
    else:
        rho_sem = np.nan
    return {
        "rho_frames": rho_frames,
        "rho_mean": rho_mean,
        "rho_sem": rho_sem,
    }



def compute_local_kappa_vs_R(
    frames: List[Frame],
    radii: np.ndarray,
    T: float,
    kB: float,
    rho_bulk: float,
    n_centers_per_frame: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Estimate kappa_T(R) from number fluctuations in random spherical subvolumes.

    For each frame, generate random centres, count N_v(R), and use:
        kappa_T(R) = Var(N_v) / (k_B T rho_b <N_v>)
    where rho_b is the bulk density of the whole system, with its SEM estimated from frame-to-frame fluctuations of N/V.

    Also returns framewise estimates and SEMs across frames.
    """
    rng = np.random.default_rng(seed)
    radii = np.asarray(radii, dtype=float)
    sphere_volumes = (4.0 / 3.0) * np.pi * radii**3

    frame_counts = []
    frame_kappa = []
    frame_var_n = []

    for i, fr in enumerate(frames):
        counts = count_particles_in_spheres_multiR(
            fr.pos, fr.box, radii, n_centers_per_frame, rng
        )
        frame_counts.append(counts)

        mean_n = counts.mean(axis=0)
        var_n = counts.var(axis=0, ddof=1) if counts.shape[0] > 1 else np.zeros_like(mean_n)
        frame_var_n.append(var_n)

        kappa_frame = np.full_like(radii, np.nan, dtype=float)
        mask = mean_n > 0
        kappa_frame[mask] = var_n[mask] / (kB * T * rho_bulk * mean_n[mask])
        frame_kappa.append(kappa_frame)

        print(
            f"frame {i+1}/{len(frames)} timestep={fr.timestep} "
            f"mean_Nv(Rmin,Rmax)=({mean_n[0]:.3f}, {mean_n[-1]:.3f})"
        )

    frame_counts = np.array(frame_counts, dtype=float)      # (n_frames, n_centers, n_radii)
    frame_kappa = np.array(frame_kappa, dtype=float)        # (n_frames, n_radii)
    frame_var_n = np.array(frame_var_n, dtype=float)        # (n_frames, n_radii)

    all_counts = frame_counts.reshape(-1, len(radii))
    mean_n = all_counts.mean(axis=0)
    var_n = all_counts.var(axis=0, ddof=1) if all_counts.shape[0] > 1 else np.zeros_like(mean_n)

    kappa = np.full_like(radii, np.nan, dtype=float)
    mask = mean_n > 0
    kappa[mask] = var_n[mask] / (kB * T * rho_bulk * mean_n[mask])

    kappa_sem = np.full_like(radii, np.nan, dtype=float)
    var_n_sem = np.full_like(radii, np.nan, dtype=float)
    if len(frames) > 1:
        valid_counts = np.sum(np.isfinite(frame_kappa), axis=0)
        std = np.nanstd(frame_kappa, axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            kappa_sem = std / np.sqrt(valid_counts)

        valid_var_counts = np.sum(np.isfinite(frame_var_n), axis=0)
        std_var = np.nanstd(frame_var_n, axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            var_n_sem = std_var / np.sqrt(valid_var_counts)

    rho_sub = np.full_like(radii, np.nan, dtype=float)
    mask_sv = sphere_volumes > 0
    rho_sub[mask_sv] = mean_n[mask_sv] / sphere_volumes[mask_sv]

    return {
        "radii": radii,
        "sphere_volumes": sphere_volumes,
        "mean_n": mean_n,
        "var_n": var_n,
        "var_n_sem": var_n_sem,
        "rho_sub": rho_sub,
        "rho_bulk": rho_bulk,
        "kappa": kappa,
        "kappa_sem": kappa_sem,
        "frame_kappa": frame_kappa,
        "frame_var_n": frame_var_n,
        "n_frames": len(frames),
        "n_centers_per_frame": n_centers_per_frame,
        "n_samples_total": all_counts.shape[0],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare isothermal compressibility from NpT volume fluctuations and local subvolume number fluctuations."
    )
    parser.add_argument("dumpfile", nargs="?", default="traj.lammpstrj", help="LAMMPS dump filename")
    parser.add_argument("--temperature", "-T", type=float, default=None, help="Temperature in K")
    parser.add_argument("--kB", type=float, default=KB_SI, help="Boltzmann constant in J/K (default: SI)")
    parser.add_argument("--rmin", type=float, default=0.5, help="Minimum sphere radius in Angstrom")
    parser.add_argument("--rmax", type=float, default=4.0, help="Maximum sphere radius in Angstrom")
    parser.add_argument("--dr", type=float, default=0.1, help="Radius spacing in Angstrom")
    parser.add_argument("--samples-per-frame", type=int, default=200, help="Random sphere centres per frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional limit on number of frames read")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--outfile", default="kappa_vs_R.dat", help="Output data filename")
    parser.add_argument("--plotfile", default="kappa_vs_R.png", help="Output plot filename")
    parser.add_argument("--varoutfile", default="var_vs_v.dat", help="Output data filename for variance versus subvolume volume")
    parser.add_argument("--varplotfile", default="var_vs_v.png", help="Output plot filename for variance versus subvolume volume")
    parser.add_argument("--gradylim", type=float, default=None, help="Optional upper y-limit for gradient panel in the combined variance figure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    T = args.temperature
    if T is None:
        T = float(input("Enter temperature T in K: ").strip())

    kB = args.kB
    radii = np.arange(args.rmin, args.rmax + 0.5 * args.dr, args.dr)

    frames = read_lammpstrj(args.dumpfile)
    if args.max_frames is not None:
        frames = frames[:args.max_frames]

    if len(frames) < 2:
        print("Warning: only one frame found; NpT fluctuation estimate and SEMs will be poor.")

    npt = compute_npt_kappa(frames, T, kB)
    bulk_density = compute_bulk_density(frames)
    rho_bulk = bulk_density["rho_mean"]
    local = compute_local_kappa_vs_R(
        frames=frames,
        radii=radii,
        T=T,
        kB=kB,
        rho_bulk=rho_bulk,
        n_centers_per_frame=args.samples_per_frame,
        seed=args.seed,
    )

    # ============================================================
    # Plot P(N_v) for selected radii
    # ============================================================

    R_targets = [1.8, 2.2, 3.0, 3.4, 3.8]

    # find nearest indices in radii array
    R_indices = [np.argmin(np.abs(radii - R)) for R in R_targets]

    # reconstruct all counts (already available via frame_kappa input)
    # we need to recompute counts cleanly (simplest approach)
    all_counts = []

    rng = np.random.default_rng(args.seed)

    for fr in frames:
        counts = count_particles_in_spheres_multiR(
            fr.pos, fr.box, radii, args.samples_per_frame, rng
        )
        all_counts.append(counts)

    all_counts = np.array(all_counts)  # (n_frames, n_centers, n_radii)
    all_counts = all_counts.reshape(-1, len(radii))  # flatten

    # plot distributions
    plt.figure(figsize=(7,5))

    for idx, R in zip(R_indices, R_targets):
        Nv = all_counts[:, idx]

        # histogram
        bins = np.arange(Nv.min(), Nv.max() + 2) - 0.5
        hist, edges = np.histogram(Nv, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        plt.plot(centers, hist, marker='o', label=f"R={radii[idx]:.1f} Å")

    plt.yscale('log')
    plt.xlabel(r"$N_v$")
    plt.ylabel(r"$P(N_v)$")
    plt.title("Subvolume occupancy distributions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("P_Nv.png", dpi=300)
    plt.show()


    kappa_ratio = local["kappa"] / npt["kappa"]
    kappa_ratio_sem = local["kappa_sem"] / npt["kappa"]

    header = (
        "# R_Angstrom  sphere_volume_A3  mean_Nv  var_Nv  var_Nv_sem  rho_sub_A^-3  rho_bulk_A^-3  "
        "kappa_local_A3_per_J  kappa_local_sem_A3_per_J  kappa_local_over_kappa_bulk  "
        "kappa_local_over_kappa_bulk_sem\n"
        f"# T_K = {T}\n"
        f"# kB_J_per_K = {kB}\n"
        f"# n_frames = {local['n_frames']}\n"
        f"# n_centers_per_frame = {local['n_centers_per_frame']}\n"
        f"# n_samples_total = {local['n_samples_total']}\n"
        f"# V_mean_A3 = {npt['V_mean']}\n"
        f"# V_mean_sem_A3 = {npt['V_mean_sem']}\n"
        f"# V_var_A6 = {npt['V_var']}\n"
        f"# V_var_sem_A6 = {npt['V_var_sem']}\n"
        f"# rho_bulk_A^-3 = {rho_bulk}\n"
        f"# rho_bulk_sem_A^-3 = {bulk_density['rho_sem']}\n"
        f"# kappa_T_from_NpT_volume_fluctuations_A3_per_J = {npt['kappa']}\n"
        f"# kappa_T_from_NpT_volume_fluctuations_sem_A3_per_J = {npt['kappa_sem']}"
    )
    out = np.column_stack(
        [
            local["radii"],
            local["sphere_volumes"],
            local["mean_n"],
            local["var_n"],
            local["var_n_sem"],
            local["rho_sub"],
            np.full_like(local["radii"], rho_bulk, dtype=float),
            local["kappa"],
            local["kappa_sem"],
            kappa_ratio,
            kappa_ratio_sem,
        ]
    )
    np.savetxt(args.outfile, out, header=header)

    plt.figure(figsize=(7, 5))
    if np.all(np.isnan(local["kappa_sem"])):
        plt.plot(local["radii"], local["kappa"], marker="o", linewidth=1.5, label="Subvolume fluctuation estimate")
    else:
        plt.errorbar(
            local["radii"],
            local["kappa"],
            yerr=local["kappa_sem"],
            fmt="o-",
            capsize=3,
            label="Subvolume fluctuation estimate",
        )
    plt.axhline(npt["kappa"], linestyle="--", linewidth=1.5, label="NpT volume-fluctuation estimate")
    plt.xlabel(r"$R$ ($\mathrm{\AA}$)")
    plt.ylabel(r"$\kappa_T$ ($\mathrm{\AA^3/J}$)")
    plt.title("Isothermal compressibility: subvolume fluctuations vs NpT")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(frameon=False)
    plt.savefig(args.plotfile, dpi=300)
    plt.show()

    ratio_plotfile = Path(args.plotfile).with_name(
        Path(args.plotfile).stem + "_normalised.png"
    )

    plt.figure(figsize=(7, 5))
    if np.all(np.isnan(kappa_ratio_sem)):
        plt.plot(local["radii"], kappa_ratio, marker="o", linewidth=1.5)
    else:
        plt.errorbar(
            local["radii"],
            kappa_ratio,
            yerr=kappa_ratio_sem,
            fmt="o-",
            capsize=3,
        )
    plt.axhline(1.0, linestyle="--", linewidth=1.5)
    plt.ylim(bottom=0)
    plt.xlabel(r"$R$ ($\mathrm{\AA}$)")
    plt.ylabel(r"$\kappa_T(R)/\kappa_T$")
    plt.title("Normalised subvolume compressibility")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ratio_plotfile, dpi=300)
    plt.show()

    bulk_slope = (rho_bulk**2) * kB * T * npt["kappa"]
    grad_var = np.gradient(local["var_n"], local["sphere_volumes"])
    if np.all(np.isnan(local["var_n_sem"])):
        grad_var_sem = np.full_like(grad_var, np.nan, dtype=float)
    else:
        grad_frame = []
        for frame_var in local["frame_var_n"]:
            grad_frame.append(np.gradient(frame_var, local["sphere_volumes"]))
        grad_frame = np.array(grad_frame, dtype=float)
        if grad_frame.shape[0] > 1:
            valid_grad_counts = np.sum(np.isfinite(grad_frame), axis=0)
            std_grad = np.nanstd(grad_frame, axis=0, ddof=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                grad_var_sem = std_grad / np.sqrt(valid_grad_counts)
        else:
            grad_var_sem = np.full_like(grad_var, np.nan, dtype=float)

    var_header = (
        "# sphere_volume_A3  var_Nv  var_Nv_sem  mean_Nv  dVar_dv  dVar_dv_sem  bulk_slope_A^-3\n"
        f"# T_K = {T}\n"
        f"# kB_J_per_K = {kB}\n"
        f"# n_frames = {local['n_frames']}\n"
        f"# n_centers_per_frame = {local['n_centers_per_frame']}\n"
        f"# n_samples_total = {local['n_samples_total']}\n"
        f"# rho_bulk_A^-3 = {rho_bulk}\n"
        f"# rho_bulk_sem_A^-3 = {bulk_density['rho_sem']}\n"
        f"# kappa_T_from_NpT_volume_fluctuations_A3_per_J = {npt['kappa']}\n"
        f"# kappa_T_from_NpT_volume_fluctuations_sem_A3_per_J = {npt['kappa_sem']}\n"
        f"# bulk_slope_A^-3 = {bulk_slope}\n"
    )
    var_out = np.column_stack(
        [
            local["sphere_volumes"],
            local["var_n"],
            local["var_n_sem"],
            local["mean_n"],
            grad_var,
            grad_var_sem,
            np.full_like(local["sphere_volumes"], bulk_slope, dtype=float),
        ]
    )
    np.savetxt(args.varoutfile, var_out, header=var_header)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    if np.all(np.isnan(local["var_n_sem"])):
        ax1.plot(local["sphere_volumes"], local["var_n"], marker="o", linewidth=1.5)
    else:
        ax1.errorbar(
            local["sphere_volumes"],
            local["var_n"],
            yerr=local["var_n_sem"],
            fmt="o-",
            capsize=3,
            linewidth=1.5,
        )
    ax1.set_ylabel(r"$\mathrm{Var}(N_v)$", fontsize=16)
    ax1.text(0.01, 0.88, r"(a)", transform=ax1.transAxes, fontsize=16)
    ax1.grid(True)

    if np.all(np.isnan(grad_var_sem)):
        ax2.plot(local["sphere_volumes"], grad_var, marker="o", linewidth=1.5)
    else:
        ax2.errorbar(
            local["sphere_volumes"],
            grad_var,
            yerr=grad_var_sem,
            fmt="o-",
            capsize=3,
            linewidth=1.5,
        )
    ax2.axhline(bulk_slope, linestyle="--", linewidth=1.5)
    ax2.set_xlabel(r"$v$ ($\mathrm{\AA^3}$)", fontsize=16)
    ax2.set_ylim(0, 0.0125)
    ax2.set_ylabel(r"$\frac{d\,\mathrm{Var}(N_v)}{dv}$", fontsize=16)
    if args.gradylim is not None:
        ax2.set_ylim(top=args.gradylim)
    ax2.text(0.01, 0.88, r"(b)", transform=ax2.transAxes, fontsize=16)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(args.varplotfile, dpi=300)
    plt.show()
    print()
    print("=== NpT compressibility from volume fluctuations ===")
    print(f"Read frames                 : {len(frames)}")
    print(f"Trajectory file             : {args.dumpfile}")
    print(f"Temperature T               : {T:.6f} K")
    print(f"Boltzmann constant k_B      : {kB:.9e} J/K")
    print(f"Average volume <V>          : {npt['V_mean']:.6f} ± {npt['V_mean_sem']:.6f} A^3")
    print(f"Volume variance Var(V)      : {npt['V_var']:.6f} ± {npt['V_var_sem']:.6f} A^6")
    print(f"kappa_T (NpT)               : {npt['kappa']:.6e} ± {npt['kappa_sem']:.6e} A^3/J")
    print()
    print("=== Bulk density used in GCE-like estimate ===")
    print(f"rho_b                       : {rho_bulk:.6e} ± {bulk_density['rho_sem']:.6e} A^-3")
    print()
    print(f"Wrote data to               : {args.outfile}")
    print(f"Saved plot to               : {args.plotfile}")
    print(f"Saved normalised plot to    : {ratio_plotfile}")
    print(f"Wrote Var(N_v), gradient, and asymptote data to : {args.varoutfile}")
    print(f"Saved combined Var(N_v)/gradient plot to        : {args.varplotfile}")


if __name__ == "__main__":
    main()
