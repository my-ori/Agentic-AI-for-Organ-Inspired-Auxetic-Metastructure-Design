#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


plt.rcParams.update({"font.size": 21})


def is_nondominated_min_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask for the nondominated points when both objectives
    are minimized.

    A point is Pareto-optimal if no other point is equal or better in both
    objectives and strictly better in at least one.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be 1D arrays of equal length.")

    n = len(x)
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        dominated = (
            (x <= x[i])
            & (y <= y[i])
            & ((x < x[i]) | (y < y[i]))
        )
        if np.any(dominated):
            keep[i] = False

    return keep


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot surrogate-predicted physical-space stress MAE versus "
            "Poisson-ratio MAE and highlight the Pareto frontier."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to physical_mae_all_candidates.csv",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default="pareto_scatter_physical_mae.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--out_pdf",
        type=str,
        default="",
        help="Optional output PDF path",
    )
    parser.add_argument(
        "--pareto_mode",
        type=str,
        default="feasible",
        choices=["feasible", "all"],
        help=(
            "Compute Pareto front using only candidates marked feasible "
            "('feasible') or using all candidates ('all')."
        ),
    )
    parser.add_argument(
        "--log_axes",
        action="store_true",
        help="Use log scale on both x and y axes",
    )
    parser.add_argument(
        "--annotate_pareto",
        action="store_true",
        help="Annotate Pareto points with epsilon values",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    required = {
        "stress_mae_phys",
        "nu_mae_phys",
        "source_epsilon",
    }

    if args.pareto_mode == "feasible":
        required.add("source_selection_is_feasible")

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Ensure objective columns are finite.
    finite_mask = (
        np.isfinite(df["stress_mae_phys"].to_numpy(dtype=float))
        & np.isfinite(df["nu_mae_phys"].to_numpy(dtype=float))
    )

    if not finite_mask.all():
        n_bad = int((~finite_mask).sum())
        print(f"[Warning] Dropping {n_bad} candidates with non-finite MAE values.")
        df = df.loc[finite_mask].copy()

    # Choose which candidates participate in the Pareto comparison.
    if args.pareto_mode == "feasible":
        feasible_col = df["source_selection_is_feasible"]

        # Robust conversion in case CSV stores booleans as text.
        if feasible_col.dtype == object:
            feasible_mask = (
                feasible_col.astype(str).str.strip().str.lower()
                .map({"true": True, "false": False})
            )
            if feasible_mask.isna().any():
                raise ValueError(
                    "Could not parse some values in "
                    "'source_selection_is_feasible' as booleans."
                )
            pareto_input_mask = feasible_mask.to_numpy(dtype=bool)
        else:
            pareto_input_mask = feasible_col.astype(bool).to_numpy()
    else:
        pareto_input_mask = np.ones(len(df), dtype=bool)

    pareto_flag = np.zeros(len(df), dtype=bool)

    if pareto_input_mask.any():
        x = df.loc[pareto_input_mask, "stress_mae_phys"].to_numpy(dtype=float)
        y = df.loc[pareto_input_mask, "nu_mae_phys"].to_numpy(dtype=float)

        pareto_local = is_nondominated_min_2d(x, y)
        pareto_flag[np.flatnonzero(pareto_input_mask)] = pareto_local

    df["pareto_physical_mae"] = pareto_flag

    # Sort Pareto points from left to right in objective space.
    pareto = (
        df[df["pareto_physical_mae"]]
        .copy()
        .sort_values(["stress_mae_phys", "nu_mae_phys"])
    )

    # Color by epsilon with logarithmic normalization because epsilon spans
    # orders of magnitude.
    eps = df["source_epsilon"].to_numpy(dtype=float)

    if not np.isfinite(eps).all():
        raise ValueError("source_epsilon contains NaN or Inf values.")
    if np.any(eps <= 0):
        raise ValueError(
            "source_epsilon must be strictly positive for logarithmic color normalization."
        )

    eps_min = eps.min()
    eps_max = eps.max()

    # If every epsilon is identical, LogNorm is not meaningful.
    if np.isclose(eps_min, eps_max):
        norm = None
    else:
        norm = LogNorm(vmin=eps_min, vmax=eps_max)

    fig, ax = plt.subplots(figsize=(8, 6))

    # All candidates.
    sc = ax.scatter(
        df["stress_mae_phys"],
        df["nu_mae_phys"],
        c=df["source_epsilon"],
        cmap="viridis",
        norm=norm,
        s=55,
        alpha=0.85,
        edgecolors="none",
        label="Candidates",
        zorder=2,
    )

    # Pareto frontier points.
    if len(pareto) > 0:
        ax.scatter(
            pareto["stress_mae_phys"],
            pareto["nu_mae_phys"],
            s=130,
            facecolors="none",
            edgecolors="black",
            linewidths=1.8,
            label="Pareto points",
            zorder=4,
        )

        ax.plot(
            pareto["stress_mae_phys"],
            pareto["nu_mae_phys"],
            linewidth=1.8,
            marker="o",
            markersize=4,
            label="Pareto frontier",
            zorder=3,
        )

        if args.annotate_pareto:
            for _, row in pareto.iterrows():
                ax.annotate(
                    f"ε={row['source_epsilon']:.4g}",
                    (row["stress_mae_phys"], row["nu_mae_phys"]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=9,
                )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$\epsilon$")

    ax.set_xlabel("Stress MAE (MPa)")
    ax.set_ylabel("Poisson's ratio MAE")

    if args.log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.legend(frameon=True, fontsize=19, loc="upper right")
    fig.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    if args.out_pdf:
        out_pdf = Path(args.out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")

    print(f"Candidates plotted: {len(df)}")
    print(f"Pareto mode: {args.pareto_mode}")
    print(f"Pareto points: {len(pareto)}")
    print(f"Saved PNG to: {out_png}")

    if args.out_pdf:
        print(f"Saved PDF to: {args.out_pdf}")


if __name__ == "__main__":
    main()
