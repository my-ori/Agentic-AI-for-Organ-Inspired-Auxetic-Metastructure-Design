#!/usr/bin/env python3
"""
Compute stress and Poisson's-ratio MAE between a target JSON file and
simulation results stored in reaction_right_grip.csv.

The simulation results are linearly interpolated onto the target strain grid,
because the strain intervals in the two files may be different.

Example:
    python compute_mae.py \
        --target "y_target.json" \
        --results reaction_right_grip.csv

Default assumptions:
    - Target strains are 0.01, 0.02, ..., based on the number of target values.
    - Simulation stress in MPa is Rx / 30.
    - The Poisson's-ratio column is nu_sec.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_target(
    path: Path,
    strain_start: float,
    strain_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load target stress and nu arrays and construct their strain grid."""
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read target JSON '{path}': {exc}") from exc

    if "stress" not in data or "nu" not in data:
        raise ValueError(
            "The target JSON must contain both 'stress' and 'nu' arrays."
        )

    stress = np.asarray(data["stress"], dtype=float)
    nu = np.asarray(data["nu"], dtype=float)

    if stress.ndim != 1 or nu.ndim != 1:
        raise ValueError("'stress' and 'nu' must each be one-dimensional arrays.")
    if len(stress) == 0:
        raise ValueError("The target arrays must not be empty.")
    if len(stress) != len(nu):
        raise ValueError(
            "'stress' and 'nu' must contain the same number of values."
        )
    if not np.all(np.isfinite(stress)) or not np.all(np.isfinite(nu)):
        raise ValueError("The target arrays contain NaN or infinite values.")
    if strain_step <= 0:
        raise ValueError("--strain-step must be greater than zero.")

    strains = strain_start + np.arange(len(stress), dtype=float) * strain_step
    return strains, stress, nu


def load_simulation(
    path: Path,
    area: float,
) -> pd.DataFrame:
    """Load, validate, and clean the simulation CSV."""
    if area <= 0:
        raise ValueError("--area must be greater than zero.")

    try:
        dataframe = pd.read_csv(path)
    except OSError as exc:
        raise ValueError(f"Could not read results CSV '{path}': {exc}") from exc

    # Remove accidental spaces such as ' nu_sec ' in CSV headers.
    dataframe.columns = dataframe.columns.str.strip()

    required_columns = {"eng_strain", "Rx", "nu_sec"}
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(
            "The results CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    simulation = dataframe[["eng_strain", "Rx", "nu_sec"]].copy()

    for column in simulation.columns:
        simulation[column] = pd.to_numeric(simulation[column], errors="coerce")

    simulation = simulation.replace([np.inf, -np.inf], np.nan)
    simulation = simulation.dropna(subset=["eng_strain", "Rx", "nu_sec"])

    if simulation.empty:
        raise ValueError(
            "No valid rows remain after removing missing or nonnumeric values."
        )

    simulation["stress_mpa"] = simulation["Rx"] / area

    # Average duplicate strain rows so that interpolation has one value per strain.
    simulation = (
        simulation.groupby("eng_strain", as_index=False)
        .agg(
            stress_mpa=("stress_mpa", "mean"),
            nu_sec=("nu_sec", "mean"),
        )
        .sort_values("eng_strain")
        .reset_index(drop=True)
    )

    if len(simulation) < 2:
        raise ValueError(
            "At least two distinct simulation strain values are required."
        )

    return simulation


def interpolate_results(
    target_strain: np.ndarray,
    simulation: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate simulation stress and nu onto target strain values.

    Extrapolation is intentionally rejected because it can make the MAE
    misleading.
    """
    sim_strain = simulation["eng_strain"].to_numpy(dtype=float)
    sim_stress = simulation["stress_mpa"].to_numpy(dtype=float)
    sim_nu = simulation["nu_sec"].to_numpy(dtype=float)

    target_min = float(np.min(target_strain))
    target_max = float(np.max(target_strain))
    sim_min = float(sim_strain[0])
    sim_max = float(sim_strain[-1])

    tolerance = 1e-12
    if target_min < sim_min - tolerance or target_max > sim_max + tolerance:
        raise ValueError(
            "The target strain range is outside the simulation strain range.\n"
            f"Target range:     [{target_min:.12g}, {target_max:.12g}]\n"
            f"Simulation range: [{sim_min:.12g}, {sim_max:.12g}]\n"
            "Change --strain-start/--strain-step or provide simulation data "
            "covering the full target range."
        )

    interpolated_stress = np.interp(target_strain, sim_strain, sim_stress)
    interpolated_nu = np.interp(target_strain, sim_strain, sim_nu)
    return interpolated_stress, interpolated_nu


def compute_mae(target: np.ndarray, prediction: np.ndarray) -> float:
    """Return the mean absolute error."""
    return float(np.mean(np.abs(target - prediction)))


def save_outputs(
    summary_path: Path,
    comparison_path: Path,
    target_strain: np.ndarray,
    target_stress: np.ndarray,
    predicted_stress: np.ndarray,
    target_nu: np.ndarray,
    predicted_nu: np.ndarray,
    stress_mae: float,
    nu_mae: float,
    area: float,
) -> None:
    """Write a text summary and a point-by-point comparison CSV."""
    comparison = pd.DataFrame(
        {
            "strain": target_strain,
            "stress_target_mpa": target_stress,
            "stress_result_mpa": predicted_stress,
            "stress_abs_error_mpa": np.abs(target_stress - predicted_stress),
            "nu_target": target_nu,
            "nu_result": predicted_nu,
            "nu_abs_error": np.abs(target_nu - predicted_nu),
        }
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)

    summary_text = (
        "MAE comparison results\n"
        "======================\n"
        f"Number of target points: {len(target_strain)}\n"
        f"Target strain range: {target_strain[0]:.12g} "
        f"to {target_strain[-1]:.12g}\n"
        f"Stress calculation: Rx / {area:.12g} MPa\n"
        "Interpolation method: linear interpolation onto target strains\n\n"
        f"Stress MAE (MPa): {stress_mae:.12g}\n"
        f"Poisson's ratio MAE: {nu_mae:.12g}\n"
    )

    summary_path.write_text(summary_text, encoding="utf-8")
    comparison.to_csv(comparison_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate reaction_right_grip.csv results onto the target "
            "strain grid and compute stress and Poisson's-ratio MAEs."
        )
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Target JSON file containing 'stress' and 'nu' arrays.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Simulation CSV containing eng_strain, Rx, and nu_sec.",
    )
    parser.add_argument(
        "--strain-start",
        type=float,
        default=0.01,
        help="Strain corresponding to the first target value; default: 0.01",
    )
    parser.add_argument(
        "--strain-step",
        type=float,
        default=0.01,
        help="Target strain interval; default: 0.01",
    )
    parser.add_argument(
        "--area",
        type=float,
        default=30.0,
        help="Area used in stress = Rx / area; default: 30",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mae_results.txt"),
        help="Text summary output; default: mae_results.txt",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("mae_comparison.csv"),
        help="Detailed comparison CSV; default: mae_comparison.csv",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        target_strain, target_stress, target_nu = load_target(
            args.target,
            args.strain_start,
            args.strain_step,
        )
        simulation = load_simulation(args.results, args.area)
        predicted_stress, predicted_nu = interpolate_results(
            target_strain,
            simulation,
        )

        stress_mae = compute_mae(target_stress, predicted_stress)
        nu_mae = compute_mae(target_nu, predicted_nu)

        save_outputs(
            summary_path=args.output,
            comparison_path=args.comparison_csv,
            target_strain=target_strain,
            target_stress=target_stress,
            predicted_stress=predicted_stress,
            target_nu=target_nu,
            predicted_nu=predicted_nu,
            stress_mae=stress_mae,
            nu_mae=nu_mae,
            area=args.area,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"File error: {exc}", file=sys.stderr)
        return 1

    print(f"Stress MAE (MPa):       {stress_mae:.12g}")
    print(f"Poisson's ratio MAE:    {nu_mae:.12g}")
    print(f"Summary saved to:       {args.output}")
    print(f"Comparison saved to:    {args.comparison_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
