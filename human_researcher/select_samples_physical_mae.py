#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select five representative samples at approximately equal cumulative
arc-length intervals along the physical-MAE Pareto front.

Selection procedure
-------------------
1. Read stress_mae_phys and nu_mae_phys for all candidates.
2. By default, keep only source_selection_is_feasible == True candidates.
3. Recompute the Pareto front by minimizing both physical-space MAEs.
4. Sort Pareto points from low stress MAE to high stress MAE.
5. Min-max normalize both objectives over the Pareto front.
6. Compute cumulative Euclidean arc length along the normalized front.
7. Normalize cumulative distance to [0, 1].
8. Select five unique Pareto samples nearest to:
       0.00, 0.25, 0.50, 0.75, 1.00

The text output keeps the same four-column tab-delimited structure as the
old selector:
    endpoint    cumulative_distance    absolute_difference    source_pt_path

The input may be either:
- a .pt payload containing physical MAEs and source paths, or
- a .csv containing the same required columns.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@dataclass(frozen=True)
class Candidate:
    """One valid candidate."""

    stress_mae_phys: float
    nu_mae_phys: float
    source_pt_path: str
    row_number: int
    is_feasible: bool


@dataclass(frozen=True)
class ParetoPoint:
    """One Pareto candidate with cumulative normalized arc-length position."""

    candidate: Candidate
    cumulative_distance: float


@dataclass(frozen=True)
class Selection:
    """A candidate selected for one target cumulative-distance endpoint."""

    endpoint: float
    point: ParetoPoint

    @property
    def absolute_difference(self) -> float:
        return abs(self.point.cumulative_distance - self.endpoint)


def parse_bool(value: Any, *, field_name: str, row_number: int) -> bool:
    """Parse a boolean robustly from bool/int/string representations."""
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, np.integer)):
        return bool(value)

    if isinstance(value, (float, np.floating)):
        if value in (0.0, 1.0):
            return bool(int(value))

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False

    raise ValueError(
        f"Invalid {field_name} value '{value}' at row {row_number}."
    )


def validate_candidate(
    stress: Any,
    nu: Any,
    source_path: Any,
    feasible: Any,
    row_number: int,
) -> Candidate:
    """Convert raw fields to a validated Candidate."""
    try:
        stress_value = float(stress)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid stress_mae_phys value '{stress}' at row {row_number}."
        ) from exc

    try:
        nu_value = float(nu)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid nu_mae_phys value '{nu}' at row {row_number}."
        ) from exc

    if not math.isfinite(stress_value):
        raise ValueError(
            f"Non-finite stress_mae_phys value at row {row_number}."
        )
    if not math.isfinite(nu_value):
        raise ValueError(
            f"Non-finite nu_mae_phys value at row {row_number}."
        )

    source_text = str(source_path).strip()
    if not source_text:
        raise ValueError(f"Empty source_pt_path at row {row_number}.")

    feasible_value = parse_bool(
        feasible,
        field_name="source_selection_is_feasible",
        row_number=row_number,
    )

    return Candidate(
        stress_mae_phys=stress_value,
        nu_mae_phys=nu_value,
        source_pt_path=source_text,
        row_number=row_number,
        is_feasible=feasible_value,
    )


def read_candidates_csv(csv_path: Path) -> list[Candidate]:
    """Read candidates from a CSV file."""
    try:
        input_file = csv_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ValueError(f"Cannot open input CSV '{csv_path}': {exc}") from exc

    with input_file:
        reader = csv.DictReader(input_file)

        required_columns = {
            "stress_mae_phys",
            "nu_mae_phys",
            "source_pt_path",
            "source_selection_is_feasible",
        }
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                f"Missing required CSV column(s): {missing_text}. "
                "If your physical-MAE CSV does not contain source_pt_path, "
                "use the augmented .pt file produced by calculate_physical_mae.py."
            )

        candidates: list[Candidate] = []

        for row_number, row in enumerate(reader, start=2):
            candidates.append(
                validate_candidate(
                    stress=row.get("stress_mae_phys"),
                    nu=row.get("nu_mae_phys"),
                    source_path=row.get("source_pt_path"),
                    feasible=row.get("source_selection_is_feasible"),
                    row_number=row_number,
                )
            )

    if not candidates:
        raise ValueError("The input CSV contains no data rows.")

    return candidates


def tensor_or_list_item(value: Any, i: int) -> Any:
    """Extract one element from a tensor/list-like payload field."""
    if torch is not None and isinstance(value, torch.Tensor):
        item = value[i]
        if item.numel() == 1:
            return item.item()
        return item

    return value[i]


def read_candidates_pt(pt_path: Path) -> list[Candidate]:
    """Read candidates from an augmented PT payload."""
    if torch is None:
        raise ValueError(
            "PyTorch is required to read .pt input files but is not installed."
        )

    try:
        payload = torch.load(
            pt_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise ValueError(f"Cannot load PT file '{pt_path}': {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Expected the PT file to contain a dictionary payload.")

    required_keys = {
        "stress_mae_phys",
        "nu_mae_phys",
        "source_pt_path",
        "source_selection_is_feasible",
    }
    missing = required_keys.difference(payload)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required PT key(s): {missing_text}")

    n = len(payload["stress_mae_phys"])

    for key in required_keys:
        if len(payload[key]) != n:
            raise ValueError(
                f"PT field '{key}' has length {len(payload[key])}, expected {n}."
            )

    candidates: list[Candidate] = []

    # row_number uses 0-based candidate index for PT input.
    for i in range(n):
        candidates.append(
            validate_candidate(
                stress=tensor_or_list_item(payload["stress_mae_phys"], i),
                nu=tensor_or_list_item(payload["nu_mae_phys"], i),
                source_path=tensor_or_list_item(payload["source_pt_path"], i),
                feasible=tensor_or_list_item(
                    payload["source_selection_is_feasible"], i
                ),
                row_number=i,
            )
        )

    if not candidates:
        raise ValueError("The input PT contains no candidates.")

    return candidates


def read_candidates(input_path: Path) -> list[Candidate]:
    """Read candidates from .pt or .csv input."""
    suffix = input_path.suffix.lower()

    if suffix == ".pt":
        return read_candidates_pt(input_path)
    if suffix == ".csv":
        return read_candidates_csv(input_path)

    raise ValueError(
        f"Unsupported input format '{input_path.suffix}'. Use .pt or .csv."
    )


def is_nondominated_min_2d(
    stress: np.ndarray,
    nu: np.ndarray,
) -> np.ndarray:
    """
    Return a boolean mask of nondominated points when both objectives
    are minimized.
    """
    stress = np.asarray(stress, dtype=float)
    nu = np.asarray(nu, dtype=float)

    if stress.ndim != 1 or nu.ndim != 1 or stress.shape != nu.shape:
        raise ValueError("Pareto objectives must be equal-length 1D arrays.")

    n = len(stress)
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        dominates_i = (
            (stress <= stress[i])
            & (nu <= nu[i])
            & ((stress < stress[i]) | (nu < nu[i]))
        )
        if np.any(dominates_i):
            keep[i] = False

    return keep


def compute_pareto_front(
    candidates: list[Candidate],
    pareto_mode: str,
) -> list[Candidate]:
    """Recompute the physical-MAE Pareto front."""
    if pareto_mode == "feasible":
        pool = [candidate for candidate in candidates if candidate.is_feasible]
    elif pareto_mode == "all":
        pool = list(candidates)
    else:
        raise ValueError(f"Unknown pareto_mode: {pareto_mode}")

    if not pool:
        raise ValueError(
            f"No candidates are available for pareto_mode='{pareto_mode}'."
        )

    stress = np.asarray(
        [candidate.stress_mae_phys for candidate in pool],
        dtype=float,
    )
    nu = np.asarray(
        [candidate.nu_mae_phys for candidate in pool],
        dtype=float,
    )

    mask = is_nondominated_min_2d(stress, nu)
    pareto = [candidate for candidate, keep in zip(pool, mask) if keep]

    # Sort from the stress-optimal extreme toward the nu-optimal extreme.
    pareto.sort(
        key=lambda candidate: (
            candidate.stress_mae_phys,
            candidate.nu_mae_phys,
            candidate.row_number,
        )
    )

    return pareto


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1D array; return zeros if its range is zero."""
    values = np.asarray(values, dtype=float)
    minimum = values.min()
    maximum = values.max()

    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=0.0):
        return np.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


def cumulative_arc_positions(
    pareto: list[Candidate],
) -> list[ParetoPoint]:
    """
    Calculate cumulative arc-length positions along the Pareto front after
    min-max normalization of both objectives.

    The final cumulative distances are normalized to [0, 1].
    """
    if len(pareto) < 2:
        raise ValueError(
            "At least two Pareto points are required to calculate arc length."
        )

    stress = np.asarray(
        [candidate.stress_mae_phys for candidate in pareto],
        dtype=float,
    )
    nu = np.asarray(
        [candidate.nu_mae_phys for candidate in pareto],
        dtype=float,
    )

    stress_norm = minmax_normalize(stress)
    nu_norm = minmax_normalize(nu)

    step_distance = np.sqrt(
        np.diff(stress_norm) ** 2
        + np.diff(nu_norm) ** 2
    )

    cumulative = np.concatenate(
        [np.array([0.0]), np.cumsum(step_distance)]
    )

    total_length = float(cumulative[-1])
    if total_length <= 0.0:
        raise ValueError(
            "The Pareto front has zero total arc length; "
            "representative spacing cannot be defined."
        )

    cumulative_fraction = cumulative / total_length

    return [
        ParetoPoint(
            candidate=candidate,
            cumulative_distance=float(distance),
        )
        for candidate, distance in zip(pareto, cumulative_fraction)
    ]


def equal_distance_endpoints(segments: int) -> list[float]:
    """
    Return segments + 1 equally spaced cumulative-distance endpoints in [0, 1].

    With the default segments=4, this gives five samples:
        0.00, 0.25, 0.50, 0.75, 1.00
    """
    if segments <= 0:
        raise ValueError("The number of segments must be greater than zero.")

    return np.linspace(0.0, 1.0, segments + 1).tolist()


def select_nearest_unique_points(
    endpoints: list[float],
    points: list[ParetoPoint],
) -> list[Selection]:
    """
    Select one unique Pareto point for each cumulative-distance endpoint.

    Selections are constrained to remain ordered along the Pareto front.
    The first and last Pareto points are selected for endpoints 0 and 1.
    """
    m = len(endpoints)
    n = len(points)

    if n < m:
        raise ValueError(
            f"Need at least {m} Pareto points to select {m} unique samples, "
            f"but only {n} Pareto points are available."
        )

    selected_indices: list[int] = []

    for j, endpoint in enumerate(endpoints):
        if j == 0:
            selected_index = 0

        elif j == m - 1:
            selected_index = n - 1

        else:
            # Keep selections strictly ordered and leave enough points for
            # all remaining endpoints.
            lower = selected_indices[-1] + 1
            remaining_after_this = (m - 1) - j
            upper = (n - 1) - remaining_after_this

            if lower > upper:
                raise ValueError(
                    "Could not select unique ordered Pareto samples."
                )

            candidate_indices = range(lower, upper + 1)
            selected_index = min(
                candidate_indices,
                key=lambda i: (
                    abs(points[i].cumulative_distance - endpoint),
                    points[i].cumulative_distance,
                    points[i].candidate.row_number,
                ),
            )

        selected_indices.append(selected_index)

    return [
        Selection(endpoint=endpoint, point=points[index])
        for endpoint, index in zip(endpoints, selected_indices)
    ]


def write_results(
    output_path: Path,
    selections: list[Selection],
) -> None:
    """
    Write the selected samples to a four-column tab-delimited text file.

    This preserves the old output layout:
        target endpoint
        selected value
        absolute difference
        source PT path
    """
    lines = [
        "endpoint\tcumulative_distance\tabsolute_difference\tsource_pt_path"
    ]

    for selection in selections:
        lines.append(
            "\t".join(
                [
                    f"{selection.endpoint:.15g}",
                    f"{selection.point.cumulative_distance:.15g}",
                    f"{selection.absolute_difference:.15g}",
                    selection.point.candidate.source_pt_path,
                ]
            )
        )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ValueError(
            f"Cannot write output file '{output_path}': {exc}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select representative samples at equally spaced cumulative "
            "arc-length positions along the physical-MAE Pareto front."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="INPUT_FILE",
        help=(
            "Input augmented .pt file or CSV containing physical MAEs, "
            "source paths, and feasibility flags."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("selected_samples.txt"),
        metavar="TXT_FILE",
        help="Output text file; default: selected_samples.txt",
    )

    parser.add_argument(
        "--segments",
        type=int,
        default=4,
        help=(
            "Number of equal arc-length segments; default: 4, "
            "which selects five representative samples."
        ),
    )

    parser.add_argument(
        "--pareto_mode",
        type=str,
        default="feasible",
        choices=["feasible", "all"],
        help=(
            "Compute the Pareto front from feasible candidates only "
            "(default) or from all candidates."
        ),
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.segments <= 0:
        parser.error("--segments must be greater than zero")

    try:
        candidates = read_candidates(args.input)

        pareto = compute_pareto_front(
            candidates=candidates,
            pareto_mode=args.pareto_mode,
        )

        points = cumulative_arc_positions(pareto)

        endpoints = equal_distance_endpoints(args.segments)

        selections = select_nearest_unique_points(
            endpoints=endpoints,
            points=points,
        )

        write_results(args.output, selections)

    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Pareto mode: {args.pareto_mode}"
    )
    print(
        f"Physical-MAE Pareto points: {len(pareto)}"
    )
    print(
        f"Selected {len(selections)} representative samples "
        f"from {len(candidates)} candidates "
        f"and saved them to: {args.output}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
