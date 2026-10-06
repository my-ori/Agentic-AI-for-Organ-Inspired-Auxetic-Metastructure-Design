#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculate_physical_mae.py

Calculate surrogate-predicted physical-space MAEs for every candidate.

Inputs
------
1. merged_all_candidates.pt
   Must contain:
       pred_stress_phys : [N, P_pred]
       pred_nu_phys     : [N, P_pred]

2. target JSON
   Must contain:
       "stress": [P_stress physical stress target values]
       "nu":     [P_nu physical Poisson-ratio target values]

The surrogate always emits its full response grid. Shorter targets describe
the leading points of that grid, matching the flexible-target samplers, so
MAEs use the corresponding prediction prefixes.

Outputs
-------
1. CSV with one row per candidate and:
       stress_mae_phys
       nu_mae_phys
   plus useful source metadata when available.

2. Optional augmented PT containing the original payload plus:
       stress_mae_phys
       nu_mae_phys

This script does NOT compute a Pareto front. That should be done separately
after the physical-space MAEs have been calculated.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate physical-space stress and Poisson-ratio MAEs "
                    "from surrogate-predicted physical response curves."
    )
    parser.add_argument(
        "--pt",
        type=str,
        required=True,
        help="Path to merged_all_candidates.pt",
    )
    parser.add_argument(
        "--target_json",
        type=str,
        required=True,
        help="Path to target JSON containing 'stress' and 'nu' arrays",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="physical_mae_all_candidates.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--out_pt",
        type=str,
        default="",
        help="Optional output PT path containing the original payload plus MAEs",
    )
    return parser.parse_args()


def as_2d_float_tensor(x: object, name: str) -> torch.Tensor:
    if x is None:
        raise KeyError(f"Missing required tensor: {name}")
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.detach().cpu().float()

    if x.ndim != 2:
        raise ValueError(
            f"{name} must have shape [num_candidates, num_points], "
            f"but got {tuple(x.shape)}"
        )
    return x


def get_scalar(payload: dict, key: str, i: int, default=""):
    value = payload.get(key)
    if value is None:
        return default

    if isinstance(value, torch.Tensor):
        if value.ndim >= 1 and value.shape[0] > i:
            item = value[i]
            if item.numel() == 1:
                return item.item()
        return default

    if isinstance(value, list) and len(value) > i:
        return value[i]

    return default


def align_prediction_to_target(
    prediction: torch.Tensor,
    target: torch.Tensor,
    response_name: str,
) -> torch.Tensor:
    """Return the prediction prefix corresponding to the target curve."""
    num_prediction_points = prediction.shape[1]
    num_target_points = target.numel()

    if num_prediction_points < num_target_points:
        raise ValueError(
            f"{response_name} curve length mismatch: prediction has "
            f"{num_prediction_points} points, but target requires "
            f"{num_target_points} points."
        )

    return prediction[:, :num_target_points]


def main() -> None:
    args = parse_args()

    pt_path = Path(args.pt)
    target_path = Path(args.target_json)
    out_csv = Path(args.out_csv)

    # Load merged candidate payload.
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict):
        raise TypeError("Expected the PT file to contain a dictionary payload.")

    pred_stress = as_2d_float_tensor(
        payload.get("pred_stress_phys"), "pred_stress_phys"
    )
    pred_nu = as_2d_float_tensor(
        payload.get("pred_nu_phys"), "pred_nu_phys"
    )

    if pred_stress.shape[0] != pred_nu.shape[0]:
        raise ValueError(
            "Stress and Poisson predictions have different candidate counts: "
            f"{pred_stress.shape[0]} vs {pred_nu.shape[0]}"
        )

    # Load physical target curves.
    with target_path.open("r", encoding="utf-8") as f:
        target = json.load(f)

    if "stress" not in target or "nu" not in target:
        raise KeyError("Target JSON must contain both 'stress' and 'nu'.")

    target_stress = torch.as_tensor(target["stress"], dtype=torch.float32).flatten()
    target_nu = torch.as_tensor(target["nu"], dtype=torch.float32).flatten()

    pred_stress = align_prediction_to_target(
        pred_stress, target_stress, "Stress"
    )
    pred_nu = align_prediction_to_target(
        pred_nu, target_nu, "Poisson-ratio"
    )

    if not torch.isfinite(pred_stress).all():
        raise ValueError("pred_stress_phys contains NaN or Inf values.")
    if not torch.isfinite(pred_nu).all():
        raise ValueError("pred_nu_phys contains NaN or Inf values.")
    if not torch.isfinite(target_stress).all():
        raise ValueError("Target stress contains NaN or Inf values.")
    if not torch.isfinite(target_nu).all():
        raise ValueError("Target nu contains NaN or Inf values.")

    # Physical-space MAE for each candidate.
    #
    # stress_mae_phys[j] =
    #   mean_i |pred_stress_phys[j, i] - target_stress[i]|
    #
    # nu_mae_phys[j] =
    #   mean_i |pred_nu_phys[j, i] - target_nu[i]|
    stress_mae_phys = torch.mean(
        torch.abs(pred_stress - target_stress.unsqueeze(0)),
        dim=1,
    )

    nu_mae_phys = torch.mean(
        torch.abs(pred_nu - target_nu.unsqueeze(0)),
        dim=1,
    )

    n = pred_stress.shape[0]

    # Write one scalar MAE pair per candidate.
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "source_run_id",
                "source_seed",
                "source_epsilon",
                "volfrac",
                "source_selection_is_feasible",
                "stress_mae_phys",
                "nu_mae_phys",
            ]
        )

        for i in range(n):
            writer.writerow(
                [
                    i,
                    get_scalar(payload, "source_run_id", i),
                    get_scalar(payload, "source_seed", i),
                    get_scalar(payload, "source_epsilon", i),
                    get_scalar(payload, "volfrac", i),
                    get_scalar(payload, "source_selection_is_feasible", i),
                    float(stress_mae_phys[i].item()),
                    float(nu_mae_phys[i].item()),
                ]
            )

    # Optionally save an augmented PT for later processing/plotting.
    if args.out_pt:
        out_pt = Path(args.out_pt)
        out_pt.parent.mkdir(parents=True, exist_ok=True)

        augmented = dict(payload)
        augmented["stress_mae_phys"] = stress_mae_phys
        augmented["nu_mae_phys"] = nu_mae_phys
        augmented["target_stress_phys"] = target_stress
        augmented["target_nu_phys"] = target_nu

        torch.save(augmented, out_pt)

    print(f"Candidates: {n}")
    print(f"Stress response points used: {pred_stress.shape[1]}")
    print(f"Nu response points used: {pred_nu.shape[1]}")
    print()
    print("Physical-space MAE summary")
    print(
        f"Stress MAE: min={stress_mae_phys.min().item():.10g}, "
        f"mean={stress_mae_phys.mean().item():.10g}, "
        f"max={stress_mae_phys.max().item():.10g}"
    )
    print(
        f"Nu MAE:     min={nu_mae_phys.min().item():.10g}, "
        f"mean={nu_mae_phys.mean().item():.10g}, "
        f"max={nu_mae_phys.max().item():.10g}"
    )
    print()
    print(f"Saved CSV: {out_csv}")

    if args.out_pt:
        print(f"Saved augmented PT: {args.out_pt}")


if __name__ == "__main__":
    main()
