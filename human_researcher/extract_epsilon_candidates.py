#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, Any, List, Optional, Tuple

import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Extract candidate epsilon values from a guided_samples_*.pt file produced by "
            "guided_sample_diffusion_flexible_target.py."
        )
    )
    ap.add_argument("--pt", type=str, required=True, help="Path to guided_samples_*.pt")
    ap.add_argument(
        "--constraint",
        type=str,
        default="nu",
        choices=["nu", "stress"],
        help="Which objective will be treated as the epsilon-constrained objective.",
    )
    ap.add_argument(
        "--space",
        type=str,
        default="normalized",
        choices=["normalized", "physical", "auto"],
        help=(
            "Which metric space to use for MSE computation. Use normalized for compatibility with the current "
            "guided sampling loss. Use physical only if you intentionally want epsilon in physical-unit MSE."
        ),
    )
    ap.add_argument(
        "--strategy",
        type=str,
        default="quantile",
        choices=["quantile", "linear", "log"],
        help="How to propose epsilon values from the observed constraint-loss distribution.",
    )
    ap.add_argument("--num_eps", type=int, default=8, help="Number of candidate epsilons to output.")
    ap.add_argument("--q_low", type=float, default=0.05, help="Lower quantile for quantile strategy.")
    ap.add_argument("--q_high", type=float, default=0.95, help="Upper quantile for quantile strategy.")
    ap.add_argument(
        "--percentiles",
        type=str,
        default="",
        help="Optional explicit percentile list in [0,1], comma-separated. Overrides q_low/q_high/num_eps.",
    )
    ap.add_argument(
        "--out_prefix",
        type=str,
        default="",
        help="Output prefix. Default: same folder as --pt with name epsilon_candidates_<constraint>_<space>",
    )
    return ap.parse_args()


def _find_target(raw: Dict[str, Any], primary: str, legacy: str) -> Optional[torch.Tensor]:
    if primary in raw:
        return torch.tensor(raw[primary], dtype=torch.float32)
    if legacy in raw:
        return torch.tensor(raw[legacy], dtype=torch.float32)
    return None


def _mse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    n = int(target.numel())
    if pred.ndim != 2:
        raise ValueError(f"Expected prediction tensor with shape [N, D], got {tuple(pred.shape)}")
    if pred.shape[1] < n:
        raise ValueError(f"Prediction dimension {pred.shape[1]} is smaller than target length {n}")
    diff = pred[:, :n] - target.view(1, -1)
    return diff.pow(2).mean(dim=1)



def _quantile_values(values: torch.Tensor, qs: List[float]) -> List[float]:
    out: List[float] = []
    for q in qs:
        q = min(max(float(q), 0.0), 1.0)
        out.append(float(torch.quantile(values, q).item()))
    return out



def _unique_sorted_with_tol(vals: List[float], rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> List[float]:
    vals_sorted = sorted(float(v) for v in vals)
    uniq: List[float] = []
    for v in vals_sorted:
        if not uniq:
            uniq.append(v)
        elif not math.isclose(v, uniq[-1], rel_tol=rel_tol, abs_tol=abs_tol):
            uniq.append(v)
    return uniq



def propose_eps(values: torch.Tensor, strategy: str, num_eps: int, q_low: float, q_high: float, percentiles: str) -> Tuple[List[float], List[float]]:
    values = values.detach().float().cpu()
    if values.numel() == 0:
        raise ValueError("No values available for epsilon extraction.")

    if percentiles.strip():
        qs = [float(x.strip()) for x in percentiles.split(",") if x.strip()]
        if not qs:
            raise ValueError("--percentiles was provided but no valid values were parsed.")
        eps = _quantile_values(values, qs)
        return _unique_sorted_with_tol(eps), qs

    lo = float(torch.min(values).item())
    hi = float(torch.max(values).item())

    if strategy == "quantile":
        if num_eps <= 1:
            qs = [0.5 * (q_low + q_high)]
        else:
            qs = [q_low + (q_high - q_low) * i / (num_eps - 1) for i in range(num_eps)]
        eps = _quantile_values(values, qs)
        return _unique_sorted_with_tol(eps), qs

    if strategy == "linear":
        if num_eps <= 1:
            eps = [0.5 * (lo + hi)]
        else:
            eps = [lo + (hi - lo) * i / (num_eps - 1) for i in range(num_eps)]
        return _unique_sorted_with_tol(eps), []

    if strategy == "log":
        positive = values[values > 0]
        if positive.numel() == 0:
            raise ValueError("Cannot use log strategy because all observed losses are zero.")
        lo_pos = float(torch.min(positive).item())
        hi_pos = max(float(torch.max(positive).item()), lo_pos)
        if num_eps <= 1 or math.isclose(lo_pos, hi_pos):
            eps = [lo_pos]
        else:
            log_lo = math.log10(lo_pos)
            log_hi = math.log10(hi_pos)
            eps = [10 ** (log_lo + (log_hi - log_lo) * i / (num_eps - 1)) for i in range(num_eps)]
        return _unique_sorted_with_tol(eps), []

    raise ValueError(f"Unknown strategy: {strategy}")



def main() -> None:
    args = parse_args()
    payload = torch.load(args.pt, map_location="cpu")

    # Detect available targets.
    target_raw = payload.get("target_raw", {})
    target_stress_norm = payload.get("target_stress_norm")
    target_nu_norm = payload.get("target_nu_norm")

    target_stress_phys = _find_target(target_raw, "stress", "y_stress")
    target_nu_phys = _find_target(target_raw, "nu", "y_nu")

    requested_space = args.space
    space = requested_space
    if requested_space == "auto":
        if args.constraint == "nu" and target_nu_norm is not None and "pred_nu_norm" in payload:
            space = "normalized"
        elif args.constraint == "stress" and target_stress_norm is not None and "pred_stress_norm" in payload:
            space = "normalized"
        else:
            space = "physical"

    # Compute both stress and nu per-sample losses when possible.
    stress_mse_norm = None
    nu_mse_norm = None
    stress_mse_phys = None
    nu_mse_phys = None

    if target_stress_norm is not None and "pred_stress_norm" in payload:
        stress_mse_norm = _mse_per_sample(payload["pred_stress_norm"].float(), target_stress_norm.float())
    if target_nu_norm is not None and "pred_nu_norm" in payload:
        nu_mse_norm = _mse_per_sample(payload["pred_nu_norm"].float(), target_nu_norm.float())
    if target_stress_phys is not None and "pred_stress_phys" in payload:
        stress_mse_phys = _mse_per_sample(payload["pred_stress_phys"].float(), target_stress_phys.float())
    if target_nu_phys is not None and "pred_nu_phys" in payload:
        nu_mse_phys = _mse_per_sample(payload["pred_nu_phys"].float(), target_nu_phys.float())

    if args.constraint == "nu":
        values = nu_mse_norm if space == "normalized" else nu_mse_phys
        paired = stress_mse_norm if space == "normalized" else stress_mse_phys
        if values is None:
            raise ValueError(f"Nu predictions/targets are not available in {space} space.")
        primary_name = "stress"
    else:
        values = stress_mse_norm if space == "normalized" else stress_mse_phys
        paired = nu_mse_norm if space == "normalized" else nu_mse_phys
        if values is None:
            raise ValueError(f"Stress predictions/targets are not available in {space} space.")
        primary_name = "nu"

    eps_list, qs = propose_eps(
        values=values,
        strategy=args.strategy,
        num_eps=int(args.num_eps),
        q_low=float(args.q_low),
        q_high=float(args.q_high),
        percentiles=args.percentiles,
    )

    pt_dir = os.path.dirname(os.path.abspath(args.pt))
    if args.out_prefix:
        out_prefix = args.out_prefix
    else:
        out_prefix = os.path.join(pt_dir, f"epsilon_candidates_{args.constraint}_{space}")

    rows: List[Dict[str, Any]] = []
    for i, eps in enumerate(eps_list):
        mask = values <= eps
        count = int(mask.sum().item())
        frac = float(mask.float().mean().item())
        row: Dict[str, Any] = {
            "index": i,
            "epsilon": float(eps),
            "num_feasible_in_file": count,
            "fraction_feasible_in_file": frac,
        }
        if qs:
            row["source_quantile"] = float(qs[min(i, len(qs) - 1)])
        if count > 0 and paired is not None:
            row[f"best_{primary_name}_mse_among_feasible"] = float(paired[mask].min().item())
            row[f"mean_{primary_name}_mse_among_feasible"] = float(paired[mask].mean().item())
        rows.append(row)

    stats = {
        "num_samples_in_file": int(values.numel()),
        "constraint": args.constraint,
        "space_used": space,
        "requested_space": requested_space,
        "strategy": args.strategy,
        "num_eps_requested": int(args.num_eps),
        "constraint_mse_min": float(values.min().item()),
        "constraint_mse_max": float(values.max().item()),
        "constraint_mse_mean": float(values.mean().item()),
        "constraint_mse_median": float(values.median().item()),
        "suggested_epsilons": [float(x) for x in eps_list],
        "note": (
            "These epsilon values are extracted from the already-selected samples stored in the PT file. "
            "If the file itself was produced by a biased guidance setting, the suggested range will inherit that bias."
        ),
        "sampling_args": payload.get("sampling_args", {}),
    }

    json_path = out_prefix + ".json"
    csv_path = out_prefix + ".csv"
    per_sample_path = out_prefix + "_per_sample.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": stats, "rows": rows}, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = ["index", "epsilon", "num_feasible_in_file", "fraction_feasible_in_file"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Also export the full per-sample loss table for plotting / manual selection.
    per_rows: List[Dict[str, Any]] = []
    num_samples = int(values.numel())
    for i in range(num_samples):
        row: Dict[str, Any] = {
            "sample_index": i,
            f"{args.constraint}_mse_{space}": float(values[i].item()),
        }
        if paired is not None:
            row[f"{primary_name}_mse_{space}"] = float(paired[i].item())
        per_rows.append(row)

    # Sort by constraint error for convenience.
    per_rows.sort(key=lambda d: d[f"{args.constraint}_mse_{space}"])
    with open(per_sample_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(per_rows[0].keys()) if per_rows else ["sample_index", f"{args.constraint}_mse_{space}"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_rows)

    print(f"[Done] wrote {json_path}")
    print(f"[Done] wrote {csv_path}")
    print(f"[Done] wrote {per_sample_path}")
    print("Suggested epsilons:")
    for eps in eps_list:
        print(f"  {eps:.10g}")


if __name__ == "__main__":
    main()
