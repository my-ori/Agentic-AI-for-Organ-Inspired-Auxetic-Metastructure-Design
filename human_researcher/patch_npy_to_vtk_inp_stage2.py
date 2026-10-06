#!/usr/bin/env python3
"""
patch_npy_to_vtk_inp_stage2.py

Stage 2 (sfepy + meshio)
------------------------
Input: selected_summary.json from stage 1, or a root directory containing *_patch.npy files.

For each selected patch:
1) load patch_npy
2) build quad mesh with SfePy
3) write VTK
4) read VTK with meshio
5) pad z=0 if needed
6) write INP

Run this in the environment that has BOTH sfepy and meshio.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import meshio
import numpy as np
from sfepy.discrete.fem import Mesh


def bitmap_to_mesh(mask01: np.ndarray, dx: float, dy: float, x0: float = 0.0, y0: float = 0.0) -> Mesh:
    mask = np.asarray(mask01, dtype=bool)
    ny, nx = mask.shape

    xs = x0 + np.arange(nx + 1) * dx
    ys = y0 + np.arange(ny + 1) * dy
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    coors_full = np.c_[xx.ravel(), yy.ravel()]

    def nid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    cells = []
    for j in range(ny):
        for i in range(nx):
            if not mask[j, i]:
                continue
            n0 = nid(i, j)
            n1 = nid(i + 1, j)
            n2 = nid(i + 1, j + 1)
            n3 = nid(i, j + 1)
            cells.append([n0, n1, n2, n3])

    if len(cells) == 0:
        raise ValueError("No solid pixels found (mask==1).")

    conn_full = np.array(cells, dtype=np.int32)
    used = np.unique(conn_full.ravel())
    remap = -np.ones(coors_full.shape[0], dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)

    coors = coors_full[used]
    conn = remap[conn_full]

    mat_ids = np.zeros(conn.shape[0], dtype=np.int32)
    descs = ["2_4"]
    return Mesh.from_data("bitmap2d", coors, None, [conn], [mat_ids], descs)


def vtk_to_inp(vtk_path: str | Path, inp_path: str | Path) -> Dict[str, int]:
    m = meshio.read(vtk_path)
    if m.points.shape[1] == 2:
        m.points = np.c_[m.points, np.zeros((m.points.shape[0], 1))]
    meshio.write(inp_path, m)
    n_cells = int(sum(len(cb.data) for cb in m.cells))
    return {"nodes": int(m.points.shape[0]), "elements": n_cells}


def load_jobs_from_manifest(manifest_path: Path) -> List[Dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Manifest must be a list.")
    return data


def load_jobs_from_root(root: Path, patch_w: float, patch_h: float) -> List[Dict]:
    jobs = []
    for patch_npy in sorted(root.rglob("*_patch.npy")):
        patch = np.load(patch_npy)
        ny, nx = patch.shape
        jobs.append({
            "rank": -1,
            "patch_npy": str(patch_npy),
            "expected_vtk_path": str(patch_npy.with_suffix(".vtk")),
            "expected_inp_path": str(patch_npy.with_suffix(".inp")),
            "patch_w": float(patch_w),
            "patch_h": float(patch_h),
            "dx": float(patch_w) / float(nx),
            "dy": float(patch_h) / float(ny),
        })
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="", help="selected_summary.json from stage 1")
    ap.add_argument("--root", type=str, default="", help="Root directory to scan for *_patch.npy")
    ap.add_argument("--patch_w", type=float, default=30.0, help="Needed only with --root")
    ap.add_argument("--patch_h", type=float, default=30.0, help="Needed only with --root")
    args = ap.parse_args()

    if not args.manifest and not args.root:
        raise SystemExit("Provide either --manifest or --root")

    if args.manifest:
        jobs = load_jobs_from_manifest(Path(args.manifest))
        base_dir = Path(args.manifest).resolve().parent
    else:
        jobs = load_jobs_from_root(Path(args.root), args.patch_w, args.patch_h)
        base_dir = Path(args.root).resolve()

    rows = []
    for job in jobs:
        patch_npy = Path(job["patch_npy"]).resolve()
        patch = np.load(patch_npy).astype(np.uint8)

        vtk_path = Path(job.get("expected_vtk_path", str(patch_npy.with_suffix(".vtk")))).resolve()
        inp_path = Path(job.get("expected_inp_path", str(patch_npy.with_suffix(".inp")))).resolve()
        vtk_path.parent.mkdir(parents=True, exist_ok=True)
        inp_path.parent.mkdir(parents=True, exist_ok=True)

        dx = float(job["dx"])
        dy = float(job["dy"])

        mesh2d = bitmap_to_mesh(patch, dx, dy, 0.0, 0.0)
        mesh2d.write(str(vtk_path))
        meta = vtk_to_inp(vtk_path, inp_path)

        row = {
            "rank": int(job.get("rank", -1)),
            "patch_npy": str(patch_npy),
            "vtk_path": str(vtk_path),
            "inp_path": str(inp_path),
            "dx": dx,
            "dy": dy,
            "nodes": meta["nodes"],
            "elements": meta["elements"],
            "solid_pixels": int(patch.sum()),
        }
        rows.append(row)
        print(f"[OK] {patch_npy.name} -> {vtk_path.name} -> {inp_path.name}")

    with open(base_dir / "conversion_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "patch_npy", "vtk_path", "inp_path", "dx", "dy", "nodes", "elements", "solid_pixels"])
        for r in rows:
            w.writerow([r["rank"], r["patch_npy"], r["vtk_path"], r["inp_path"], r["dx"], r["dy"], r["nodes"], r["elements"], r["solid_pixels"]])

    with open(base_dir / "conversion_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"[Done: Stage 2] converted={len(rows)} summary_dir={base_dir}")


if __name__ == "__main__":
    main()
