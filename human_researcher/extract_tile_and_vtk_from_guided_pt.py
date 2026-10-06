#!/usr/bin/env python3
"""
extract_tile_and_vtk_from_guided_pt.py

Load a guided diffusion output .pt file (for example guided_samples_320.pt),
extract selected designs by index, tile each unit cell into an nx-by-ny patch,
save preview images / .npy masks, and optionally write .vtk and .inp meshes.

This script is designed to match the payload structure written by
`guided_sample_diffusion.py`, which stores the final kept samples in keys such as
`x01_bin` and `xm11`.

Typical usage
-------------
python extract_tile_and_vtk_from_guided_pt.py \
  --pt runs/ddpm/guided/guided_samples_320.pt \
  --indices 0,10,13 \
  --tile_nx 3 --tile_ny 3 \
  --patch_w 30.0 --patch_h 30.0 \
  --out_dir selected_for_fe

If sfepy and meshio are installed, this will also write .vtk and .inp files.
If not, it will still save *_patch.npy plus a selected_summary.json that can be
fed into your existing patch_npy_to_vtk_inp_stage2.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

# Optional plotting
try:
    from PIL import Image
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

# Optional mesh writing
try:
    import meshio  # type: ignore
    from sfepy.discrete.fem import Mesh  # type: ignore
    HAVE_MESH = True
except Exception:
    meshio = None
    Mesh = None
    HAVE_MESH = False


def parse_indices(text: str) -> List[int]:
    items = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    if not items:
        raise ValueError("No valid indices were provided.")
    return items


def load_design_stack(pt_path: Path, threshold: float) -> np.ndarray:
    payload = torch.load(pt_path, map_location="cpu")

    if not isinstance(payload, dict):
        raise ValueError("Expected the .pt file to contain a dict payload.")

    if "x01_bin" in payload:
        x = payload["x01_bin"]
    elif "xm11" in payload:
        x = (payload["xm11"] > float(threshold)).float()
    else:
        raise KeyError("Could not find either 'x01_bin' or 'xm11' in the .pt payload.")

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    # expected shapes: (N,1,H,W) or (N,H,W)
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0, :, :]
    elif x.ndim != 3:
        raise ValueError(f"Unexpected design array shape: {x.shape}")

    x = (x > 0.5).astype(np.uint8)
    return x


def tile_patch(mask01: np.ndarray, tile_nx: int, tile_ny: int) -> np.ndarray:
    return np.tile(mask01.astype(np.uint8), (tile_ny, tile_nx))


def save_mask_png(mask01: np.ndarray, png_path: Path, upscale: int = 8) -> None:
    if not HAVE_PIL:
        return
    # PNG preview only: swap black/white so solid=black and void=white.
    # This does not change the saved .npy mask or the VTK/INP generation.
    arr = ((1 - mask01.astype(np.uint8)) * 255)
    im = Image.fromarray(arr, mode="L")
    if upscale > 1:
        im = im.resize((im.width * upscale, im.height * upscale), resample=Image.NEAREST)
    im.save(png_path)


def bitmap_to_mesh(mask01: np.ndarray, dx: float, dy: float, x0: float = 0.0, y0: float = 0.0):
    if not HAVE_MESH:
        raise RuntimeError("bitmap_to_mesh requires sfepy and meshio.")

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


def vtk_to_inp(vtk_path: Path, inp_path: Path) -> Dict[str, int]:
    if not HAVE_MESH:
        raise RuntimeError("vtk_to_inp requires meshio.")

    m = meshio.read(vtk_path)
    if m.points.shape[1] == 2:
        m.points = np.c_[m.points, np.zeros((m.points.shape[0], 1))]
    meshio.write(inp_path, m)
    n_cells = int(sum(len(cb.data) for cb in m.cells))
    return {"nodes": int(m.points.shape[0]), "elements": n_cells}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Path to guided_samples_*.pt")
    ap.add_argument("--indices", default="0", help="Comma-separated sample indices, e.g. 0,10,13")
    ap.add_argument("--out_dir", default="fe")
    ap.add_argument("--tile_nx", type=int, default=3)
    ap.add_argument("--tile_ny", type=int, default=3)
    ap.add_argument("--patch_w", type=float, default=30.0, help="Overall physical width of the tiled patch")
    ap.add_argument("--patch_h", type=float, default=30.0, help="Overall physical height of the tiled patch")
    ap.add_argument("--binarize_threshold", type=float, default=0.0, help="Used only when x01_bin is absent and xm11 is used")
    ap.add_argument("--png_upscale", type=int, default=8)
    ap.add_argument("--write_vtk", action="store_true", help="Write .vtk directly if sfepy+meshio are available")
    ap.add_argument("--write_inp", action="store_true", help="Write .inp directly if mesh tools are available")
    args = ap.parse_args()

    pt_path = Path(args.pt).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = parse_indices(args.indices)
    all_masks = load_design_stack(pt_path, threshold=args.binarize_threshold)

    n_total = int(all_masks.shape[0])
    bad = [idx for idx in indices if idx < 0 or idx >= n_total]
    if bad:
        raise IndexError(f"Indices out of range: {bad}. Valid range is [0, {n_total - 1}].")

    rows: List[Dict] = []

    for idx in indices:
        unit = all_masks[idx]
        patch = tile_patch(unit, tile_nx=args.tile_nx, tile_ny=args.tile_ny)

        ny, nx = patch.shape
        dx = float(args.patch_w) / float(nx)
        dy = float(args.patch_h) / float(ny)

        stem = f"design_{idx:03d}_tile{args.tile_nx}x{args.tile_ny}"
        unit_npy = out_dir / f"{stem}_unit.npy"
        patch_npy = out_dir / f"{stem}_patch.npy"
        unit_png = out_dir / f"{stem}_unit.png"
        patch_png = out_dir / f"{stem}_patch.png"
        vtk_path = out_dir / f"{stem}.vtk"
        inp_path = out_dir / f"{stem}.inp"

        np.save(unit_npy, unit.astype(np.uint8))
        np.save(patch_npy, patch.astype(np.uint8))
        save_mask_png(unit, unit_png, upscale=args.png_upscale)
        save_mask_png(patch, patch_png, upscale=max(1, args.png_upscale // 2))

        vtk_written = False
        inp_written = False
        mesh_meta: Dict[str, int] = {}

        if (args.write_vtk or args.write_inp) and HAVE_MESH:
            mesh2d = bitmap_to_mesh(patch, dx=dx, dy=dy, x0=0.0, y0=0.0)
            mesh2d.write(str(vtk_path))
            vtk_written = True
            if args.write_inp:
                mesh_meta = vtk_to_inp(vtk_path, inp_path)
                inp_written = True
        elif args.write_vtk or args.write_inp:
            print("[WARN] sfepy/meshio not available in this environment, so VTK/INP were not written.")

        rows.append(
            {
                "rank": int(idx),
                "selected_index": int(idx),
                "pt_path": str(pt_path),
                "unit_npy": str(unit_npy),
                "patch_npy": str(patch_npy),
                "unit_png": str(unit_png),
                "patch_png": str(patch_png),
                "expected_vtk_path": str(vtk_path),
                "expected_inp_path": str(inp_path),
                "vtk_written": bool(vtk_written),
                "inp_written": bool(inp_written),
                "patch_w": float(args.patch_w),
                "patch_h": float(args.patch_h),
                "tile_nx": int(args.tile_nx),
                "tile_ny": int(args.tile_ny),
                "unit_shape": [int(unit.shape[0]), int(unit.shape[1])],
                "patch_shape": [int(patch.shape[0]), int(patch.shape[1])],
                "dx": float(dx),
                "dy": float(dy),
                "solid_pixels_unit": int(unit.sum()),
                "solid_pixels_patch": int(patch.sum()),
                **mesh_meta,
            }
        )

        print(f"[OK] index={idx:03d} -> {patch_npy.name}" + (f" -> {vtk_path.name}" if vtk_written else ""))

    with open(out_dir / "selected_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"[Done] wrote {len(rows)} selected designs to: {out_dir}")
    print(f"[Done] summary: {out_dir / 'selected_summary.json'}")
    if not HAVE_MESH:
        print("[Note] You can now run your existing stage-2 converter on this folder or summary JSON.")


if __name__ == "__main__":
    main()
