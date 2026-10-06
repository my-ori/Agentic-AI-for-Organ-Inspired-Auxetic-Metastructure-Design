from __future__ import absolute_import
import numpy as np
from sfepy.discrete.fem import Mesh
import os
import csv
from sfepy.base.base import output
# ------------------------------------------------------------
# All Custom Variables
# ------------------------------------------------------------
strain_final = 0.30
n_step = int(10 * 100 * strain_final) + 1
resol = 0.2
NCX, NCY = 3, 3
IC, JC = 1, 1                # central cell index
MU    = -2.6788     # MPa
KAPPA = 4.3142     # MPa



# ------------------------------------------------------------
# 1) Mesh + target strain
# ------------------------------------------------------------
#filename_mesh = "design_000_tile3x3.vtk"
filename_mesh = os.path.join(os.getcwd(), "design_000_tile3x3.vtk")

_mesh = Mesh.from_file(filename_mesh)
coors = _mesh.coors

x_min = float(coors[:, 0].min())
x_max = float(coors[:, 0].max())
y_min = float(coors[:, 1].min())
y_max = float(coors[:, 1].max())

Lx = x_max - x_min
Ly = y_max - y_min
ux_right_final = strain_final * Lx

# Robust tolerances for boundary selection
tolx = 1e-8 * max(Lx, 1.0)
toly = 1e-8 * max(Ly, 1.0)
# "Grip" width: apply BCs over a strip of a few pixel columns
grip_w = 0.1 * resol

tol_anchor_y = max(toly, 0.51 * resol)

bottom_ids = np.where(coors[:, 1] < (y_min + tol_anchor_y))[0]
if bottom_ids.size == 0:
    raise RuntimeError("No vertices found on the bottom strip - increase tol_anchor_y!")
# leftmost among bottom vertices
anchor_id = int(bottom_ids[np.argmin(coors[bottom_ids, 0])])
anchor_xy = coors[anchor_id]


# Time-dependent essential BC: ramps 0 -> ux_right_final
def ramp_ux(ts, coor, **kwargs):
    return np.full((coor.shape[0],), ux_right_final * ts.nt, dtype=np.float64)

functions = {
    "ramp_ux": (ramp_ux,),
}
_reaction_hist = []
_w0 = None
_h0 = None
_prev_ex = None
_prev_ey = None
CSV_NAME = "reaction_right_grip.csv"
_csv_header_written = False

def step_hook(pb, ts, variables, **kwargs):

    # Region and field.
    rreg = pb.domain.regions['RightGrip']
    fu = pb.fields['displacement']

    # Current mean displacement/strain on RightGrip (engineering strain in x).
    u_on_r = variables['u'].get_state_in_region(rreg, reshape=True)
    ux_mean = float(u_on_r[:, 0].mean())
    eng_strain = ux_mean / Lx

    # ------------------------------------------------------------
    # Poisson ratio for central unit cell (gauge-strip method)
    # ------------------------------------------------------------
    global _w0, _h0, _prev_ex, _prev_ey

    mesh_coors = pb.domain.mesh.coors

    regL = pb.domain.regions["CenL"]
    regR = pb.domain.regions["CenR"]
    regB = pb.domain.regions["CenB"]
    regT = pb.domain.regions["CenT"]

    # Safety check: ensure strips are not empty
    if (regL.vertices.size == 0) or (regR.vertices.size == 0) or (regB.vertices.size == 0) or (regT.vertices.size == 0):
        output(f"[poisson] WARNING: empty gauge strip(s): "
               f"L={regL.vertices.size}, R={regR.vertices.size}, B={regB.vertices.size}, T={regT.vertices.size}. ")
        ex_cell = ey_cell = nu_sec = nu_inc = np.nan
    else:
        # Reference coordinates for each strip
        X0L = mesh_coors[regL.vertices]
        X0R = mesh_coors[regR.vertices]
        X0B = mesh_coors[regB.vertices]
        X0T = mesh_coors[regT.vertices]

        # Displacements on each strip (same ordering as vertices)
        uL = variables["u"].get_state_in_region(regL, reshape=True)
        uR = variables["u"].get_state_in_region(regR, reshape=True)
        uB = variables["u"].get_state_in_region(regB, reshape=True)
        uT = variables["u"].get_state_in_region(regT, reshape=True)

        # Deformed coordinates
        xL_def = X0L[:, 0] + uL[:, 0]
        xR_def = X0R[:, 0] + uR[:, 0]
        yB_def = X0B[:, 1] + uB[:, 1]
        yT_def = X0T[:, 1] + uT[:, 1]

        # Gauge lengths (mean right - mean left, etc.)
        w = float(xR_def.mean() - xL_def.mean())
        h = float(yT_def.mean() - yB_def.mean())

        # Initialize reference lengths once (step 0 typically)
        if (_w0 is None) or (_h0 is None):
            w0 = float(X0R[:, 0].mean() - X0L[:, 0].mean())
            h0 = float(X0T[:, 1].mean() - X0B[:, 1].mean())
            _w0, _h0 = w0, h0
            output(f"[poisson] init w0={_w0:.6g}, h0={_h0:.6g}")

        # Engineering strains (secant from reference)
        ex_cell = (w - _w0) / _w0
        ey_cell = (h - _h0) / _h0

        # Secant Poisson (from reference)
        nu_sec = (-ey_cell / ex_cell) if abs(ex_cell) > 1e-14 else np.nan

        # Incremental Poisson (more stable in nonlinear regime)
        if (_prev_ex is None) or (_prev_ey is None):
            nu_inc = np.nan
        else:
            dex = ex_cell - _prev_ex
            dey = ey_cell - _prev_ey
            nu_inc = (-dey / dex) if abs(dex) > 1e-14 else np.nan

        _prev_ex, _prev_ey = ex_cell, ey_cell

        output(f"[poisson] ex={ex_cell:.6e} ey={ey_cell:.6e} nu_sec={nu_sec:.6e} nu_inc={nu_inc:.6e}")


    # ------------------------------------------------------------
    # Reaction Force
    # ------------------------------------------------------------
    # --- Key idea: compute residuals with EBCs temporarily removed. ---
    # Save original EBCs.
    ebcs0 = pb.ebcs

    # Get the full state vector of the *solved* problem (includes EBC values).
    vec_full = variables.get_state(reduced=False, force=False)

    # Remove EBCs and update equation mapping (do NOT rebuild matrix graph).
    pb.time_update(ts=ts, ebcs={}, is_matrix=False, create_matrix=False)

    # Set the same solved state into the "no-EBC" variables.
    v2 = pb.get_variables()
    v2.set_state(vec_full, reduced=False, force=False, apply_ebc=False)

    # Assemble residuals. Reaction forces live mainly on the originally constrained DOFs.
    res = pb.equations.eval_residuals(v2())

    # DOFs on RightGrip (see SfePy FAQ).
    ii = fu.get_dofs_in_region(rreg)
    idofx = fu.n_components * ii + 0  # x components
    idofy = fu.n_components * ii + 1  # y components

    Rx = float(res[idofx].sum())
    Ry = float(res[idofy].sum())
    # --- write CSV each step ---
    global _csv_header_written
    os.makedirs(pb.output_dir, exist_ok=True)
    csv_path = os.path.join(pb.output_dir, CSV_NAME)

    if not _csv_header_written:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "time", "nt", "ux_mean", "eng_strain", "Rx", "Ry", "ex_cell", "ey_cell", "nu_sec", "nu_inc"])
        _csv_header_written = True

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([ts.step, float(ts.time), float(ts.nt), ux_mean, eng_strain, Rx, Ry, ex_cell, ey_cell, nu_sec, nu_inc])
    # Restore original EBCs.
    pb.time_update(ts=ts, ebcs=ebcs0, is_matrix=False, create_matrix=False)

    _reaction_hist.append((ts.step, float(ts.time), float(ts.nt), ux_mean, eng_strain, Rx, Ry, ex_cell, ey_cell, nu_sec, nu_inc))

    output(f"[reaction] step {ts.step:4d}  strain={eng_strain:.6e}  Rx={Rx:.6e}  Ry={Ry:.6e}")

 
# ------------------------------------------------------------
# Central unit-cell gauge definition for Poisson ratio
# ------------------------------------------------------------

cell_w = Lx / NCX
cell_h = Ly / NCY

# Nominal central-cell bounding box (reference coordinates)
xL = x_min + IC * cell_w
xR = x_min + (IC + 1) * cell_w
yB = y_min + JC * cell_h
yT = y_min + (JC + 1) * cell_h

# Interior window used to bound the strips in the orthogonal direction
xLi = xL
xRi = xR
yBi = yB
yTi = yT
    
    
# ------------------------------------------------------------
# 2) Regions
# ------------------------------------------------------------
edge_tol = max(0.1 * resol, tolx, toly)
regions = {
    "Omega": "all",
    "LeftGrip":  (f"vertices in (x < {x_min + tolx})", "facet"),
    "RightGrip": (f"vertices in (x > {x_max - tolx})", "facet"),
    "Anchor": (f"vertex {anchor_id}", "vertex"),
    "PRef": (f"vertex {anchor_id}", "vertex"),

    # --- Poisson gauge regions (central cell) ---
    "CentralCell": (f"vertices in ((x > {xL}) & (x < {xR}) & (y > {yB}) & (y < {yT}))", "vertex"),
    "CenL": (f"vertices in ((x >= {xL - edge_tol}) & (x <= {xL + edge_tol}) & (y >= {yB - edge_tol}) & (y <= {yT + edge_tol}))", "vertex"),
    "CenR": (f"vertices in ((x >= {xR - edge_tol}) & (x <= {xR + edge_tol}) & (y >= {yB - edge_tol}) & (y <= {yT + edge_tol}))", "vertex"),
    "CenB": (f"vertices in ((y >= {yB - edge_tol}) & (y <= {yB + edge_tol}) & (x >= {xL - edge_tol}) & (x <= {xR + edge_tol}))", "vertex"),
    "CenT": (f"vertices in ((y >= {yT - edge_tol}) & (y <= {yT + edge_tol}) & (x >= {xL - edge_tol}) & (x <= {xR + edge_tol}))", "vertex"),
}


# ------------------------------------------------------------
# 3) Material: Total Lagrangian Neo-Hookean (finite strain)
# ------------------------------------------------------------
# Pick parameters from (E, nu) as small-strain match

materials = {
    "solid": ({"mu": MU, "kappa": KAPPA,},),
}

# ------------------------------------------------------------
# 4) Fields & Variables
# ------------------------------------------------------------
fields = {
    "displacement": ("real", "vector", "Omega", 1),
    "pressure":     ("real", "scalar", "Omega", 0),
}
variables = {
    "u": ("unknown field", "displacement", 0),
    "v": ("test field", "displacement", "u"),
    "p": ("unknown field", "pressure", 1),
    "q": ("test field",    "pressure", "p"),
}

# ------------------------------------------------------------
# 5) Boundary conditions
# ------------------------------------------------------------
ebcs = {
    "FixLeft": ("LeftGrip", {"u.0": 0.0}),
    "PullRightRamp": ("RightGrip", {"u.0": "ramp_ux"}),  # uy free
    "FixAnchorY": ("Anchor", {"u.1": 0.0}),
    "FixPRef": ("PRef", {"p.0": 0.0}),
}


# ------------------------------------------------------------
# 6) Equations (finite strain, Total Lagrangian)
# ------------------------------------------------------------
integrals = {"i": 2}

equations = {
    "balance": r"""
        dw_tl_he_neohook.i.Omega(solid.mu, v, u)
      + dw_tl_he_mooney_rivlin.i.Omega(solid.kappa, v, u)
      + dw_tl_bulk_pressure.i.Omega(v, u, p)
      = 0
    """,
    
    "incompressibility": r"""
        dw_tl_volume.i.Omega(q, u)
      - dw_integrate.i.Omega(q)
      = 0
    """,
    
}
# ------------------------------------------------------------
# 7) Solvers 
# ------------------------------------------------------------
solvers = {
    "ls": ("ls.scipy_superlu", {}),

    "newton": ("nls.newton", {
        "i_max": 10,
        "eps_a": 1e-8,
        "eps_r": 1e-7,

        # Line search settings (helps prevent overly aggressive Newton steps)
        "ls_on": 0.9,
        "ls_red": 0.5,
        "ls_min": 1e-6,
        "ls_red_warp": 0.001,
    }),

    "ts": ("ts.simple", {
        "t0": 0.0,
        "t1": 1.0,
        "n_step": n_step,
        "verbose": 1,
    }),
}
output_dir = os.path.join(os.getcwd(), "out_bitmap2d")
options = {
    "ts": "ts",
    "nls": "newton",
    "ls": "ls",
    "save_times": "all",
    "output_dir": output_dir,
    "save_results": False, 
    "step_hook": "step_hook",
}
