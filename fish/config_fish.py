"""
config_fish.py — Parameters for 3D CLBM-IBM bio-inspired swimmer (D3Q19).

Body: 4:1 ellipse (chord 100 lu, thickness 25 lu) approximating a fish cross-section.
Motion: large-amplitude heave (0.25c) + pitch (15°) with 90° phase lag (k = 0.8).
This regime produces a reverse Kármán vortex street — the hallmark of thrust generation
in biological swimming.

All physical quantities in lattice units (lu / step).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Fluid domain
# ---------------------------------------------------------------------------
Nx, Ny = 800, 250
Nz     = 25           # spanwise extent (periodic BC)

U_inf = 0.05          # free-stream velocity [lu/step]
Re    = 300           # Reynolds number (chord-based)

# ---------------------------------------------------------------------------
# CLBM D3Q19 parameters
# ---------------------------------------------------------------------------
cs2  = 1.0 / 3.0
nu   = U_inf * 100 / Re       # kinematic viscosity  (chord c = 100 lu)
tau  = 0.5 + nu / cs2         # shear relaxation time
s_v  = 1.0 / tau
s_e  = 1.0
s_q  = 1.0
s_pi = 1.0

# ---------------------------------------------------------------------------
# D3Q19 lattice constants (same ordering as config_moving.py)
# ---------------------------------------------------------------------------
EX = ( 0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0)
EY = ( 0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1)
EZ = ( 0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1)
W  = (
    1/3,
    1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
    1/36, 1/36, 1/36, 1/36,
    1/36, 1/36, 1/36, 1/36,
    1/36, 1/36, 1/36, 1/36,
)

# ---------------------------------------------------------------------------
# 19×19 M_raw matrix and its inverse (required by clbm_fish.py)
# ---------------------------------------------------------------------------
_pqr = [
    (0,0,0), (1,0,0), (0,1,0), (0,0,1),
    (1,1,0), (1,0,1), (0,1,1),
    (2,0,0), (0,2,0), (0,0,2),
    (2,1,0), (1,2,0), (2,0,1), (0,2,1), (1,0,2), (0,1,2),
    (2,2,0), (2,0,2), (0,2,2),
]
_M_raw = np.array([
    [int(EX[c])**p * int(EY[c])**q * int(EZ[c])**r for c in range(19)]
    for (p, q, r) in _pqr
], dtype=np.float64)
M_raw_inv_np = np.linalg.inv(_M_raw).astype(np.float32)

# ---------------------------------------------------------------------------
# Fish body geometry — 4:1 ellipse in local frame (reference point = origin)
# ---------------------------------------------------------------------------
c        = 100           # chord / major-axis length [lu]
x0       = 200.0         # initial reference-point x
y0       = float(Ny) / 2.0
ds_IBM   = 0.7           # Lagrangian marker spacing [lu]
AoA0_deg = 0.0           # no mean incidence for swimmer


def _ellipse_markers(chord, thickness, ds):
    """
    Generate 4:1 ellipse surface markers in local frame (reference = origin).
    Semi-axes: a = chord/2 along x,  b = thickness/2 along y.

    Returns X_tmpl, Y_tmpl : np.ndarray float32, shape (N,)
    """
    a = chord / 2.0
    b = thickness / 2.0
    # Approximate perimeter → number of markers
    perim = np.pi * (3*(a + b) - np.sqrt((3*a + b)*(a + 3*b)))
    N = max(int(np.ceil(perim / ds)), 64)
    if N % 2 != 0:
        N += 1
    theta = np.linspace(0.0, 2.0 * np.pi, N + 1)[:-1]
    X = (a * np.cos(theta)).astype(np.float32)
    Y = (b * np.sin(theta)).astype(np.float32)
    return X, Y


FISH_THICKNESS = 25     # minor-axis length [lu] — 25% thickness, fish-like
X_template, Y_template = _ellipse_markers(c, FISH_THICKNESS, ds_IBM)
N_MARKERS_2D = len(X_template)

# ---------------------------------------------------------------------------
# 3D marker arrays — uniform z-extrusion (one layer per spanwise plane)
# ---------------------------------------------------------------------------
ds_z      = 1.0
z_layers  = np.arange(0, Nz, ds_z, dtype=np.float32)
Nz_layers = len(z_layers)

X_marker_np = np.tile(X_template, Nz_layers).astype(np.float32)
Y_marker_np = np.tile(Y_template, Nz_layers).astype(np.float32)
Z_marker_np = np.repeat(z_layers, N_MARKERS_2D).astype(np.float32)
N_MARKERS   = len(X_marker_np)

# ---------------------------------------------------------------------------
# Prescribed motion — large-amplitude swimming kinematics
#   y_c(t)  = A_heave * sin(ω t)          (heave)
#   θ(t)    = A_pitch * sin(ω t + φ)      (pitch, leads by 90°)
# Reduced frequency k = 0.8 → reverse Kármán (thrust) regime
# ---------------------------------------------------------------------------
k_reduced = 0.8
f_motion  = k_reduced * U_inf / (np.pi * c)   # oscillation frequency [step^-1]

A_heave   = 25.0                               # heave amplitude [lu]  (= 0.25c)
A_pitch   = np.radians(15.0)                   # pitch amplitude [rad] (= 15°)
phi_pitch = np.pi / 2.0                        # heave leads pitch
AoA0_rad  = np.radians(AoA0_deg)              # 0 rad for swimmer

# ---------------------------------------------------------------------------
# Time stepping
# ---------------------------------------------------------------------------
N_steps      = 25000
vis_interval = 150      # frames at 15 fps → 166 frames ≈ 11 s video

# ---------------------------------------------------------------------------
# Video output
# ---------------------------------------------------------------------------
VIDEO_FPS  = 15
VIDEO_UMAG = "fish_umag.mp4"        # velocity magnitude  (viridis)
VIDEO_VORT = "fish_vorticity.mp4"   # z-vorticity         (RdBu_r)
VIDEO_PRES = "fish_pressure.mp4"    # pressure deviation  (seismic)
VIDEO_COMP = "fish_composite.mp4"   # 2×2 composite panel

# ---------------------------------------------------------------------------
# Visualization colour-scale limits
# ---------------------------------------------------------------------------
# Boundary-layer vorticity scale: U_inf / δ_BL ≈ U_inf * sqrt(Re/c) ≈ 0.0087
# Shed vortex cores are weaker; ±0.004 saturates the body but preserves the wake.
VORT_MAX = 4.0 * U_inf / c    # ≈ 0.002  (symmetric ± range for RdBu_r)

# Pressure deviation scale: ~3 × dynamic pressure = 3 × 0.5 × rho × U_inf²
PRES_MAX = 3.0 * 0.5 * U_inf ** 2   # ≈ 0.00375

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Grid         : {Nx} × {Ny} × {Nz}")
    print(f"Re           : {Re}   U_inf : {U_inf}   c : {c} lu")
    print(f"nu           : {nu:.5f}   tau : {tau:.4f}   s_v : {s_v:.4f}")
    print(f"Body         : ellipse {c}×{FISH_THICKNESS} lu  (aspect {c//FISH_THICKNESS}:1)")
    print(f"N_MARKERS_2D : {N_MARKERS_2D}   Nz_layers : {Nz_layers}   N_MARKERS : {N_MARKERS}")
    period = 1.0 / f_motion
    print(f"k            : {k_reduced}   f : {f_motion:.2e} step^-1   T : {period:.0f} steps")
    print(f"A_heave      : {A_heave} lu ({A_heave/c:.2f}c)   "
          f"A_pitch : {np.degrees(A_pitch):.1f}°   phi : 90°")
    print(f"N_steps      : {N_steps}  ({N_steps*f_motion:.1f} periods)   "
          f"vis_interval : {vis_interval}   frames : {N_steps//vis_interval}")
    print(f"VORT_MAX     : ±{VORT_MAX:.4f}   PRES_MAX : ±{PRES_MAX:.5f}")
    err = np.max(np.abs(_M_raw @ M_raw_inv_np.astype(np.float64) - np.eye(19)))
    print(f"M @ M_inv err: {err:.2e}")
