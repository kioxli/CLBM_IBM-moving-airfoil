"""
config_moving.py — Parameters for 3D CLBM-IBM simplified airplane model (D3Q19).

Airplane geometry (local body frame, pivot = reference = origin):
  - Main wing    : NACA 0012 cross-sections stacked in z, span 2*b_half
  - Fuselage     : elongated axis-aligned ellipsoid centred at pivot
  - H-stabilizer : NACA 0012 cross-sections, smaller, offset aft by x_tail_arm
  - V-stabilizer : NACA 0012 profiles stacked in LOCAL y (vertical),
                   offset aft by x_tail_arm; z-coord = NACA thickness + z0
  - Wing tips    : filled interior markers at z = z0 +/- b_half
  - H-tail tips  : filled interior markers at z = z0 +/- b_half_htail
  - V-tail tip   : filled interior markers at LOCAL y = fuse_ay + h_vtail

Transformation from body frame to world frame (applied each step by IBM):
    x_world = xc + X0*cos(theta) - Y0*sin(theta)
    y_world = yc + X0*sin(theta) + Y0*cos(theta)
    z_world = Z0   (absolute z, invariant)

Prescribed heave + pitch motion in x-y plane; z is always rigid (periodic).
Pure Python / NumPy only — no Taichi imports here.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Fluid domain
# ---------------------------------------------------------------------------
Nx, Ny = 600, 200
Nz     = 160          # spanwise lattice (periodic BC)

U_inf = 0.05          # free-stream velocity (lattice units / step)
Re    = 200           # Reynolds number based on chord

# ---------------------------------------------------------------------------
# CLBM D3Q19 parameters
# ---------------------------------------------------------------------------
cs2  = 1.0 / 3.0
nu   = U_inf * 80 / Re        # kinematic viscosity  (chord c = 80 lu)
tau  = 0.5 + nu / cs2         # shear relaxation time
s_v  = 1.0 / tau              # shear rate
s_e  = 1.0                    # bulk / energy mode
s_q  = 1.0                    # heat-flux modes
s_pi = 1.0                    # high-order modes

# ---------------------------------------------------------------------------
# D3Q19 lattice constants
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
# 19x19 M_raw matrix  (row = moment (p,q,r), col = velocity direction)
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
M_raw_inv_np = np.linalg.inv(_M_raw).astype(np.float32)   # shape (19, 19)

# ---------------------------------------------------------------------------
# Airplane geometry — all lengths in lattice units (lu)
# ---------------------------------------------------------------------------
c        = 80           # main wing chord length (lu)
b_half   = 60           # main wing half-span (lu); total span = 120 lu, AR = 1.5
ds_IBM   = 0.7          # Lagrangian marker spacing (lu)
AoA0_deg = 8.0          # mean angle-of-attack (degrees)

# Pivot (reference point) initial world position
x0  = 160.0
y0  = float(Ny) / 2.0   # = 100.0
z0  = float(Nz) / 2.0   # = 80.0  (mid-span of domain)

# Fuselage: elongated ellipsoid (airplane-style, not drone)
fuse_ax = 75.0   # half-length along chord direction (x) — 150 lu total
fuse_ay = 10.0   # half-height (y, vertical)
fuse_az = 10.0   # half-width  (z, spanwise)

# Tail geometry
x_tail_arm   = 80.0   # distance: pivot -> tail quarter-chord (local x)
c_htail      = 36.0   # horizontal stabilizer chord (lu)
b_half_htail = 32.0   # horizontal stabilizer half-span (lu)
c_vtail      = 36.0   # vertical stabilizer chord (lu)
h_vtail      = 28.0   # vertical stabilizer height (lu, above fuselage top)


# ---------------------------------------------------------------------------
# Geometry helper functions
# ---------------------------------------------------------------------------

def _naca0012_2d(chord, ds):
    """
    NACA 0012 surface markers for a single 2D cross-section.
    Returns X_local, Y_local in the chord-centred frame (x in [-c/2, c/2]).
    """
    t_max = 0.12
    N_est = max(int(np.ceil(np.pi * chord / ds)), 64)
    if N_est % 2 != 0:
        N_est += 1
    theta    = np.linspace(0.0, 2.0 * np.pi, N_est + 1)[:-1]
    xc_local = 0.5 * (1.0 - np.cos(theta))          # cosine-spaced in [0, 1]
    yt       = (5.0 * t_max
                * (0.2969 * np.sqrt(xc_local)
                   - 0.1260 * xc_local
                   - 0.3516 * xc_local**2
                   + 0.2843 * xc_local**3
                   - 0.1015 * xc_local**4))
    sign     = np.where(theta <= np.pi, 1.0, -1.0)
    X = ((xc_local - 0.5) * chord).astype(np.float32)
    Y = (sign * yt * chord).astype(np.float32)
    return X, Y


def _naca0012_interior(chord, ds):
    """
    Fill the NACA 0012 interior with grid markers (used for tip caps).
    Returns X_local, Y_local arrays (float32).
    """
    t_max = 0.12
    x_arr = np.arange(-chord / 2.0 + ds * 0.5, chord / 2.0, ds)
    all_x, all_y = [], []
    for xi in x_arr:
        x_unit = float(xi) / chord + 0.5           # normalised chord [0, 1]
        x_unit = max(x_unit, 1e-6)
        yt = (chord * 5.0 * t_max
              * (0.2969 * np.sqrt(x_unit)
                 - 0.1260 * x_unit
                 - 0.3516 * x_unit**2
                 + 0.2843 * x_unit**3
                 - 0.1015 * x_unit**4))
        if yt < ds * 0.5:
            continue
        y_arr = np.arange(-yt + ds * 0.5, yt, ds)
        for yi in y_arr:
            all_x.append(xi)
            all_y.append(yi)
    return (np.array(all_x, dtype=np.float32),
            np.array(all_y, dtype=np.float32))


def _ellipsoid_markers(ax, ay, az, ds):
    """
    Surface markers on an axis-aligned ellipsoid centred at the origin.
    Semi-axes ax (x), ay (y), az (z).  Spacing ~= ds.
    Returns X_local, Y_local, Z_local (float32) in the LOCAL body frame.
    """
    N_z_bands = max(int(np.ceil(np.pi * az / ds)), 8)
    all_x, all_y, all_z = [], [], []

    for i in range(N_z_bands + 1):
        phi     = np.pi * i / N_z_bands        # 0 = +z pole, pi = -z pole
        z_local = az * np.cos(phi)
        frac    = np.sin(phi)

        if frac < 1e-8:                         # pole: single point
            all_x.append(0.0)
            all_y.append(0.0)
            all_z.append(float(z_local))
            continue

        a_ring = ax * frac
        b_ring = ay * frac
        # Ramanujan approximation for ellipse circumference
        h      = ((a_ring - b_ring) / (a_ring + b_ring)) ** 2
        circ   = (np.pi * (a_ring + b_ring)
                  * (1.0 + 3.0 * h / (10.0 + np.sqrt(4.0 - 3.0 * h))))
        N_th   = max(int(np.ceil(circ / ds)), 4)
        theta_arr = np.linspace(0.0, 2.0 * np.pi, N_th, endpoint=False)
        for theta in theta_arr:
            all_x.append(ax * frac * np.cos(theta))
            all_y.append(ay * frac * np.sin(theta))
            all_z.append(float(z_local))

    return (np.array(all_x, dtype=np.float32),
            np.array(all_y, dtype=np.float32),
            np.array(all_z, dtype=np.float32))


# ---------------------------------------------------------------------------
# Generate airplane Lagrangian markers
# ---------------------------------------------------------------------------
ds_z = 1.0   # spanwise / vertical layer spacing

# --- 1. Main wing: NACA 0012 cross-sections stacked in z ---
X_template, Y_template = _naca0012_2d(c, ds_IBM)
N_MARKERS_2D = len(X_template)

z_wing_start  = z0 - b_half                              # = 20.0
z_wing_end    = z0 + b_half                              # = 140.0
z_wing_layers = np.arange(z_wing_start,
                           z_wing_end + ds_z * 0.5,
                           ds_z, dtype=np.float32)       # 121 layers
Nz_layers     = len(z_wing_layers)

X_wing = np.tile(X_template, Nz_layers).astype(np.float32)
Y_wing = np.tile(Y_template, Nz_layers).astype(np.float32)
Z_wing = np.repeat(z_wing_layers, N_MARKERS_2D).astype(np.float32)

# --- 2. Main wing-tip caps (filled interior at both z ends) ---
X_tip_int, Y_tip_int = _naca0012_interior(c, ds_IBM)
N_TIP_INT = len(X_tip_int)

X_tip = np.tile(X_tip_int, 2).astype(np.float32)
Y_tip = np.tile(Y_tip_int, 2).astype(np.float32)
Z_tip = np.concatenate([
    np.full(N_TIP_INT, z_wing_start, dtype=np.float32),
    np.full(N_TIP_INT, z_wing_end,   dtype=np.float32),
])

# --- 3. Fuselage: elongated ellipsoid ---
X_fuse_loc, Y_fuse_loc, Z_fuse_loc = _ellipsoid_markers(
    fuse_ax, fuse_ay, fuse_az, ds_IBM)
X_fuse = X_fuse_loc.copy()
Y_fuse = Y_fuse_loc.copy()
Z_fuse = (z0 + Z_fuse_loc).astype(np.float32)   # local z -> absolute z

# --- 4. Horizontal stabilizer: NACA 0012 cross-sections, offset aft ---
#     X0 = chord + x_tail_arm  (pitches rigidly with the body)
#     Y0 = NACA thickness (local y, pitches with body)
#     Z0 = absolute z (invariant)
X_htail_tmpl, Y_htail_tmpl = _naca0012_2d(c_htail, ds_IBM)
N_HTAIL_2D = len(X_htail_tmpl)

z_htail_start  = z0 - b_half_htail   # = 48.0
z_htail_end    = z0 + b_half_htail   # = 112.0
z_htail_layers = np.arange(z_htail_start,
                             z_htail_end + ds_z * 0.5,
                             ds_z, dtype=np.float32)
Nz_htail_layers = len(z_htail_layers)

X_htail = np.tile(X_htail_tmpl + x_tail_arm, Nz_htail_layers).astype(np.float32)
Y_htail = np.tile(Y_htail_tmpl, Nz_htail_layers).astype(np.float32)
Z_htail = np.repeat(z_htail_layers, N_HTAIL_2D).astype(np.float32)

# --- 5. Horizontal stabilizer tip caps ---
X_htail_int, Y_htail_int = _naca0012_interior(c_htail, ds_IBM)
N_HTAIL_INT = len(X_htail_int)

X_htail_tip = np.tile(X_htail_int + x_tail_arm, 2).astype(np.float32)
Y_htail_tip = np.tile(Y_htail_int, 2).astype(np.float32)
Z_htail_tip = np.concatenate([
    np.full(N_HTAIL_INT, z_htail_start, dtype=np.float32),
    np.full(N_HTAIL_INT, z_htail_end,   dtype=np.float32),
])

# --- 6. Vertical stabilizer (fin): NACA profiles stacked in LOCAL y ---
#     Each y-slice is a NACA profile in the x-z plane:
#       X0 = chord direction (local x, offset aft by x_tail_arm)
#       Y0 = fin height (LOCAL y, from fuse_ay to fuse_ay+h_vtail)
#            rotates with pitch — the fin pitches with the body
#       Z0 = NACA thickness mapped to z, centred at z0 (absolute)
X_vtail_tmpl, Y_vtail_tmpl = _naca0012_2d(c_vtail, ds_IBM)
N_VTAIL_2D = len(X_vtail_tmpl)

y_vtail_start = fuse_ay             # LOCAL y root  (top of fuselage)
y_vtail_end   = fuse_ay + h_vtail  # LOCAL y tip
y_vtail_layers = np.arange(y_vtail_start,
                             y_vtail_end + ds_z * 0.5,
                             ds_z, dtype=np.float32)
Ny_vtail_layers = len(y_vtail_layers)

X_vtail = np.tile(X_vtail_tmpl + x_tail_arm, Ny_vtail_layers).astype(np.float32)
Y_vtail = np.repeat(y_vtail_layers, N_VTAIL_2D).astype(np.float32)   # LOCAL y
Z_vtail = np.tile(Y_vtail_tmpl + z0, Ny_vtail_layers).astype(np.float32)  # abs z

# --- 7. Vertical stabilizer fin-tip cap (filled interior at LOCAL y tip) ---
X_vtip_int, Y_vtip_int = _naca0012_interior(c_vtail, ds_IBM)
N_VTIP_INT = len(X_vtip_int)

X_vtail_tip = (X_vtip_int + x_tail_arm).astype(np.float32)
Y_vtail_tip = np.full(N_VTIP_INT, y_vtail_end, dtype=np.float32)   # LOCAL y
Z_vtail_tip = (Y_vtip_int + z0).astype(np.float32)                 # abs z

# --- 8. Combine all components ---
X_marker_np = np.concatenate([
    X_wing, X_tip, X_fuse,
    X_htail, X_htail_tip,
    X_vtail, X_vtail_tip,
]).astype(np.float32)

Y_marker_np = np.concatenate([
    Y_wing, Y_tip, Y_fuse,
    Y_htail, Y_htail_tip,
    Y_vtail, Y_vtail_tip,
]).astype(np.float32)

Z_marker_np = np.concatenate([
    Z_wing, Z_tip, Z_fuse,
    Z_htail, Z_htail_tip,
    Z_vtail, Z_vtail_tip,
]).astype(np.float32)

N_MARKERS = len(X_marker_np)

# ---------------------------------------------------------------------------
# Prescribed motion parameters
# ---------------------------------------------------------------------------
k_reduced = 0.5
f_motion  = k_reduced * U_inf / (np.pi * c)

A_heave   = 10.0
A_pitch   = np.radians(5.0)
phi_pitch = np.pi / 2.0
AoA0_rad  = np.radians(AoA0_deg)

# ---------------------------------------------------------------------------
# Time stepping
# ---------------------------------------------------------------------------
N_steps      = 20000
vis_interval = 200

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
DATA_DIR   = "drone_frames"           # 逐帧 3D 流场快照输出目录
DATA_FILE  = "drone_snapshots.npz"    # (保留兼容) 旧版 2D 切片快照文件名
VIDEO_FILE = "clbm_drone_moving_3d.mp4"
VIDEO_FPS  = 10


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Grid             : {Nx} x {Ny} x {Nz}")
    print(f"Re               : {Re}   U_inf : {U_inf}   c : {c} lu")
    print(f"nu               : {nu:.6f}   tau : {tau:.4f}   s_v : {s_v:.4f}")
    print(f"AoA0             : {AoA0_deg} deg   pivot : ({x0}, {y0}, {z0})")
    print(f"")
    print(f"--- Geometry ---")
    print(f"Main wing        : chord={c} lu  half-span={b_half} lu  AR={2*b_half/c:.2f}")
    print(f"Fuselage         : ax={fuse_ax}  ay={fuse_ay}  az={fuse_az} lu  (150 lu long)")
    print(f"H-stabilizer     : chord={c_htail} lu  half-span={b_half_htail} lu  arm={x_tail_arm} lu")
    print(f"V-stabilizer     : chord={c_vtail} lu  height={h_vtail} lu  arm={x_tail_arm} lu")
    print(f"")
    print(f"--- Markers ---")
    print(f"Wing (surface)   : {len(X_wing)}  ({Nz_layers} z-layers x {N_MARKERS_2D} pts)")
    print(f"Wing tip caps    : {len(X_tip)}  ({N_TIP_INT} per tip)")
    print(f"Fuselage         : {len(X_fuse)}")
    print(f"H-tail (surface) : {len(X_htail)}  ({Nz_htail_layers} z-layers x {N_HTAIL_2D} pts)")
    print(f"H-tail tip caps  : {len(X_htail_tip)}  ({N_HTAIL_INT} per tip)")
    print(f"V-tail (surface) : {len(X_vtail)}  ({Ny_vtail_layers} y-layers x {N_VTAIL_2D} pts)")
    print(f"V-tail tip cap   : {len(X_vtail_tip)}")
    print(f"N_MARKERS total  : {N_MARKERS}")
    print(f"")
    print(f"--- Motion ---")
    print(f"f_motion         : {f_motion:.6e} step^-1   period : {1.0/f_motion:.1f} steps")
    print(f"A_heave          : {A_heave} lu   A_pitch : {np.degrees(A_pitch):.1f} deg")
    print(f"")
    err = np.max(np.abs(_M_raw @ M_raw_inv_np.astype(np.float64) - np.eye(19)))
    print(f"M @ M_inv err    : {err:.2e}  (should be ~1e-14)")
    print(f"X_template       : [{X_template.min():.1f}, {X_template.max():.1f}]")
    print(f"Y_template       : [{Y_template.min():.1f}, {Y_template.max():.1f}]")
