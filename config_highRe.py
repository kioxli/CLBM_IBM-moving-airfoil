"""
config_highRe.py -- High-Re configuration for 3D CLBM-IBM simplified airplane (D3Q19).

Compared to config_moving.py (Re=200, c=80 lu):
  * Reynolds number raised to Re = 1000
  * Chord length c = 120 lu  (1.5x, improves boundary-layer resolution)
  * Grid enlarged to Nx=800 x Ny=260 x Nz=220  (~2.3x total cell count)
  * All geometric lengths scaled by the same factor 1.5
  * kinematic viscosity nu = U_inf * c / Re = 0.006
  * relaxation time        tau = 0.5 + nu/cs2 = 0.518
    (MRT / cascade scheme tolerates tau < 0.55 better than plain BGK)

Physics notes
-------------
At Re=1000 the flow past the airfoil becomes unsteady (Karman vortex shedding
in the wake, leading-edge separation at high AoA).  The MRT cascade LBM used
here is more stable than single-relaxation-time LBM at low tau, but tau=0.518
is still demanding -- keep Ma = U_inf <= 0.05 and watch for divergence signals
(rho_min < 0.5 triggers a print warning in main_moving.py).

How to run the high-Re simulation
----------------------------------
  python main_highRe.py

The solver classes (CLBMSolver, DroneMovingIBM) are reused unchanged via a
sys.modules alias in main_highRe.py, so this file needs no extra solver code.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Fluid domain
# ---------------------------------------------------------------------------
Nx, Ny = 800, 260
Nz     = 220          # spanwise lattice (periodic BC)

U_inf = 0.05          # free-stream velocity (lattice units / step)
Re    = 1000          # Reynolds number based on chord

# ---------------------------------------------------------------------------
# CLBM D3Q19 parameters
# ---------------------------------------------------------------------------
cs2  = 1.0 / 3.0
nu   = U_inf * 120 / Re      # kinematic viscosity  (chord c = 120 lu)
tau  = 0.5 + nu / cs2        # shear relaxation time   -> 0.518
s_v  = 1.0 / tau             # shear rate
s_e  = 1.0                   # bulk / energy mode
s_q  = 1.0                   # heat-flux modes
s_pi = 1.0                   # high-order modes

# ---------------------------------------------------------------------------
# D3Q19 lattice constants   (identical to config_moving.py)
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
# Airplane geometry -- all lengths in lattice units (lu)
# Scale factor = 1.5 relative to config_moving.py (c: 80 -> 120)
# ---------------------------------------------------------------------------
c        = 120          # main wing chord length (lu)
b_half   = 90           # main wing half-span (lu); total span = 180 lu, AR = 1.5
ds_IBM   = 0.7          # Lagrangian marker spacing (lu)
AoA0_deg = 8.0          # mean angle-of-attack (degrees)

# Pivot (reference point) initial world position
x0  = 200.0
y0  = float(Ny) / 2.0   # = 130.0
z0  = float(Nz) / 2.0   # = 110.0  (mid-span of domain)

# Fuselage: elongated ellipsoid (scaled 1.5x from Re=200 case)
fuse_ax = 112.5  # half-length along chord direction (x) -- 225 lu total
fuse_ay =  15.0  # half-height (y, vertical)
fuse_az =  15.0  # half-width  (z, spanwise)

# Tail geometry (scaled 1.5x)
x_tail_arm   = 120.0   # distance: pivot -> tail quarter-chord (local x)
c_htail      =  54.0   # horizontal stabilizer chord (lu)
b_half_htail =  48.0   # horizontal stabilizer half-span (lu)
c_vtail      =  54.0   # vertical stabilizer chord (lu)
h_vtail      =  42.0   # vertical stabilizer height (lu, above fuselage top)


# ---------------------------------------------------------------------------
# Geometry helper functions  (identical algorithms, different scale)
# ---------------------------------------------------------------------------

def _naca0012_2d(chord, ds):
    t_max = 0.12
    N_est = max(int(np.ceil(np.pi * chord / ds)), 64)
    if N_est % 2 != 0:
        N_est += 1
    theta    = np.linspace(0.0, 2.0 * np.pi, N_est + 1)[:-1]
    xc_local = 0.5 * (1.0 - np.cos(theta))
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
    t_max = 0.12
    x_arr = np.arange(-chord / 2.0 + ds * 0.5, chord / 2.0, ds)
    all_x, all_y = [], []
    for xi in x_arr:
        x_unit = float(xi) / chord + 0.5
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
    N_z_bands = max(int(np.ceil(np.pi * az / ds)), 8)
    all_x, all_y, all_z = [], [], []
    for i in range(N_z_bands + 1):
        phi     = np.pi * i / N_z_bands
        z_local = az * np.cos(phi)
        frac    = np.sin(phi)
        if frac < 1e-8:
            all_x.append(0.0); all_y.append(0.0); all_z.append(float(z_local))
            continue
        a_ring = ax * frac
        b_ring = ay * frac
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

# --- 1. Main wing ---
X_template, Y_template = _naca0012_2d(c, ds_IBM)
N_MARKERS_2D = len(X_template)

z_wing_start  = z0 - b_half                 # = 20.0
z_wing_end    = z0 + b_half                 # = 200.0
z_wing_layers = np.arange(z_wing_start, z_wing_end + ds_z * 0.5,
                            ds_z, dtype=np.float32)
Nz_layers     = len(z_wing_layers)

X_wing = np.tile(X_template, Nz_layers).astype(np.float32)
Y_wing = np.tile(Y_template, Nz_layers).astype(np.float32)
Z_wing = np.repeat(z_wing_layers, N_MARKERS_2D).astype(np.float32)

# --- 2. Main wing-tip caps ---
X_tip_int, Y_tip_int = _naca0012_interior(c, ds_IBM)
N_TIP_INT = len(X_tip_int)

X_tip = np.tile(X_tip_int, 2).astype(np.float32)
Y_tip = np.tile(Y_tip_int, 2).astype(np.float32)
Z_tip = np.concatenate([
    np.full(N_TIP_INT, z_wing_start, dtype=np.float32),
    np.full(N_TIP_INT, z_wing_end,   dtype=np.float32),
])

# --- 3. Fuselage ---
X_fuse_loc, Y_fuse_loc, Z_fuse_loc = _ellipsoid_markers(
    fuse_ax, fuse_ay, fuse_az, ds_IBM)
X_fuse = X_fuse_loc.copy()
Y_fuse = Y_fuse_loc.copy()
Z_fuse = (z0 + Z_fuse_loc).astype(np.float32)

# --- 4. Horizontal stabiliser ---
X_htail_tmpl, Y_htail_tmpl = _naca0012_2d(c_htail, ds_IBM)
N_HTAIL_2D = len(X_htail_tmpl)

z_htail_start  = z0 - b_half_htail   # = 62.0
z_htail_end    = z0 + b_half_htail   # = 158.0
z_htail_layers = np.arange(z_htail_start, z_htail_end + ds_z * 0.5,
                              ds_z, dtype=np.float32)
Nz_htail_layers = len(z_htail_layers)

X_htail = np.tile(X_htail_tmpl + x_tail_arm, Nz_htail_layers).astype(np.float32)
Y_htail = np.tile(Y_htail_tmpl, Nz_htail_layers).astype(np.float32)
Z_htail = np.repeat(z_htail_layers, N_HTAIL_2D).astype(np.float32)

# --- 5. Horizontal stabiliser tip caps ---
X_htail_int, Y_htail_int = _naca0012_interior(c_htail, ds_IBM)
N_HTAIL_INT = len(X_htail_int)

X_htail_tip = np.tile(X_htail_int + x_tail_arm, 2).astype(np.float32)
Y_htail_tip = np.tile(Y_htail_int, 2).astype(np.float32)
Z_htail_tip = np.concatenate([
    np.full(N_HTAIL_INT, z_htail_start, dtype=np.float32),
    np.full(N_HTAIL_INT, z_htail_end,   dtype=np.float32),
])

# --- 6. Vertical stabiliser ---
X_vtail_tmpl, Y_vtail_tmpl = _naca0012_2d(c_vtail, ds_IBM)
N_VTAIL_2D = len(X_vtail_tmpl)

y_vtail_start  = fuse_ay
y_vtail_end    = fuse_ay + h_vtail
y_vtail_layers = np.arange(y_vtail_start, y_vtail_end + ds_z * 0.5,
                              ds_z, dtype=np.float32)
Ny_vtail_layers = len(y_vtail_layers)

X_vtail = np.tile(X_vtail_tmpl + x_tail_arm, Ny_vtail_layers).astype(np.float32)
Y_vtail = np.repeat(y_vtail_layers, N_VTAIL_2D).astype(np.float32)
Z_vtail = np.tile(Y_vtail_tmpl + z0, Ny_vtail_layers).astype(np.float32)

# --- 7. Vertical stabiliser tip cap ---
X_vtip_int, Y_vtip_int = _naca0012_interior(c_vtail, ds_IBM)
N_VTIP_INT = len(X_vtip_int)

X_vtail_tip = (X_vtip_int + x_tail_arm).astype(np.float32)
Y_vtail_tip = np.full(N_VTIP_INT, y_vtail_end, dtype=np.float32)
Z_vtail_tip = (Y_vtip_int + z0).astype(np.float32)

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
# Prescribed motion parameters  (same reduced frequency k=0.5)
# ---------------------------------------------------------------------------
k_reduced = 0.5
f_motion  = k_reduced * U_inf / (np.pi * c)   # -> ~6.6e-5 step^-1

A_heave   = 15.0                               # scaled 1.5x: 10 -> 15 lu
A_pitch   = np.radians(5.0)
phi_pitch = np.pi / 2.0
AoA0_rad  = np.radians(AoA0_deg)

# ---------------------------------------------------------------------------
# Time stepping
# ---------------------------------------------------------------------------
# One full motion cycle at Re=1000, c=120:
#   T_period = 1/f_motion = pi*c / (k*U_inf) = pi*120 / (0.5*0.05) ~ 15080 steps
# N_steps = 30000 covers ~2 full cycles; use 20000 for a quick run.
N_steps      = 20000
vis_interval = 200

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
DATA_DIR   = "drone_frames_highRe"
DATA_FILE  = "drone_highRe_snapshots.npz"
VIDEO_FILE = "clbm_highRe_3d.mp4"
VIDEO_FPS  = 10


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Grid             : {Nx} x {Ny} x {Nz}")
    print(f"Re               : {Re}   U_inf : {U_inf}   c : {c} lu")
    print(f"nu               : {nu:.6f}   tau : {tau:.4f}   s_v : {s_v:.4f}")
    print(f"WARNING          : tau={tau:.3f} is low; MRT cascade improves stability")
    print(f"AoA0             : {AoA0_deg} deg   pivot : ({x0}, {y0}, {z0})")
    print(f"")
    print(f"--- Geometry (1.5x scaled from Re=200 config) ---")
    print(f"Main wing        : chord={c} lu  half-span={b_half} lu  AR={2*b_half/c:.2f}")
    print(f"Fuselage         : ax={fuse_ax}  ay={fuse_ay}  az={fuse_az} lu")
    print(f"H-stabilizer     : chord={c_htail} lu  half-span={b_half_htail} lu  arm={x_tail_arm} lu")
    print(f"V-stabilizer     : chord={c_vtail} lu  height={h_vtail} lu  arm={x_tail_arm} lu")
    print(f"z_wing           : [{z_wing_start}, {z_wing_end}]  (Nz={Nz}, margins: {z_wing_start:.0f} lu)")
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
    print(f"f_motion         : {f_motion:.6e} step^-1   period : {1.0/f_motion:.0f} steps")
    print(f"A_heave          : {A_heave} lu   A_pitch : {np.degrees(A_pitch):.1f} deg")
    print(f"")
    err = np.max(np.abs(_M_raw @ M_raw_inv_np.astype(np.float64) - np.eye(19)))
    print(f"M @ M_inv err    : {err:.2e}  (should be ~1e-14)")
    print(f"")
    print(f"--- Memory estimate ---")
    cells = Nx * Ny * Nz
    mem_f32_GB = cells * 19 * 2 * 4 / 1e9   # double-buffer, float32
    print(f"Grid cells       : {cells/1e6:.1f} M  (vs {600*200*160/1e6:.1f} M for Re=200)")
    print(f"f-field VRAM est : {mem_f32_GB:.2f} GB  (double buffer, float32)")
