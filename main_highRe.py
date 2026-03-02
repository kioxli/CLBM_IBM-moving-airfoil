"""
main_highRe.py -- High-Re 3D CLBM-IBM airplane simulation (Re=1000, c=120 lu).

Reuses the solver classes from clbm_moving.py and airfoil_moving.py unchanged,
by aliasing config_highRe as 'config_moving' in sys.modules before those
modules are imported.  This avoids duplicating ~600 lines of Taichi kernel code.

Grid    : 800 x 260 x 220  (~2.3x cells vs Re=200 case)
Re      : 1000  (chord-based)
nu      : 0.006  ->  tau = 0.518  (MRT cascade scheme for stability)
Chord   : 120 lu  (1.5x finer boundary-layer resolution)
Motion  : heave A=15 lu + pitch A=5 deg  (k_red=0.5, same reduced frequency)

Run:
  python main_highRe.py
"""

import sys
import os

# ---------------------------------------------------------------------------
# CRITICAL: alias config_highRe as 'config_moving' BEFORE any solver import.
# clbm_moving.py and airfoil_moving.py both do  "from config_moving import ..."
# They will find this alias and pick up the high-Re parameters transparently.
# ---------------------------------------------------------------------------
import config_highRe as _cfg_highRe
sys.modules["config_moving"] = _cfg_highRe

# ---------------------------------------------------------------------------
# Taichi initialisation  -- MUST precede any ti.field allocation
# ---------------------------------------------------------------------------
import taichi as ti

try:
    ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32, fast_math=True)
    print("[Taichi] GPU mode")
except Exception as _e:
    ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32,
            cpu_max_num_threads=8)
    print(f"[Taichi] CPU fallback ({_e})")

# ---------------------------------------------------------------------------
# Project imports -- after ti.init and sys.modules patch
# ---------------------------------------------------------------------------
import math
import time
import numpy as np

from config_highRe import (
    Nx, Ny, Nz, U_inf, Re,
    x0, y0, z0,
    f_motion, A_heave, A_pitch, phi_pitch, AoA0_rad,
    fuse_ax, fuse_ay, fuse_az,
    b_half, z_wing_start, z_wing_end,
    x_tail_arm, c_htail, b_half_htail,
    z_htail_start, z_htail_end,
    c_vtail, h_vtail,
    X_template, Y_template,
    X_htail_tmpl, Y_htail_tmpl,
    N_MARKERS_2D, N_MARKERS,
    N_steps, vis_interval,
    DATA_DIR,
    nu, tau, c,
)

# Solver classes import config_moving at module level.
# Since sys.modules['config_moving'] already points to config_highRe,
# the kernels will be compiled with the high-Re grid / viscosity values.
from clbm_moving    import CLBMSolver
from airfoil_moving import DroneMovingIBM


# ---------------------------------------------------------------------------
# Prescribed motion kinematics
# ---------------------------------------------------------------------------
_W_MOTION = 2.0 * math.pi * f_motion


def _motion_state(step: int):
    t     = float(step)
    w     = _W_MOTION
    xc    = x0
    yc    = y0 + A_heave * math.sin(w * t)
    theta = AoA0_rad + A_pitch * math.sin(w * t + phi_pitch)
    vx    = 0.0
    vy    = A_heave * w * math.cos(w * t)
    omega = A_pitch * w * math.cos(w * t + phi_pitch)
    return xc, yc, theta, vx, vy, omega


# ---------------------------------------------------------------------------
# Slice indices (for metadata)
# ---------------------------------------------------------------------------
_IZ = int(round(z0))
_IX = int(round(x0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("3D CLBM-IBM  High-Re Airplane  (Re=1000, c=120 lu, D3Q19)")
    print("=" * 65)
    print(f"  Grid       : {Nx} x {Ny} x {Nz}")
    print(f"  Re={Re}  U_inf={U_inf}  chord={c} lu")
    print(f"  nu={nu:.6f}  tau={tau:.4f}  (MRT cascade, low-tau regime)")
    print(f"  Wing       : span={2*b_half} lu  AR={2*b_half/c:.2f}")
    print(f"  Fuselage   : {2*fuse_ax:.0f}x{2*fuse_ay:.0f}x{2*fuse_az:.0f} lu")
    print(f"  H-stab     : chord={c_htail} lu  span={2*b_half_htail} lu  arm={x_tail_arm} lu")
    print(f"  V-fin      : chord={c_vtail} lu  height={h_vtail} lu")
    print(f"  N_MARKERS  : {N_MARKERS}")
    print(f"  N_steps    : {N_steps}   vis every {vis_interval} steps"
          f"  -> {N_steps // vis_interval} snapshots")
    period = 1.0 / f_motion
    print(f"  T_period   : {period:.0f} steps  ({N_steps / period:.1f} cycles total)")
    print(f"  DATA_DIR   : {DATA_DIR}")
    print()

    os.makedirs(DATA_DIR, exist_ok=True)

    lbm = CLBMSolver()
    ibm = DroneMovingIBM()

    snap_xc    = []
    snap_yc    = []
    snap_theta = []
    snap_step  = []
    frame_idx  = 0

    t0   = time.perf_counter()
    last = t0

    for step in range(1, N_steps + 1):

        xc, yc, theta, vx, vy, omega = _motion_state(step)

        fx, fy, fz = ibm.compute_force(lbm.ux, lbm.uy,
                                        xc, yc, theta, vx, vy, omega)
        lbm.step(fx, fy, fz)

        if step % vis_interval == 0:
            rho_np = lbm.rho.to_numpy()
            ux_np  = lbm.ux.to_numpy()
            uy_np  = lbm.uy.to_numpy()
            uz_np  = lbm.uz.to_numpy()

            rho_min = float(rho_np.min())
            rho_max = float(rho_np.max())
            u_max   = float(np.sqrt(ux_np**2 + uy_np**2 + uz_np**2).max())

            now  = time.perf_counter()
            rate = vis_interval / (now - last)
            last = now

            print(f"  step {step:6d}  |  u_max={u_max:.5f}  "
                  f"rho=[{rho_min:.4f},{rho_max:.4f}]  "
                  f"yc={yc:.2f}  theta={math.degrees(theta):.2f} deg"
                  f"  |  {rate:.0f} steps/s")

            if not np.isfinite(rho_np).all() or rho_min < 0.3:
                print(f"\n*** Diverged at step {step}!  rho_min={rho_min:.4f} ***")
                print("    Consider reducing U_inf or increasing c for higher tau.")
                break

            frame_idx += 1
            frame_path = os.path.join(DATA_DIR, f"frame_{frame_idx:06d}.npz")
            np.savez_compressed(
                frame_path,
                ux  = ux_np.astype(np.float32),
                uy  = uy_np.astype(np.float32),
                uz  = uz_np.astype(np.float32),
                rho = rho_np.astype(np.float32),
            )
            print(f"    -> saved {frame_path}")

            snap_xc.append(xc)
            snap_yc.append(yc)
            snap_theta.append(theta)
            snap_step.append(step)

    elapsed = time.perf_counter() - t0
    print(f"\nDone. {step} steps in {elapsed:.1f} s  ({step / elapsed:.0f} steps/s).")

    n_snap = len(snap_step)
    if n_snap == 0:
        print("No snapshots captured.")
        return

    meta_path = os.path.join(DATA_DIR, "meta.npz")
    print(f"Saving metadata ({n_snap} frames) -> {meta_path} ...")
    np.savez_compressed(
        meta_path,
        xc     = np.array(snap_xc,    dtype=np.float32),
        yc     = np.array(snap_yc,    dtype=np.float32),
        theta  = np.array(snap_theta, dtype=np.float32),
        step   = np.array(snap_step,  dtype=np.int32),
        X_template   = X_template,
        Y_template   = Y_template,
        X_htail_tmpl = X_htail_tmpl,
        Y_htail_tmpl = Y_htail_tmpl,
        grid_meta = np.array([
            Nx, Ny, Nz,                     # 0-2
            U_inf, Re,                      # 3-4
            x0, y0, z0,                     # 5-7
            fuse_ax, fuse_ay,               # 8-9
            b_half,                         # 10
            z_wing_start, z_wing_end,       # 11-12
            float(_IZ), float(_IX),         # 13-14
            vis_interval,                   # 15
            fuse_az,                        # 16
            x_tail_arm,                     # 17
            c_htail, b_half_htail,          # 18-19
            z_htail_start, z_htail_end,     # 20-21
            c_vtail, h_vtail,               # 22-23
        ], dtype=np.float64),
    )
    print(f"Saved: {meta_path}")
    print(f"Total {n_snap} full-field frames in {DATA_DIR}/")
    print("Run  `python render_2d.py --dir drone_frames_highRe`  to generate the 2D video.")
    print("Run  `python render_3d.py --dir drone_frames_highRe`  to generate the 3D video.")


if __name__ == "__main__":
    main()
