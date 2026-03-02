"""
main_fish.py  -- 3D CLBM-IBM bio-inspired fish swimmer  (3D video output)

Four output MP4 videos (one simulation run):
  fish_umag.mp4        -- 3D perspective  velocity magnitude   (inferno)
  fish_vorticity.mp4   -- 3D perspective  z-vorticity omega_z  (dark diverging)
  fish_pressure.mp4    -- 3D perspective  pressure deviation   (dark diverging)
  fish_composite.mp4   -- 2x2: 3D-vort | 3D-vel / 2D-midspan | 3D-pres

Run:  python main_fish.py
"""

import sys
import math
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import imageio

# ---------------------------------------------------------------------------
# Taichi initialisation -- MUST precede any ti.field allocation
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
# Project imports -- after ti.init
# ---------------------------------------------------------------------------
from config_fish import (
    Nx, Ny, Nz, U_inf, Re, AoA0_deg,
    c as CHORD,
    x0, y0, f_motion, k_reduced,
    A_heave, A_pitch, phi_pitch, AoA0_rad,
    FISH_THICKNESS,
    N_MARKERS_2D, N_MARKERS,
    X_template, Y_template,
    N_steps, vis_interval,
    VIDEO_FPS, VIDEO_UMAG, VIDEO_VORT, VIDEO_PRES, VIDEO_COMP,
    VORT_MAX, PRES_MAX, cs2,
)
from clbm_fish import CLBMSolver
from ibm_fish  import FishIBM

# ---------------------------------------------------------------------------
# Motion kinematics (analytic)
# ---------------------------------------------------------------------------
_W = 2.0 * math.pi * f_motion


def _motion_state(step: int):
    t     = float(step)
    w     = _W
    xc    = x0
    yc    = y0 + A_heave * math.sin(w * t)
    theta = AoA0_rad + A_pitch * math.sin(w * t + phi_pitch)
    vx    = 0.0
    vy    = A_heave * w * math.cos(w * t)
    omega = A_pitch * w * math.cos(w * t + phi_pitch)
    return xc, yc, theta, vx, vy, omega


# ---------------------------------------------------------------------------
# Custom colormaps -- dark-theme, SIGGRAPH aesthetic
# ---------------------------------------------------------------------------
# Vorticity: deep-space bg, electric-blue CW / crimson-amber CCW, near-black zero
_VORT_CMAP = LinearSegmentedColormap.from_list(
    "vort_dark",
    [
        (0.00, "#0a1060"),   # deep navy  (max CW)
        (0.28, "#2060ff"),   # electric blue
        (0.46, "#060618"),   # near-black (zero)
        (0.54, "#060618"),   # near-black (zero)
        (0.72, "#ff2020"),   # crimson red
        (1.00, "#ffcc00"),   # golden amber  (max CCW)
    ],
    N=256,
)

# Pressure: cool-blue low / warm-orange high, dark neutral
_PRES_CMAP = LinearSegmentedColormap.from_list(
    "pres_dark",
    [
        (0.00, "#0a0a40"),   # deep indigo  (min pressure)
        (0.28, "#1040c0"),   # medium blue
        (0.46, "#0a0a18"),   # near-black (ambient)
        (0.54, "#0a0a18"),   # near-black (ambient)
        (0.72, "#c02010"),   # dark red
        (1.00, "#ff7700"),   # bright orange  (max pressure)
    ],
    N=256,
)

_UMAG_CMAP = "inferno"   # black->purple->red->orange->yellow (perceptual, dark bg)

# ---------------------------------------------------------------------------
# Visualization parameters
# ---------------------------------------------------------------------------
_BG_COLOR    = "#050510"     # deep-space background
_BODY_COLOR  = "#c8d8ff"     # pale blue-white body surface
_AXIS_COLOR  = "#1e1e3a"     # subtle dark pane edge / grid
_LABEL_COLOR = "#7080b0"     # axis label / tick colour

# 3D figure geometry  (9.6 in x 7.2 in @ 80 dpi  -->  768 x 576 px)
_STRIDE_3D  = 8              # spatial stride for 3D downsampling
_Z_SLICES   = [0, Nz // 4, Nz // 2, 3 * Nz // 4, Nz - 1]  # 5 z-planes
_ALPHA_3D   = 0.58           # per-slice transparency
_AZIM       = -38.0          # camera azimuth  (deg)
_ELEV       = 24.0           # camera elevation (deg)
_FIG_W_3D   = 9.6            # figure width  (in)
_FIG_H_3D   = 7.2            # figure height (in)
_DPI_3D     = 80

# 2D reference panel uses same pixel dimensions for composite compatibility
_DPI_2D     = _DPI_3D
_Z_MID      = Nz // 2        # mid-span slice index


# ---------------------------------------------------------------------------
# Body helpers
# ---------------------------------------------------------------------------
def _body_contour(xc, yc, theta):
    """2D body outline for the mid-span reference panel."""
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    Xo = xc + X_template * cos_t - Y_template * sin_t
    Yo = yc + X_template * sin_t + Y_template * cos_t
    return np.append(Xo, Xo[0]), np.append(Yo, Yo[0])


def _body_surface_3d(xc, yc, theta):
    """
    3D body surface as (Xb, Yb, Zb) meshgrids for plot_surface.
    Ellipse extruded uniformly along z.  Returns shapes (nzb, nth).
    """
    nth, nzb = 60, 20
    t_arr = np.linspace(0.0, 2.0 * np.pi, nth)
    z_arr = np.linspace(0.0, float(Nz - 1), nzb)
    TH, ZB = np.meshgrid(t_arr, z_arr)
    a   = CHORD / 2.0
    b   = FISH_THICKNESS / 2.0
    Xbl = a * np.cos(TH)
    Ybl = b * np.sin(TH)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    Xb = xc + Xbl * cos_t - Ybl * sin_t
    Yb = yc + Xbl * sin_t + Ybl * cos_t
    return Xb, Yb, ZB


# ---------------------------------------------------------------------------
# 3D perspective frame renderer
# ---------------------------------------------------------------------------
def _make_frame_3d(field_3d, cmap, vmin, vmax,
                   xc, yc, theta,
                   title: str = "") -> np.ndarray:
    """
    Render field_3d (Nx, Ny, Nz) as a 3D perspective figure.

    Stacks z-plane slices using plot_surface with per-quad facecolors,
    then overlays the 3D ellipse body.
    Returns uint8 RGB array shape (H, W, 3).
    """
    s        = _STRIDE_3D
    xg       = np.arange(0, Nx, s, dtype=np.float32)
    yg       = np.arange(0, Ny, s, dtype=np.float32)
    XX, YY   = np.meshgrid(xg, yg)          # (len(yg), len(xg))
    norm     = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    fig = plt.figure(figsize=(_FIG_W_3D, _FIG_H_3D), dpi=_DPI_3D)
    fig.patch.set_facecolor(_BG_COLOR)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(_BG_COLOR)

    # ---- stacked z-plane slices (facecolor-mapped) -------------------------
    for iz in _Z_SLICES:
        data_s = field_3d[::s, ::s, iz].T          # (len(yg), len(xg))
        fc     = cmap_obj(norm(data_s))             # RGBA
        fc     = fc.copy()
        fc[..., 3] = _ALPHA_3D
        ZZ = np.full_like(XX, float(iz))
        ax.plot_surface(XX, YY, ZZ,
                        facecolors=fc,
                        rstride=1, cstride=1,
                        linewidth=0, antialiased=False,
                        shade=False)

    # ---- 3D body (ellipse extruded in z) -----------------------------------
    Xb, Yb, Zb = _body_surface_3d(xc, yc, theta)
    ax.plot_surface(Xb, Yb, Zb,
                    color=_BODY_COLOR, alpha=0.90,
                    linewidth=0, antialiased=True,
                    rstride=2, cstride=2,
                    shade=True)

    # ---- axes & cosmetics --------------------------------------------------
    ax.set_xlim(0, Nx)
    ax.set_ylim(0, Ny)
    ax.set_zlim(0, Nz - 1)

    try:
        ax.set_box_aspect((4.0, 1.3, 1.8))
    except AttributeError:
        pass  # matplotlib < 3.3 -- skip

    ax.view_init(elev=_ELEV, azim=_AZIM)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(_AXIS_COLOR)

    ax.tick_params(colors=_LABEL_COLOR, labelsize=5, pad=1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(_LABEL_COLOR)

    ax.set_xlabel("x", fontsize=6, labelpad=3)
    ax.set_ylabel("y", fontsize=6, labelpad=3)
    ax.set_zlabel("z", fontsize=6, labelpad=3)
    ax.grid(True, color=_AXIS_COLOR, linewidth=0.3)

    if title:
        ax.set_title(title, color="#c0d0ff", fontsize=9,
                     fontweight="bold", pad=6)

    fig.subplots_adjust(left=0.0, right=1.0, top=0.96, bottom=0.0)
    fig.canvas.draw()
    buf    = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w_px, h_px = fig.canvas.get_width_height()
    frame  = buf.reshape(h_px, w_px, 4)[..., :3].copy()
    plt.close(fig)
    return frame


# ---------------------------------------------------------------------------
# Enhanced 2D mid-span reference renderer (dark theme, same pixel size)
# ---------------------------------------------------------------------------
def _make_frame_2d_ref(data_norm, cmap, xc, yc, theta,
                       label: str) -> np.ndarray:
    """
    Render a normalised (Ny, Nx) field as a 2D dark-theme frame.
    figsize matches 3D panels so the composite assembles without resizing.
    Returns uint8 RGB array shape (H, W, 3).
    """
    fig, ax = plt.subplots(figsize=(_FIG_W_3D, _FIG_H_3D), dpi=_DPI_2D)
    fig.patch.set_facecolor(_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)

    ax.imshow(data_norm, origin="lower",
              cmap=cmap, vmin=-1.0, vmax=1.0,
              aspect="auto", interpolation="bilinear",
              extent=[0, Nx, 0, Ny])

    # Body polygon overlay
    Xo, Yo = _body_contour(xc, yc, theta)
    ax.fill(Xo, Yo, color=_BODY_COLOR, zorder=3, alpha=0.95)
    ax.plot(Xo, Yo, color="#ffffff", lw=0.7, zorder=4, alpha=0.6)

    # Panel label
    ax.text(8, 8, label, color="#c0d0ff", fontsize=8,
            va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor=_BG_COLOR, alpha=0.75),
            zorder=5)

    ax.set_xlim(0, Nx)
    ax.set_ylim(0, Ny)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    buf    = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w_px, h_px = fig.canvas.get_width_height()
    frame  = buf.reshape(h_px, w_px, 4)[..., :3].copy()
    plt.close(fig)
    return frame


# ---------------------------------------------------------------------------
# Capture all 4 frames for one time snapshot
# ---------------------------------------------------------------------------
def _capture_all_frames(ux_np, uy_np, rho_np, xc, yc, theta):
    """
    Generate four visualization frames for one snapshot.

    Returns
    -------
    f_umag, f_vort, f_pres, f_comp : np.ndarray uint8 (H, W, 3)

    Composite layout
    ----------------
    top-left  : 3D perspective  vorticity  omega_z
    top-right : 3D perspective  velocity   |u|
    bot-left  : 2D mid-span     vorticity  (reference slice)
    bot-right : 3D perspective  pressure   p'
    """
    # ---- full 3D field computations ----------------------------------------
    umag_3d = np.sqrt(ux_np**2 + uy_np**2)        # (Nx, Ny, Nz)

    duy_dx  = np.gradient(uy_np, axis=0)
    dux_dy  = np.gradient(ux_np, axis=1)
    wz_3d   = duy_dx - dux_dy                      # (Nx, Ny, Nz)

    p_3d    = rho_np * cs2
    dp_3d   = p_3d - float(p_3d.mean())            # pressure deviation

    umag_max = float(U_inf * 1.8)

    # ---- 3D rendered frames ------------------------------------------------
    f_vort = _make_frame_3d(wz_3d,   _VORT_CMAP, -VORT_MAX,  VORT_MAX,
                             xc, yc, theta, title="Vorticity  omega_z")
    f_umag = _make_frame_3d(umag_3d, _UMAG_CMAP,  0.0,       umag_max,
                             xc, yc, theta, title="Velocity Magnitude  |u|")
    f_pres = _make_frame_3d(dp_3d,   _PRES_CMAP, -PRES_MAX,  PRES_MAX,
                             xc, yc, theta, title="Pressure Deviation  p'")

    # ---- 2D mid-span vorticity reference slice -----------------------------
    wz_mid  = wz_3d[:, :, _Z_MID].T               # (Ny, Nx)
    wz_norm = np.clip(wz_mid / VORT_MAX, -1.0, 1.0)
    f_2d    = _make_frame_2d_ref(wz_norm, _VORT_CMAP,
                                  xc, yc, theta,
                                  "Vorticity omega_z  (mid-span 2-D)")

    # ---- composite 2x2 panel -----------------------------------------------
    H, W   = f_vort.shape[:2]
    H2, W2 = f_2d.shape[:2]

    # Resize f_2d to match 3D panel size if they differ (nearest-neighbour)
    if H2 != H or W2 != W:
        ri = (np.arange(H) * H2 / H).astype(int)
        ci = (np.arange(W) * W2 / W).astype(int)
        f_2d = f_2d[np.ix_(ri, ci)]

    sep  = 2          # must keep composite width/height even for H.264
    comp = np.zeros((2 * H + sep, 2 * W + sep, 3), dtype=np.uint8)
    comp[:H,      :W]      = f_vort
    comp[:H,      W+sep:]  = f_umag
    comp[H+sep:,  :W]      = f_2d
    comp[H+sep:,  W+sep:]  = f_pres

    return f_umag, f_vort, f_pres, comp


# ---------------------------------------------------------------------------
# Helper: open video writer
# ---------------------------------------------------------------------------
def _open_writer(path):
    try:
        return imageio.get_writer(path, fps=VIDEO_FPS, format="ffmpeg",
                                  quality=8, macro_block_size=1)
    except Exception as e:
        gif_path = path.replace(".mp4", ".gif")
        print(f"  [Warning] MP4 unavailable ({e}), falling back to {gif_path}")
        return imageio.get_writer(gif_path, fps=VIDEO_FPS, format="gif")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    period   = 1.0 / f_motion
    n_frames = N_steps // vis_interval
    print(f"3D CLBM-IBM  fish swimmer  --  Grid {Nx}x{Ny}x{Nz}   Re={Re}")
    print(f"  Body    : ellipse  chord={CHORD} lu   "
          f"thickness={FISH_THICKNESS} lu  ({CHORD//FISH_THICKNESS}:1 aspect)")
    print(f"  Motion  : A_heave={A_heave} lu ({A_heave/CHORD:.2f}c)   "
          f"A_pitch={math.degrees(A_pitch):.1f} deg   k={k_reduced}   phi=90 deg")
    print(f"  f={f_motion:.3e} step^-1   T={period:.0f} steps   "
          f"N_steps={N_steps} ({N_steps*f_motion:.1f} periods)")
    print(f"  Frames every {vis_interval} steps -> {n_frames} frames   "
          f"FPS={VIDEO_FPS}   Duration~{n_frames/VIDEO_FPS:.1f} s")
    print(f"  Frame size : {int(_FIG_W_3D*_DPI_3D)}x{int(_FIG_H_3D*_DPI_3D)} px  "
          f"(3D panels)   stride={_STRIDE_3D}   z-slices={_Z_SLICES}")
    print(f"  Outputs : {VIDEO_UMAG}  {VIDEO_VORT}  {VIDEO_PRES}  {VIDEO_COMP}")

    lbm = CLBMSolver()
    ibm = FishIBM()

    wr_umag = _open_writer(VIDEO_UMAG)
    wr_vort = _open_writer(VIDEO_VORT)
    wr_pres = _open_writer(VIDEO_PRES)
    wr_comp = _open_writer(VIDEO_COMP)

    t0   = time.perf_counter()
    last = t0

    try:
        for step in range(1, N_steps + 1):

            xc, yc, theta, vx, vy, omega = _motion_state(step)

            fx, fy, fz = ibm.compute_force(lbm.ux, lbm.uy,
                                           xc, yc, theta, vx, vy, omega)
            lbm.step(fx, fy, fz)

            if step % vis_interval == 0:
                rho_np = lbm.rho.to_numpy()
                ux_np  = lbm.ux.to_numpy()
                uy_np  = lbm.uy.to_numpy()

                rho_min = float(rho_np.min())
                rho_max = float(rho_np.max())
                u_max   = float(np.sqrt(ux_np**2 + uy_np**2).max())

                now  = time.perf_counter()
                rate = vis_interval / (now - last)
                last = now
                print(f"  step {step:6d}/{N_steps}  u_max={u_max:.4f}  "
                      f"rho=[{rho_min:.4f},{rho_max:.4f}]  "
                      f"yc={yc:.1f}  theta={math.degrees(theta):.1f} deg  "
                      f"|  {rate:.0f} stp/s")

                if not np.isfinite(rho_np).all() or rho_min < 0.3:
                    print(f"\n*** Diverged at step {step}!  rho_min={rho_min:.4f} ***")
                    break

                f_umag, f_vort, f_pres, f_comp = _capture_all_frames(
                    ux_np, uy_np, rho_np, xc, yc, theta
                )
                wr_umag.append_data(f_umag)
                wr_vort.append_data(f_vort)
                wr_pres.append_data(f_pres)
                wr_comp.append_data(f_comp)

    finally:
        for wr in (wr_umag, wr_vort, wr_pres, wr_comp):
            wr.close()

    elapsed = time.perf_counter() - t0
    print(f"\nDone. {step} steps in {elapsed:.1f} s  ({step/elapsed:.0f} stp/s).")
    print("Videos written:")
    import os
    for fn in (VIDEO_UMAG, VIDEO_VORT, VIDEO_PRES, VIDEO_COMP):
        if os.path.exists(fn):
            print(f"  {fn}  ({os.path.getsize(fn)//1024} KB)")


if __name__ == "__main__":
    main()
