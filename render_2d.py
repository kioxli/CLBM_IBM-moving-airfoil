"""
render_2d.py (v2) -- optimised 2D velocity-field video renderer (CLBM-IBM airplane)

Improvements over v1
--------------------
  Left panel  : velocity magnitude |u| (plasma)
                + speed-coloured, width-varying streamlines
                  (bright/thick = fast, dim/thin = slow)
  Right panel : spanwise vorticity  wz = dux/dy - duy/dx  (RdBu_r symmetric)
                + ux = 0  separation / reattachment contour  (gold dashed)
                + sparse direction-only quiver (white, alpha 0.3)
  Both panels : zoomed to aerodynamically relevant region around the airfoil,
                chord-length scale bar, geometry overlays

Physics notes
-------------
  Coloured streamlines encode local speed on the same plasma scale as the
  background heat-map, making slow recirculation bubbles immediately visible.

  The vorticity panel shows the positive (CCW, red) and negative (CW, blue)
  vortex structures shed from the wing and fuselage.  The colour range is
  auto-calibrated to the boundary-layer scale:
      wz_range = CLIM_VORT_FACTOR * U_inf * sqrt(Re) / c

  The gold ux = 0 iso-line marks the boundary of the reverse-flow region.

Run:
  python render_2d.py
  python render_2d.py --dir drone_frames --out vel2d.mp4 --fps 15 --stride 2
"""

import argparse
import glob
import math
import os
import sys
import time as _time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm

try:
    import imageio
except ImportError:
    sys.exit("imageio required:  pip install imageio[ffmpeg]")


# ============================================================
# Render parameters  (tune here, no code changes needed)
# ============================================================
DATA_DIR    = "drone_frames"
VIDEO_FILE  = "vel2d.mp4"
VIDEO_FPS   = 15
STRIDE      = 2          # x-y spatial sub-sampling  (1 = full, 2 = half each axis)

# ── Streamplot (left panel) ────────────────────────────────
STREAM_SKIP    = 4       # sub-sample factor for the streamplot grid (on top of STRIDE)
STREAM_DENSITY = 1.4     # matplotlib density parameter
LW_MIN, LW_MAX = 0.4, 2.2  # streamline linewidth range (thin=slow, thick=fast)

# ── Vorticity (right panel) ────────────────────────────────
CLIM_VORT_FACTOR = 1.5   # wz range = FACTOR * U_inf * sqrt(Re) / c
                          # increase if vorticity is clipped in the wake
QUIVER_SKIP      = 18    # sparse direction-quiver spacing (post-stride grid units)

# ── View region (chords relative to pivot x0, y0) ─────────
VIEW_UP    = 1.5   # chords upstream  of x0 shown
VIEW_DOWN  = 4.0   # chords downstream
VIEW_VERT  = 1.4   # chords above/below y0

# ── Appearance ─────────────────────────────────────────────
DPI       = 120
BG_COLOR  = "#0a0a14"   # very dark indigo
CMAP_MAG  = "plasma"    # |u| and streamline colour
CMAP_VORT = "RdBu_r"    # vorticity (diverging)
CLIM_MAG  = 1.6         # |u|_max  = U_inf * CLIM_MAG

FRAME_RANGE = None       # None = all; (start, stop) for a subset


# ============================================================
# Data I/O
# ============================================================

def load_meta(data_dir: str) -> dict:
    m  = np.load(os.path.join(data_dir, "meta.npz"), allow_pickle=False)
    gm = m["grid_meta"]
    Xt = np.array(m["X_template"])
    meta = dict(
        Nx=int(gm[0]),  Ny=int(gm[1]),  Nz=int(gm[2]),
        U_inf=float(gm[3]),  Re=float(gm[4]),
        x0=float(gm[5]),  y0=float(gm[6]),  z0=float(gm[7]),
        fuse_ax=float(gm[8]),  fuse_ay=float(gm[9]),
        b_half=float(gm[10]),
        iz=int(round(gm[13])),
        X_template=Xt,
        Y_template=np.array(m["Y_template"]),
        xc   =np.array(m["xc"]),
        yc   =np.array(m["yc"]),
        theta=np.array(m["theta"]),
        step =np.array(m["step"]),
        # chord derived from template extent
        c    =float(Xt.max() - Xt.min()),
    )
    meta["X_htail_tmpl"] = np.array(m["X_htail_tmpl"]) if "X_htail_tmpl" in m else None
    meta["Y_htail_tmpl"] = np.array(m["Y_htail_tmpl"]) if "Y_htail_tmpl" in m else None
    meta["x_tail_arm"]   = float(gm[17]) if len(gm) > 17 else 0.0
    meta["c_vtail"]      = float(gm[22]) if len(gm) > 22 else 0.0
    meta["h_vtail"]      = float(gm[23]) if len(gm) > 23 else 0.0
    return meta


def load_slice(path: str, iz: int, stride: int = 1):
    """Load ux, uy at the z=iz mid-span slice; returns (Nx//stride, Ny//stride)."""
    d = np.load(path, allow_pickle=False)
    s = slice(None, None, stride) if stride > 1 else slice(None)
    return (d["ux"][s, s, iz].astype(np.float32),
            d["uy"][s, s, iz].astype(np.float32))


# ============================================================
# Geometry helpers
# ============================================================

def _b2w(xc, yc, ct, st, X0, Y0):
    return xc + X0 * ct - Y0 * st, yc + X0 * st + Y0 * ct


def wing_outline_xy(xc, yc, theta, meta):
    ct, st = math.cos(theta), math.sin(theta)
    Xw, Yw = _b2w(xc, yc, ct, st, meta["X_template"], meta["Y_template"])
    return np.append(Xw, Xw[0]), np.append(Yw, Yw[0])


def fuse_outline_xy(xc, yc, theta, meta, n=80):
    ct, st = math.cos(theta), math.sin(theta)
    t  = np.linspace(0, 2 * np.pi, n)
    return _b2w(xc, yc, ct, st,
                meta["fuse_ax"] * np.cos(t),
                meta["fuse_ay"] * np.sin(t))


def htail_outline_xy(xc, yc, theta, meta):
    if meta["X_htail_tmpl"] is None or meta["x_tail_arm"] <= 0:
        return None, None
    ct, st = math.cos(theta), math.sin(theta)
    xw, yw = _b2w(xc, yc, ct, st,
                  meta["X_htail_tmpl"] + meta["x_tail_arm"],
                  meta["Y_htail_tmpl"])
    return np.append(xw, xw[0]), np.append(yw, yw[0])


def vtail_outline_xy(xc, yc, theta, meta):
    arm = meta.get("x_tail_arm", 0.0)
    hc  = meta.get("c_vtail", 0.0) / 2.0
    ht  = meta.get("h_vtail", 0.0)
    fay = meta["fuse_ay"]
    if arm <= 0 or ht <= 0:
        return None, None
    ct, st = math.cos(theta), math.sin(theta)
    cx = [arm - hc, arm + hc, arm + hc, arm - hc, arm - hc]
    cy = [fay,      fay,      fay + ht, fay + ht, fay     ]
    xw = [xc + c_ * ct - c2 * st for c_, c2 in zip(cx, cy)]
    yw = [yc + c_ * st + c2 * ct for c_, c2 in zip(cx, cy)]
    return np.array(xw), np.array(yw)


def _overlay_geom(ax, xc, yc, theta, meta):
    wx, wy = wing_outline_xy(xc, yc, theta, meta)
    ax.plot(wx, wy, color="white",   lw=1.8, alpha=0.95, zorder=6)

    fx, fy = fuse_outline_xy(xc, yc, theta, meta)
    ax.plot(fx, fy, color="#00e5ff", lw=1.1, ls="--", alpha=0.65, zorder=6)

    hx, hy = htail_outline_xy(xc, yc, theta, meta)
    if hx is not None:
        ax.plot(hx, hy, color="white", lw=1.0, alpha=0.70, zorder=6)

    vx, vy = vtail_outline_xy(xc, yc, theta, meta)
    if vx is not None:
        ax.plot(vx, vy, color="#ffcc44", lw=1.0, ls=":", alpha=0.60, zorder=6)


def _scale_bar(ax, x_lo, x_hi, y_lo, y_hi, c_est):
    """Draw a chord-length scale bar in the top-left corner."""
    bx0 = x_lo + 0.04 * (x_hi - x_lo)
    by  = y_hi - 0.07 * (y_hi - y_lo)
    ax.plot([bx0, bx0 + c_est], [by, by],
            color="white", lw=2.0, solid_capstyle="round", zorder=10)
    ax.text(bx0 + c_est * 0.5, by + 0.025 * (y_hi - y_lo),
            f"c = {c_est:.0f} lu",
            color="white", fontsize=7, ha="center", va="bottom", zorder=10)


# ============================================================
# Figure setup  (called once)
# ============================================================

def make_figure(meta, norm_mag, norm_vort, cmap_mag, cmap_vort):
    fig, axes = plt.subplots(1, 2, figsize=(19, 7), dpi=DPI, facecolor=BG_COLOR)
    fig.subplots_adjust(left=0.055, right=0.945, top=0.91, bottom=0.09, wspace=0.40)

    for ax in axes:
        ax.set_facecolor(BG_COLOR)
        for sp in ax.spines.values():
            sp.set_color("#2a2a4a")
        ax.tick_params(colors="#888", labelsize=7)

    sm_mag = plt.cm.ScalarMappable(norm=norm_mag,  cmap=cmap_mag)
    sm_mag.set_array([])
    cb1 = fig.colorbar(sm_mag,  ax=axes[0], fraction=0.036, pad=0.025)
    cb1.set_label("|u|  [lu/step]", color="#bbb", fontsize=8)
    cb1.ax.tick_params(colors="#bbb", labelsize=7)
    cb1.ax.yaxis.set_tick_params(length=3)

    sm_vort = plt.cm.ScalarMappable(norm=norm_vort, cmap=cmap_vort)
    sm_vort.set_array([])
    cb2 = fig.colorbar(sm_vort, ax=axes[1], fraction=0.036, pad=0.025)
    cb2.set_label("wz  [step\u207b\u00b9]", color="#bbb", fontsize=8)
    cb2.ax.tick_params(colors="#bbb", labelsize=7)
    cb2.ax.yaxis.set_tick_params(length=3)

    return fig, (axes[0], axes[1])


# ============================================================
# Per-frame drawing
# ============================================================

def _view_bounds(meta, stride):
    """
    Compute fixed view window in original lattice coordinates.
    Centred at (x0, y0); follows VIEW_UP/DOWN/VERT constants.
    Clamped to [0, Nx] x [0, Ny].
    """
    c  = meta["c"]
    x0 = meta["x0"]
    y0 = meta["y0"]
    Nx = meta["Nx"]
    Ny = meta["Ny"]
    xl = max(0.0,  x0 - VIEW_UP   * c)
    xh = min(float(Nx), x0 + VIEW_DOWN * c)
    yl = max(0.0,  y0 - VIEW_VERT * c)
    yh = min(float(Ny), y0 + VIEW_VERT * c)
    return xl, xh, yl, yh


def draw_frame(fig, axes, ux, uy, meta, fidx, stride,
               norm_mag, norm_vort, cmap_mag, cmap_vort,
               view_bounds):
    ax_mag, ax_vort = axes
    Nx_s, Ny_s = ux.shape
    U_inf  = meta["U_inf"]
    c_est  = meta["c"]
    xc     = float(meta["xc"][fidx])
    yc     = float(meta["yc"][fidx])
    theta  = float(meta["theta"][fidx])
    step_n = int(meta["step"][fidx])
    x_lo, x_hi, y_lo, y_hi = view_bounds

    # Full-domain extent in lattice units
    extent = [0, Nx_s * stride, 0, Ny_s * stride]

    # ── Derived fields ────────────────────────────────────────────────────
    umag = np.sqrt(ux**2 + uy**2)

    # Vorticity: wz = dux/dy - duy/dx  (spacing = stride lu)
    dux_dy = np.gradient(ux, float(stride), axis=1)
    duy_dx = np.gradient(uy, float(stride), axis=0)
    vort_z = dux_dy - duy_dx

    # ── Streamplot sub-grid (only within view window for speed) ───────────
    ss    = STREAM_SKIP
    xi0   = max(0,     int(x_lo / (stride * ss)))
    xi1   = min(Nx_s // ss + 1, int(x_hi / (stride * ss)) + 2)
    yi0   = max(0,     int(y_lo / (stride * ss)))
    yi1   = min(Ny_s // ss + 1, int(y_hi / (stride * ss)) + 2)

    x_sp  = np.arange(xi0, xi1) * ss * stride
    y_sp  = np.arange(yi0, yi1) * ss * stride
    U_sp  = ux[xi0*ss : xi1*ss : ss, yi0*ss : yi1*ss : ss].T  # (len(y_sp), len(x_sp))
    V_sp  = uy[xi0*ss : xi1*ss : ss, yi0*ss : yi1*ss : ss].T
    spd_sp = np.sqrt(U_sp**2 + V_sp**2)

    # Variable linewidth: thin / dim for slow regions, thick for fast
    lw_sp  = np.clip(LW_MIN + (LW_MAX - LW_MIN) * spd_sp / (U_inf * CLIM_MAG + 1e-10),
                     LW_MIN, LW_MAX)

    # Contour coordinates for ux=0 line
    x_cont = np.arange(Nx_s) * stride
    y_cont = np.arange(Ny_s) * stride

    # Quiver (right panel, direction only)
    qs   = QUIVER_SKIP
    xqv  = np.arange(0, Nx_s, qs) * stride
    yqv  = np.arange(0, Ny_s, qs) * stride
    Xqv, Yqv = np.meshgrid(xqv, yqv)
    Uqv  = ux[::qs, ::qs].T
    Vqv  = uy[::qs, ::qs].T
    spd_qv = np.sqrt(Uqv**2 + Vqv**2) + 1e-10

    # ── Left panel: |u| + speed-coloured streamlines ─────────────────────
    ax_mag.cla()
    ax_mag.set_facecolor(BG_COLOR)

    ax_mag.imshow(
        umag.T, origin="lower", extent=extent,
        cmap=cmap_mag, norm=norm_mag,
        interpolation="bicubic", aspect="auto",
    )



    _overlay_geom(ax_mag, xc, yc, theta, meta)
    _scale_bar(ax_mag, x_lo, x_hi, y_lo, y_hi, c_est)

    ax_mag.set_xlim(x_lo, x_hi)
    ax_mag.set_ylim(y_lo, y_hi)
    ax_mag.set_xlabel("x  [lu]", color="#aaa", fontsize=8)
    ax_mag.set_ylabel("y  [lu]", color="#aaa", fontsize=8)
    ax_mag.set_title("|u|  velocity magnitude  +  speed-coloured streamlines",
                     color="white", fontsize=9, pad=5)
    ax_mag.tick_params(colors="#777", labelsize=7)

    # ── Right panel: vorticity + ux=0 + direction quiver ─────────────────
    ax_vort.cla()
    ax_vort.set_facecolor(BG_COLOR)

    ax_vort.imshow(
        vort_z.T, origin="lower", extent=extent,
        cmap=cmap_vort, norm=norm_vort,
        interpolation="bicubic", aspect="auto",
    )

    # ux = 0 separation / reattachment contour
    try:
        cs = ax_vort.contour(
            x_cont, y_cont, ux.T,
            levels=[0.0],
            colors=["#ffdd00"],
            linewidths=[1.6],
            linestyles=["--"],
            zorder=5,
        )
        # Suppress the default contour label (no clabel needed here)
        cs.collections[0].set_label("ux = 0")
    except Exception:
        pass

    # Sparse direction-only quiver
    ax_vort.quiver(
        Xqv, Yqv, Uqv / spd_qv, Vqv / spd_qv,
        color="white", alpha=0.28,
        scale=5.0, width=0.0018,
        headwidth=3.5, headlength=4.5,
        zorder=4,
    )

    _overlay_geom(ax_vort, xc, yc, theta, meta)
    _scale_bar(ax_vort, x_lo, x_hi, y_lo, y_hi, c_est)

    ax_vort.set_xlim(x_lo, x_hi)
    ax_vort.set_ylim(y_lo, y_hi)
    ax_vort.set_xlabel("x  [lu]", color="#aaa", fontsize=8)
    ax_vort.set_ylabel("y  [lu]", color="#aaa", fontsize=8)
    ax_vort.set_title(
        "wz  vorticity  (red=CCW / blue=CW)  |  gold = ux=0 separation line",
        color="white", fontsize=9, pad=5,
    )
    ax_vort.tick_params(colors="#777", labelsize=7)

    # ── Shared title ─────────────────────────────────────────────────────
    fig.suptitle(
        f"CLBM-IBM Airplane   Re = {meta['Re']:.0f}   c = {c_est:.0f} lu   "
        f"step = {step_n:6d}   "
        f"theta = {math.degrees(theta):+.1f}\u00b0   "
        f"yc = {yc:.1f} lu   "
        f"z = {meta['iz']}",
        color="white", fontsize=10, y=0.975,
    )

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    return np.asarray(buf)[..., :3].copy()


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="2D velocity-field + vorticity video renderer (CLBM-IBM airplane)")
    ap.add_argument("--dir",    default=DATA_DIR)
    ap.add_argument("--out",    default=VIDEO_FILE)
    ap.add_argument("--fps",    type=int, default=VIDEO_FPS)
    ap.add_argument("--stride", type=int, default=STRIDE,
                    help="Spatial sub-sampling stride (default: %(default)s)")
    args = ap.parse_args()

    print("=" * 65)
    print("2D Velocity-Field + Vorticity Render  --  CLBM-IBM Airplane")
    print("=" * 65)

    if not os.path.isdir(args.dir):
        sys.exit(f"ERROR: '{args.dir}' not found.  Run main_moving.py first.")
    if not os.path.exists(os.path.join(args.dir, "meta.npz")):
        sys.exit(f"ERROR: meta.npz not found in '{args.dir}'.")

    meta  = load_meta(args.dir)
    U_inf = meta["U_inf"]
    Re    = meta["Re"]
    c_est = meta["c"]
    iz    = meta["iz"]

    # Physics-calibrated vorticity colour range
    vort_range = CLIM_VORT_FACTOR * U_inf * (Re ** 0.5) / c_est

    print(f"  Grid      : {meta['Nx']} x {meta['Ny']} x {meta['Nz']}")
    print(f"  U_inf={U_inf}   Re={Re:.0f}   c={c_est:.0f} lu")
    print(f"  z-slice   : iz = {iz}  (z0 = {meta['z0']})")
    print(f"  Colormaps : {CMAP_MAG} (|u|)  /  {CMAP_VORT} (vorticity)")
    print(f"  wz range  : +/- {vort_range:.5f} step^-1"
          f"  (factor={CLIM_VORT_FACTOR} * U*sqrt(Re)/c)")

    view_bounds = _view_bounds(meta, args.stride)
    print(f"  View      : x=[{view_bounds[0]:.0f},{view_bounds[1]:.0f}]  "
          f"y=[{view_bounds[2]:.0f},{view_bounds[3]:.0f}] lu"
          f"  ({VIEW_UP}c up / {VIEW_DOWN}c down / +/-{VIEW_VERT}c vert)")

    frame_files = sorted(glob.glob(os.path.join(args.dir, "frame_*.npz")))
    n_total     = len(frame_files)
    if n_total == 0:
        sys.exit("No frame files found.")

    s_fr = 0
    e_fr = n_total
    if FRAME_RANGE:
        s_fr, e_fr = FRAME_RANGE[0], min(FRAME_RANGE[1], n_total)
    indices = list(range(s_fr, e_fr))
    print(f"  Frames    : {len(indices)} / {n_total}   stride={args.stride}")

    # Normalisation objects
    norm_mag  = mcolors.Normalize(vmin=0.0, vmax=U_inf * CLIM_MAG)
    norm_vort = TwoSlopeNorm(vmin=-vort_range, vcenter=0.0, vmax=vort_range)
    cmap_mag  = plt.get_cmap(CMAP_MAG)
    cmap_vort = plt.get_cmap(CMAP_VORT)

    fig, axes = make_figure(meta, norm_mag, norm_vort, cmap_mag, cmap_vort)

    frames_out = []
    t0 = _time.perf_counter()

    for i, fi in enumerate(indices):
        ux, uy = load_slice(frame_files[fi], iz, args.stride)
        img    = draw_frame(fig, axes, ux, uy, meta, fi, args.stride,
                            norm_mag, norm_vort, cmap_mag, cmap_vort,
                            view_bounds)
        frames_out.append(img)

        elapsed = _time.perf_counter() - t0
        rate    = (i + 1) / max(elapsed, 1e-6)
        eta     = (len(indices) - i - 1) / max(rate, 1e-6)
        print(f"\r  [{i+1:3d}/{len(indices)}]  {rate:.1f} fr/s   ETA {eta:.0f}s   ",
              end="", flush=True)

    print()
    plt.close(fig)

    out = args.out
    print(f"Writing -> {out}  ({len(frames_out)} frames @ {args.fps} fps) ...")
    try:
        imageio.mimwrite(out, frames_out, fps=args.fps,
                         format="ffmpeg", quality=8, macro_block_size=1)
    except Exception as exc:
        gif_out = out.rsplit(".", 1)[0] + ".gif"
        print(f"MP4 failed ({exc}), fallback GIF -> {gif_out}")
        imageio.mimsave(gif_out, frames_out, duration=1.0 / args.fps)
        out = gif_out

    print(f"Done: {out}  ({_time.perf_counter() - t0:.1f} s total)")


if __name__ == "__main__":
    main()
