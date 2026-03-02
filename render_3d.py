"""
render_3d.py — 3D matplotlib 体绘制渲染 CLBM-IBM 飞机仿真。

仅保留 3D 透视视图：全场速度模体散点 + 飞机几何（单色），无剖面图。

数据: drone_frames/meta.npz + drone_frames/frame_*.npz
输出: airplane_3d.mp4 (或 .gif)

运行:
  python render_3d.py
  python render_3d.py --dir drone_frames --out out.mp4 --fps 10 --stride 2
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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D, art3d   # noqa: F401  (activates 3D projection)

try:
    import imageio
except ImportError:
    sys.exit("imageio required:  pip install imageio[ffmpeg]")


# ============================================================================
# Render parameters  (edit here, no need to touch code below)
# ============================================================================
DATA_DIR    = "drone_frames"
VIDEO_FILE  = "airplane_3d.mp4"
VIDEO_FPS   = 10

# Spatial sub-sampling when loading frame data (1 = full, 2 = half each axis)
STRIDE      = 2

# 3D 体绘制：体素/散点下采样步长（在 STRIDE 之后再每隔多少格取一点）
STRIDE_3D   = 5
# 散点大小与不透明度（点太小或太透明会看不见）
SCATTER_S   = 12
SCATTER_ALPHA = 0.3
# 飞机几何统一颜色（单色绘制）
GEOM_COLOR  = "#e8e8ff"

# Process only a subset: None = all frames; (start, stop) = index range
FRAME_RANGE = None

DPI         = 100          # figure DPI
CMAP        = "inferno"    # velocity colormap
BG_COLOR    = "#0a0a14"    # figure / axes background
CLIM_FACTOR = 1.6          # colour scale = U_inf * CLIM_FACTOR
AX3D_ELEV   = 22           # 3D view elevation (degrees)
AX3D_AZIM   = -52          # 3D view azimuth   (degrees)


# ============================================================================
# Data I/O
# ============================================================================

def load_meta(data_dir: str) -> dict:
    m  = np.load(os.path.join(data_dir, "meta.npz"), allow_pickle=False)
    gm = m["grid_meta"]
    meta = dict(
        Nx=int(gm[0]),  Ny=int(gm[1]),  Nz=int(gm[2]),
        U_inf=float(gm[3]),  Re=float(gm[4]),
        x0=float(gm[5]),  y0=float(gm[6]),  z0=float(gm[7]),
        fuse_ax=float(gm[8]),  fuse_ay=float(gm[9]),
        b_half=float(gm[10]),
        z_wing_start=float(gm[11]),  z_wing_end=float(gm[12]),
        iz=int(round(gm[13])),  ix=int(round(gm[14])),
        vis_interval=int(gm[15]),
        fuse_az   =float(gm[16]) if len(gm) > 16 else 10.0,
        x_tail_arm=float(gm[17]) if len(gm) > 17 else  0.0,
        c_htail      =float(gm[18]) if len(gm) > 18 else 0.0,
        b_half_htail =float(gm[19]) if len(gm) > 19 else 0.0,
        z_htail_start=float(gm[20]) if len(gm) > 20 else 0.0,
        z_htail_end  =float(gm[21]) if len(gm) > 21 else 0.0,
        c_vtail=float(gm[22]) if len(gm) > 22 else 0.0,
        h_vtail=float(gm[23]) if len(gm) > 23 else 0.0,
        X_template=np.array(m["X_template"]),
        Y_template=np.array(m["Y_template"]),
        xc   =np.array(m["xc"]),
        yc   =np.array(m["yc"]),
        theta=np.array(m["theta"]),
        step =np.array(m["step"]),
    )
    meta["X_htail_tmpl"] = np.array(m["X_htail_tmpl"]) if "X_htail_tmpl" in m else None
    meta["Y_htail_tmpl"] = np.array(m["Y_htail_tmpl"]) if "Y_htail_tmpl" in m else None
    # derive mid-height slice index (will be refined from actual Ny after loading)
    meta["iy"] = int(round(meta["y0"]))
    return meta


def load_frame(path: str, stride: int = 1):
    d = np.load(path, allow_pickle=False)
    s = slice(None, None, stride) if stride > 1 else slice(None)
    return (d["ux"][s, s, s].astype(np.float32),
            d["uy"][s, s, s].astype(np.float32),
            d["uz"][s, s, s].astype(np.float32),
            d["rho"][s, s, s].astype(np.float32))


# ============================================================================
# Airplane geometry  (world frame, parameterised by current xc, yc, theta)
# ============================================================================

def _body2world(xc, yc, ct, st, X0, Y0):
    """Apply pitch rotation + translation: local → world (x, y)."""
    return xc + X0 * ct - Y0 * st, yc + X0 * st + Y0 * ct


def build_airplane_geom(xc, yc, theta, meta) -> dict:
    """
    Pre-compute all world-frame geometry for one simulation frame.
    Returns a dict with arrays ready to be plotted.
    """
    ct, st = math.cos(theta), math.sin(theta)
    g = {}

    # ── Main wing ──────────────────────────────────────────────────────────
    Xw, Yw = _body2world(xc, yc, ct, st, meta["X_template"], meta["Y_template"])
    # closed profile
    g["wing_x"] = np.append(Xw, Xw[0])
    g["wing_y"] = np.append(Yw, Yw[0])
    # LE / TE world positions (for span lines)
    x_le = meta["X_template"].min()
    x_te = meta["X_template"].max()
    g["wing_le_x"], g["wing_le_y"] = xc + x_le * ct, yc + x_le * st
    g["wing_te_x"], g["wing_te_y"] = xc + x_te * ct, yc + x_te * st
    g["z_ws"] = meta["z_wing_start"]
    g["z_we"] = meta["z_wing_end"]

    # ── Fuselage ───────────────────────────────────────────────────────────
    t = np.linspace(0, 2 * np.pi, 100)
    xl = meta["fuse_ax"] * np.cos(t)
    yl = meta["fuse_ay"] * np.sin(t)
    g["fuse_x"],       g["fuse_y"]       = _body2world(xc, yc, ct, st, xl, yl)
    # Top-view footprint (x-z, no pitch since z is rigid)
    g["fuse_x_top"] = xc + meta["fuse_ax"] * np.cos(t)
    g["fuse_z_top"] = meta["z0"] + meta["fuse_az"] * np.sin(t)
    # Front-view cross-section (y-z)
    g["fuse_z_front"] = meta["z0"] + meta["fuse_az"] * np.cos(t)
    g["fuse_y_front"] = yc          + meta["fuse_ay"] * np.sin(t)

    # ── Horizontal stabiliser ──────────────────────────────────────────────
    if meta["X_htail_tmpl"] is not None and meta["x_tail_arm"] > 0:
        Xht_loc = meta["X_htail_tmpl"] + meta["x_tail_arm"]
        Yht_loc = meta["Y_htail_tmpl"]
        Xht, Yht = _body2world(xc, yc, ct, st, Xht_loc, Yht_loc)
        g["htail_x"] = np.append(Xht, Xht[0])
        g["htail_y"] = np.append(Yht, Yht[0])
        ht_le = meta["x_tail_arm"] + meta["X_htail_tmpl"].min()
        ht_te = meta["x_tail_arm"] + meta["X_htail_tmpl"].max()
        g["htail_le_x"], g["htail_le_y"] = xc + ht_le * ct, yc + ht_le * st
        g["htail_te_x"], g["htail_te_y"] = xc + ht_te * ct, yc + ht_te * st
        g["z_hts"] = meta["z_htail_start"]
        g["z_hte"] = meta["z_htail_end"]
        # Top-view rectangle corners (approximate: no pitch rotation on z)
        g["htail_xle_top"] = xc + ht_le * ct
        g["htail_xte_top"] = xc + ht_te * ct
    else:
        g["htail_x"] = None

    # ── Vertical fin ──────────────────────────────────────────────────────
    if meta["h_vtail"] > 0 and meta["x_tail_arm"] > 0:
        arm  = meta["x_tail_arm"]
        h0   = meta["fuse_ay"]           # LOCAL y: fin root
        h1   = h0 + meta["h_vtail"]      # LOCAL y: fin tip
        # fin root world (x_local=arm, y_local=h0)
        g["vfin_root_x"] = xc + arm * ct - h0 * st
        g["vfin_root_y"] = yc + arm * st + h0 * ct
        # fin tip world (x_local=arm, y_local=h1)
        g["vfin_tip_x"]  = xc + arm * ct - h1 * st
        g["vfin_tip_y"]  = yc + arm * st + h1 * ct
        # chord extent in top view at z = z0 (local: x from arm-c/2 to arm+c/2, y_local~0)
        hc = meta["c_vtail"] / 2.0
        g["vfin_chord_le_x"] = xc + (arm - hc) * ct
        g["vfin_chord_te_x"] = xc + (arm + hc) * ct
    else:
        g["vfin_root_x"] = None

    return g


# ============================================================================
# Figure setup
# ============================================================================

def make_figure(norm, cmap_fn):
    """Create the persistent figure + single 3D axes + colorbar (called once)."""
    fig = plt.figure(figsize=(12, 9), dpi=DPI, facecolor=BG_COLOR)
    ax3d = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.88, top=0.95, bottom=0.05)
    ax3d.set_facecolor("#080814")
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#1a1a2e")
    sm = ScalarMappable(norm=norm, cmap=cmap_fn)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3d, shrink=0.6, aspect=25, label="|u|")
    cbar.ax.tick_params(colors="#aaa", labelsize=7)
    cbar.ax.yaxis.label.set_color("#aaa")
    return fig, ax3d


# ============================================================================
# Per-frame rendering
# ============================================================================

def draw_frame(fig, ax3d, ux, uy, uz, rho, meta, fidx, stride, norm, cmap_fn):
    """
    Clear 3D axes and draw one frame: volume scatter + airplane (single color).
    Returns RGB image as uint8 ndarray of shape (H, W, 3).
    """
    ax3d.cla()

    Nx_s, Ny_s, Nz_s = ux.shape
    xc    = float(meta["xc"][fidx])
    yc    = float(meta["yc"][fidx])
    theta = float(meta["theta"][fidx])
    step_n = int(meta["step"][fidx])

    x_arr = np.arange(Nx_s) * stride
    y_arr = np.arange(Ny_s) * stride
    z_arr = np.arange(Nz_s) * stride

    umag = np.sqrt(ux**2 + uy**2 + uz**2)
    g = build_airplane_geom(xc, yc, theta, meta)

    # ── 3D 体散点（先画，避免被几何完全遮挡）────────────────────────────────
    ax3d.set_facecolor("#080814")
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#1a1a2e")

    s3 = STRIDE_3D
    umag_s = umag[::s3, ::s3, ::s3]
    xs = x_arr[::s3]
    ys = y_arr[::s3]
    zs = z_arr[::s3]
    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing="ij")
    x_flat = Xg.ravel()
    y_flat = Yg.ravel()
    z_flat = Zg.ravel()
    c_flat = umag_s.ravel()
    # 用标量 c + cmap/norm 让 3D scatter 正确上色并可见（点过小/预算 RGBA 易不显示）
    ax3d.scatter(
        x_flat, z_flat, y_flat,
        c=c_flat, cmap=cmap_fn, norm=norm,
        s=SCATTER_S, alpha=SCATTER_ALPHA, edgecolors="none", marker="o"
    )

    # ── 飞机几何（实心，单色）────────────────────────────────────────────
    geom = GEOM_COLOR
    z0, fuse_az = meta["z0"], meta["fuse_az"]
    ct, st = math.cos(theta), math.sin(theta)

    def add_solid_wing(wx, wy, z_lo, z_hi):
        """机翼：上下两底面 + 侧面四边形，坐标系 (x, z, y)。"""
        n = len(wx) - 1  # 闭合多边形的点数（不含重复首点）
        polys = []
        bot = np.column_stack([wx[:n], np.full(n, z_lo), wy[:n]])
        top = np.column_stack([wx[:n], np.full(n, z_hi), wy[:n]])
        polys.append(bot)
        polys.append(top)
        for i in range(n):
            polys.append(np.array([
                [wx[i], z_lo, wy[i]], [wx[i + 1], z_lo, wy[i + 1]],
                [wx[i + 1], z_hi, wy[i + 1]], [wx[i], z_hi, wy[i]],
            ]))
        coll = art3d.Poly3DCollection(polys, facecolors=geom, edgecolors="none", alpha=0.92)
        ax3d.add_collection3d(coll)

    add_solid_wing(g["wing_x"], g["wing_y"], g["z_ws"], g["z_we"])

    # 机身：椭圆柱面 (x,y 椭圆随 theta，z 方向拉伸)
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(-1, 1, 24)
    U, V = np.meshgrid(u, v)
    xb = meta["fuse_ax"] * np.cos(U)
    yb = meta["fuse_ay"] * np.sin(U)
    X_f = xc + xb * ct - yb * st
    Y_f = yc + xb * st + yb * ct
    Z_f = z0 + fuse_az * V
    ax3d.plot_surface(X_f, Z_f, Y_f, color=geom, shade=False, alpha=0.92, edgecolor="none")

    if g["htail_x"] is not None:
        def add_solid_htail(hx, hy, z_lo, z_hi):
            n = len(hx) - 1
            polys = []
            bot = np.column_stack([hx[:n], np.full(n, z_lo), hy[:n]])
            top = np.column_stack([hx[:n], np.full(n, z_hi), hy[:n]])
            polys.append(bot)
            polys.append(top)
            for i in range(n):
                polys.append(np.array([
                    [hx[i], z_lo, hy[i]], [hx[i + 1], z_lo, hy[i + 1]],
                    [hx[i + 1], z_hi, hy[i + 1]], [hx[i], z_hi, hy[i]],
                ]))
            coll = art3d.Poly3DCollection(polys, facecolors=geom, edgecolors="none", alpha=0.92)
            ax3d.add_collection3d(coll)
        add_solid_htail(g["htail_x"], g["htail_y"], g["z_hts"], g["z_hte"])

    if g["vfin_root_x"] is not None:
        # 垂尾：一片四边形 (x,z,y)
        quad = np.array([
            [g["vfin_chord_le_x"], z0, g["vfin_root_y"]],
            [g["vfin_chord_te_x"], z0, g["vfin_root_y"]],
            [g["vfin_chord_te_x"], z0, g["vfin_tip_y"]],
            [g["vfin_chord_le_x"], z0, g["vfin_tip_y"]],
        ])
        coll = art3d.Poly3DCollection([quad], facecolors=geom, edgecolors="none", alpha=0.92)
        ax3d.add_collection3d(coll)

    ax3d.set_xlabel("x (stream)", color="#888", fontsize=7, labelpad=0)
    ax3d.set_ylabel("z (span)",   color="#888", fontsize=7, labelpad=0)
    ax3d.set_zlabel("y (vert)",   color="#888", fontsize=7, labelpad=0)
    ax3d.tick_params(colors="#555", labelsize=6, pad=0)
    ax3d.set_xlim(0, Nx_s * stride)
    ax3d.set_ylim(0, Nz_s * stride)
    ax3d.set_zlim(0, Ny_s * stride)
    ax3d.view_init(elev=AX3D_ELEV, azim=AX3D_AZIM)
    ax3d.set_title(
        f"3D  step={step_n:6d}   theta={math.degrees(theta):+.1f}   yc={yc:.1f}",
        color="white", fontsize=9, pad=4,
    )
    fig.suptitle(
        f"CLBM-IBM Airplane   Re={meta['Re']:.0f}   step={step_n}   "
        f"theta={math.degrees(theta):+.1f} deg   yc={yc:.1f} lu",
        color="white", fontsize=10, y=0.98,
    )

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    return np.asarray(buf)[..., :3].copy()


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="3D matplotlib render of CLBM-IBM airplane simulation")
    ap.add_argument("--dir",    default=DATA_DIR,   help="Frame data directory")
    ap.add_argument("--out",    default=VIDEO_FILE,  help="Output video (.mp4)")
    ap.add_argument("--fps",    type=int, default=VIDEO_FPS)
    ap.add_argument("--stride", type=int, default=STRIDE,
                    help="Spatial sub-sampling (1=full, 2=half each axis)")
    args = ap.parse_args()

    print("=" * 65)
    print("3D Render  —  CLBM-IBM Simplified Airplane Model")
    print("=" * 65)

    if not os.path.isdir(args.dir):
        sys.exit(f"ERROR: '{args.dir}' not found.  Run main_moving.py first.")
    meta_path = os.path.join(args.dir, "meta.npz")
    if not os.path.exists(meta_path):
        sys.exit(f"ERROR: '{meta_path}' not found.")

    meta = load_meta(args.dir)
    print(f"  Grid      : {meta['Nx']} x {meta['Ny']} x {meta['Nz']}")
    print(f"  U_inf={meta['U_inf']}   Re={meta['Re']:.0f}")
    print(f"  Wing      : span={2*meta['b_half']:.0f} lu  z=[{meta['z_wing_start']:.0f},{meta['z_wing_end']:.0f}]")
    print(f"  H-stab    : arm={meta['x_tail_arm']:.0f} lu  span={2*meta['b_half_htail']:.0f} lu")
    print(f"  V-fin     : h={meta['h_vtail']:.0f} lu")

    frame_files = sorted(glob.glob(os.path.join(args.dir, "frame_*.npz")))
    n_total = len(frame_files)
    print(f"  Frames    : {n_total}")
    if n_total == 0:
        sys.exit("No frame files found.  Run main_moving.py first.")

    s, e = 0, n_total
    if FRAME_RANGE:
        s = FRAME_RANGE[0]
        e = min(FRAME_RANGE[1], n_total)
    indices = list(range(s, e))
    print(f"  Rendering : {len(indices)} frames   stride={args.stride}   stride_3d={STRIDE_3D}")

    norm    = Normalize(vmin=0.0, vmax=meta["U_inf"] * CLIM_FACTOR)
    cmap_fn = plt.get_cmap(CMAP)
    fig, ax3d = make_figure(norm, cmap_fn)

    frames_out = []
    t0 = _time.perf_counter()

    for i, fi in enumerate(indices):
        ux, uy, uz, rho = load_frame(frame_files[fi], args.stride)
        img = draw_frame(fig, ax3d, ux, uy, uz, rho, meta, fi, args.stride, norm, cmap_fn)
        frames_out.append(img)

        elapsed = _time.perf_counter() - t0
        rate    = (i + 1) / max(elapsed, 1e-6)
        eta     = (len(indices) - i - 1) / max(rate, 1e-6)
        print(f"\r  [{i+1:3d}/{len(indices)}]  {rate:.1f} fr/s  ETA {eta:.0f}s   ",
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

    elapsed_total = _time.perf_counter() - t0
    print(f"Done: {out}  ({elapsed_total:.1f} s total)")


if __name__ == "__main__":
    main()
