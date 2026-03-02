"""
render_moving.py — 3D 全场体渲染 (Volume Rendering)

从 drone_frames/ 目录读取逐帧完整 3D 流场数据，
使用 PyVista 进行三维体渲染生成视频。

配色参照 SIGGRAPH 流体仿真论文经典风格:
  deep indigo → sapphire → teal → emerald → lime → gold → orange → red → white

不透明度采用 V 形转移函数——自由来流 (u ≈ U_inf) 透明，
流动扰动区 (尾迹/涡/边界层) 渐现，使流场结构自然浮现。

Run:
  python render_moving.py
  python render_moving.py --dir drone_frames --out video.mp4 --stride 2

依赖:
  pip install pyvista imageio[ffmpeg]
"""

import argparse
import glob
import math
import os
import sys
import time
import numpy as np

try:
    import pyvista as pv
except ImportError:
    sys.exit(
        "ERROR: PyVista is required for 3D volume rendering.\n"
        "  pip install pyvista\n"
        "  (GPU volume rendering also needs a working OpenGL driver)"
    )

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import imageio


# ============================================================================
# SIGGRAPH 风格配色  (cool → warm, 深色背景优化)
# ============================================================================

def _hex(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


SIGGRAPH_CMAP = LinearSegmentedColormap.from_list("siggraph_fluid", [
    (0.00, _hex("#0d0829")),
    (0.08, _hex("#1b0c5e")),
    (0.18, _hex("#2b21a0")),
    (0.28, _hex("#1850c8")),
    (0.38, _hex("#0080c0")),
    (0.48, _hex("#00a890")),
    (0.58, _hex("#20c858")),
    (0.68, _hex("#80dc22")),
    (0.78, _hex("#d0d418")),
    (0.88, _hex("#ff8c18")),
    (0.95, _hex("#e83820")),
    (1.00, _hex("#fff0e4")),
], N=256)

try:
    plt.colormaps.register(SIGGRAPH_CMAP, name="siggraph_fluid")
except (ValueError, AttributeError):
    pass


# ============================================================================
# == 渲染参数  (修改此处即可, 无需改动下方逻辑) ==
# ============================================================================

DATA_DIR       = "drone_frames"
VIDEO_FILE     = "clbm_drone_3d_volume.mp4"
VIDEO_FPS      = 10

FRAME_RANGE    = None           # None = 全部;  (start, end) 左闭右开
STRIDE         = 2              # 空间降采样 (1=原始, 2=各轴减半, ...)

WINDOW_SIZE    = (1920, 1080)
BG_COLOR       = "#080818"

BODY_COLOR     = "#c8c8d0"
BODY_OPACITY   = 0.85

OPACITY_POWER  = 0.5            # <1 亚线性 → 放大小扰动可见性
CLIM_FACTOR    = 2.0            # 色标范围 [0, U_inf × CLIM_FACTOR]
OPACITY_MAX    = 0.35           # 体渲染最大不透明度 (防止射线积累过度遮挡)

CAM_POSITION   = None           # None → 自动计算默认相机

SHOW_SCALAR_BAR = True


# ============================================================================
# == 数据加载 ==
# ============================================================================

def load_meta(data_dir: str) -> dict:
    m = np.load(os.path.join(data_dir, "meta.npz"), allow_pickle=False)
    gm = m["grid_meta"]
    return dict(
        Nx=int(gm[0]), Ny=int(gm[1]), Nz=int(gm[2]),
        U_inf=float(gm[3]), Re=float(gm[4]),
        x0=float(gm[5]), y0=float(gm[6]), z0=float(gm[7]),
        fuse_ax=float(gm[8]), fuse_ay=float(gm[9]),
        b_half=float(gm[10]),
        z_wing_start=float(gm[11]), z_wing_end=float(gm[12]),
        iz=int(round(gm[13])), ix=int(round(gm[14])),
        vis_interval=int(gm[15]),
        fuse_az=float(gm[16]) if len(gm) > 16 else 8.0,
        X_template=np.array(m["X_template"]),
        Y_template=np.array(m["Y_template"]),
        xc=np.array(m["xc"]), yc=np.array(m["yc"]),
        theta=np.array(m["theta"]), step=np.array(m["step"]),
    )


def list_frame_files(data_dir: str):
    return sorted(glob.glob(os.path.join(data_dir, "frame_*.npz")))


def load_frame(path: str, stride: int = 1):
    d = np.load(path, allow_pickle=False)
    s = slice(None, None, stride) if stride > 1 else slice(None)
    return (d["ux"][s, s, s].astype(np.float32),
            d["uy"][s, s, s].astype(np.float32),
            d["uz"][s, s, s].astype(np.float32),
            d["rho"][s, s, s].astype(np.float32))


# ============================================================================
# == 几何体构建 ==
# ============================================================================

def build_wing_mesh(xc, yc, theta, X_tmpl, Y_tmpl, z_start, z_end):
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    Xw = xc + X_tmpl * cos_t - Y_tmpl * sin_t
    Yw = yc + X_tmpl * sin_t + Y_tmpl * cos_t
    n_profile = len(Xw)
    nz_layers = 30
    z_vals = np.linspace(z_start, z_end, nz_layers)

    pts = np.zeros((n_profile * nz_layers, 3))
    for i, zv in enumerate(z_vals):
        sl = slice(i * n_profile, (i + 1) * n_profile)
        pts[sl, 0] = Xw
        pts[sl, 1] = Yw
        pts[sl, 2] = zv

    grid = pv.StructuredGrid()
    grid.points = pts
    grid.dimensions = [n_profile, nz_layers, 1]
    return grid.extract_surface()


def build_fuse_mesh(xc, yc, z0, theta, ax, ay, az):
    fuse = pv.ParametricEllipsoid(ax, ay, az)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    T = np.eye(4)
    T[0, 0], T[0, 1] = cos_t, -sin_t
    T[1, 0], T[1, 1] = sin_t, cos_t
    T[0, 3], T[1, 3], T[2, 3] = xc, yc, z0
    return fuse.transform(T, inplace=False)


# ============================================================================
# == 不透明度转移函数 ==
# ============================================================================

def build_opacity_tf(U_inf: float, clim_max: float, n: int = 256):
    """
    V 形不透明度: u ≈ U_inf 处透明, 偏离越远越不透明。
    使自由来流区域消隐, 仅显示尾迹 / 涡 / 驻点等流动特征。
    OPACITY_MAX 限制单体素最大不透明度, 避免射线积累过度遮挡。
    """
    u = np.linspace(0.0, clim_max, n)
    perturbation = np.abs(u - U_inf) / max(U_inf, 1e-12)
    p_max = max(perturbation.max(), 1e-12)
    raw = (perturbation / p_max) ** OPACITY_POWER
    opacity = np.clip(raw, 0.0, 1.0) * OPACITY_MAX
    return opacity


# ============================================================================
# == 单帧渲染 ==
# ============================================================================

def render_one_frame(plotter, frame_path, meta, frame_idx,
                     stride, cam_pos, opacity_tf):
    plotter.clear()

    ux, uy, uz, _ = load_frame(frame_path, stride)
    umag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
    nx, ny, nz = umag.shape

    clim_max = meta["U_inf"] * CLIM_FACTOR

    # region agent log
    if frame_idx < 3:
        import json as _j2, time as _t2
        _perturb = np.abs(umag - meta["U_inf"]) / max(meta["U_inf"], 1e-12)
        _pcts = [float(np.percentile(umag, p)) for p in [0,1,5,25,50,75,95,99,100]]
        _ppcts = [float(np.percentile(_perturb, p)) for p in [0,50,75,90,95,99,100]]
        open("debug-67f16d.log","a").write(_j2.dumps({"sessionId":"67f16d","hypothesisId":"STATS","location":"render_moving.py:render_one_frame","message":"velocity_stats","data":{"frame_idx":frame_idx,"U_inf":meta["U_inf"],"clim_max":clim_max,"umag_min":float(umag.min()),"umag_max":float(umag.max()),"umag_mean":float(umag.mean()),"umag_std":float(umag.std()),"umag_percentiles":_pcts,"perturb_percentiles":_ppcts,"shape":[nx,ny,nz]},"timestamp":int(_t2.time()*1000)})+"\n")
    # endregion

    vol = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=(float(stride),) * 3,
    )
    vol.point_data["velocity"] = umag.ravel(order="F")

    plotter.add_volume(
        vol, scalars="velocity",
        clim=[0, clim_max],
        cmap=SIGGRAPH_CMAP,
        opacity="linear",
        shade=False,
        show_scalar_bar=False,
    )

    # region agent log
    if frame_idx < 2:
        import json as _j3, time as _t3
        _otf_vals = [float(v) for v in opacity_tf[::32]]
        plotter.camera_position = cam_pos
        _test_img = plotter.screenshot(return_img=True)
        _px_mean = float(_test_img.mean())
        _px_max = float(_test_img.max())
        _nonzero = int(np.count_nonzero(_test_img.sum(axis=2)))
        _total = _test_img.shape[0] * _test_img.shape[1]
        open("debug-67f16d.log","a").write(_j3.dumps({"sessionId":"67f16d","hypothesisId":"H-C-D-E","location":"render_moving.py:after_add_volume","message":"volume_render_check","data":{"frame_idx":frame_idx,"opacity_tf_sample":_otf_vals,"img_shape":list(_test_img.shape),"px_mean":_px_mean,"px_max":_px_max,"nonzero_pixels":_nonzero,"total_pixels":_total,"pct_nonzero":round(_nonzero/_total*100,2),"n_actors":len(plotter.actors)},"timestamp":int(_t3.time()*1000)})+"\n")
    # endregion

    xc    = float(meta["xc"][frame_idx])
    yc    = float(meta["yc"][frame_idx])
    theta = float(meta["theta"][frame_idx])
    z0    = meta["z0"]

    wing = build_wing_mesh(
        xc, yc, theta,
        meta["X_template"], meta["Y_template"],
        meta["z_wing_start"], meta["z_wing_end"],
    )
    plotter.add_mesh(wing, color=BODY_COLOR, opacity=BODY_OPACITY,
                     smooth_shading=True)

    try:
        fuse = build_fuse_mesh(
            xc, yc, z0, theta,
            meta["fuse_ax"], meta["fuse_ay"], meta["fuse_az"],
        )
        plotter.add_mesh(fuse, color=BODY_COLOR, opacity=BODY_OPACITY,
                         smooth_shading=True)
    except Exception:
        pass

    step_num = int(meta["step"][frame_idx])
    # region agent log
    import json as _json, time as _time; open("debug-67f16d.log","a").write(_json.dumps({"sessionId":"67f16d","hypothesisId":"A-B","location":"render_moving.py:add_text","message":"before add_text and scalar_bar","data":{"step_num":step_num,"frame_idx":frame_idx},"timestamp":int(_time.time()*1000)})+"\n")
    # endregion
    plotter.add_text(
        f"step {step_num}", position="upper_left",
        font_size=14, color="white",
    )

    if SHOW_SCALAR_BAR:
        plotter.add_scalar_bar(
            title="Velocity  |u|",
            color="white", n_labels=5,
            italic=False, fmt="%.2f",
            width=0.3, height=0.05,
            position_x=0.35, position_y=0.02,
        )

    plotter.camera_position = cam_pos
    return plotter.screenshot(return_img=True)


# ============================================================================
# == Main ==
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="3D volume rendering of drone flow field (SIGGRAPH style)")
    parser.add_argument("--dir",    default=DATA_DIR,
                        help="Frame data directory")
    parser.add_argument("--out",    default=VIDEO_FILE,
                        help="Output video file (.mp4)")
    parser.add_argument("--fps",    type=int, default=VIDEO_FPS)
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help="Spatial subsampling (1=full, 2=half each axis)")
    args = parser.parse_args()

    print("=" * 65)
    print("3D Volume Rendering  —  SIGGRAPH-style colormap")
    print("=" * 65)

    print(f"Loading metadata: {args.dir}/meta.npz ...")
    meta = load_meta(args.dir)
    Nx, Ny, Nz = meta["Nx"], meta["Ny"], meta["Nz"]
    U_inf = meta["U_inf"]
    print(f"  Grid     : {Nx} × {Ny} × {Nz}")
    print(f"  U_inf={U_inf}  Re={meta['Re']}")

    frame_files = list_frame_files(args.dir)
    n_total = len(frame_files)
    print(f"  Frames   : {n_total}")

    s, e = 0, n_total
    if FRAME_RANGE is not None:
        s, e = FRAME_RANGE[0], min(FRAME_RANGE[1], n_total)
    indices = list(range(s, e))
    print(f"  Rendering: {len(indices)} frames   stride={args.stride}")

    if args.stride > 1:
        eff = (Nx // args.stride, Ny // args.stride, Nz // args.stride)
        print(f"  Effective: {eff[0]} × {eff[1]} × {eff[2]} per frame")

    clim_max   = U_inf * CLIM_FACTOR
    opacity_tf = build_opacity_tf(U_inf, clim_max)

    pv.global_theme.background = BG_COLOR
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)

    cam_pos = CAM_POSITION
    if cam_pos is None:
        cam_pos = [
            (Nx * 1.1,  -Ny * 1.8,  Nz * 2.2),
            (Nx * 0.35,  Ny * 0.5,  Nz * 0.5),
            (0, 1, 0),
        ]

    frames_out = []
    t0 = time.perf_counter()

    for i, idx in enumerate(indices):
        img = render_one_frame(
            plotter, frame_files[idx], meta,
            idx, args.stride, cam_pos, opacity_tf,
        )
        frames_out.append(img)

        if (i + 1) % 5 == 0 or (i + 1) == len(indices):
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / max(elapsed, 0.01)
            eta  = (len(indices) - i - 1) / max(rate, 0.01)
            print(f"\r  {i+1}/{len(indices)}  "
                  f"({rate:.1f} fps, ETA {eta:.0f}s)   ",
                  end="", flush=True)
    print()

    plotter.close()

    out = args.out
    print(f"Writing -> {out}  ({len(frames_out)} frames @ {args.fps} fps) ...")
    try:
        imageio.mimwrite(
            out, frames_out, fps=args.fps,
            format="ffmpeg", quality=8, macro_block_size=1,
        )
    except Exception as exc:
        gif_out = out.rsplit(".", 1)[0] + ".gif"
        print(f"MP4 failed ({exc}), fallback GIF: {gif_out}")
        imageio.mimsave(gif_out, frames_out, duration=1.0 / args.fps)
        out = gif_out

    print(f"Done: {out}")


if __name__ == "__main__":
    main()
