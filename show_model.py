"""
show_model.py — 3D visualization of the simplified airplane model markers.

Reads geometry arrays from config_moving.py and produces a 4-panel figure:
  - 3D scatter (colour-coded by component)
  - Top view   (x-z plane, looking down)
  - Side view  (x-y plane, looking from starboard)
  - Front view (y-z plane, looking forward)

Saves: airplane_model.png
Usage: python show_model.py
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import numpy as np

from config_moving import (
    # component marker arrays (LOCAL X/Y, absolute Z)
    X_wing,  Y_wing,  Z_wing,
    X_tip,   Y_tip,   Z_tip,
    X_fuse,  Y_fuse,  Z_fuse,
    X_htail, Y_htail, Z_htail,
    X_htail_tip, Y_htail_tip, Z_htail_tip,
    X_vtail, Y_vtail, Z_vtail,
    X_vtail_tip, Y_vtail_tip, Z_vtail_tip,
    # pivot (world position at theta=0, heave=0)
    x0, y0, z0,
    # key dimensions
    c, b_half, fuse_ax, fuse_ay, x_tail_arm,
    c_htail, b_half_htail, c_vtail, h_vtail,
    N_MARKERS,
)

# ---------------------------------------------------------------------------
# Convert LOCAL (X0, Y0) + absolute Z0  →  world (x, y, z) at theta=0, heave=0
# ---------------------------------------------------------------------------
def to_world(Xloc, Yloc, Zabs):
    return Xloc + x0, Yloc + y0, Zabs

comps = {
    "Main wing":    to_world(X_wing,      Y_wing,      Z_wing),
    "Wing tips":    to_world(X_tip,       Y_tip,       Z_tip),
    "Fuselage":     to_world(X_fuse,      Y_fuse,      Z_fuse),
    "H-stab":       to_world(X_htail,     Y_htail,     Z_htail),
    "H-stab tips":  to_world(X_htail_tip, Y_htail_tip, Z_htail_tip),
    "V-fin":        to_world(X_vtail,     Y_vtail,     Z_vtail),
    "V-fin tip":    to_world(X_vtail_tip, Y_vtail_tip, Z_vtail_tip),
}

COLORS = {
    "Main wing":   "#2166ac",   # deep blue
    "Wing tips":   "#74add1",   # light blue
    "Fuselage":    "#f46d43",   # orange-red
    "H-stab":      "#1a9850",   # green
    "H-stab tips": "#66bd63",   # light green
    "V-fin":       "#d73027",   # red
    "V-fin tip":   "#f99f59",   # salmon
}

# thin-out factor (plot 1 in N pts for 3D scatter, keep all for 2D projections)
DECIMATE_3D = 8

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#1a1a2e")

ax3d = fig.add_subplot(2, 2, 1, projection="3d")
ax_top  = fig.add_subplot(2, 2, 2)   # x-z  (top view, looking -y)
ax_side = fig.add_subplot(2, 2, 3)   # x-y  (side view, looking +z)
ax_front = fig.add_subplot(2, 2, 4)  # y-z  (front view, looking +x)

for ax in [ax_top, ax_side, ax_front]:
    ax.set_facecolor("#0d1117")

ax3d.set_facecolor("#0d1117")
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False

for name, (wx, wy, wz) in comps.items():
    col = COLORS[name]
    s   = 0.4

    # 3D (decimated)
    idx = np.arange(0, len(wx), DECIMATE_3D)
    ax3d.scatter(wx[idx], wz[idx], wy[idx],
                 c=col, s=s, alpha=0.6, linewidths=0)

    # Top view: x (streamwise) vs z (spanwise)
    ax_top.scatter(wx, wz, c=col, s=s * 0.5, alpha=0.5, linewidths=0)

    # Side view: x (streamwise) vs y (vertical)
    ax_side.scatter(wx, wy, c=col, s=s * 0.5, alpha=0.5, linewidths=0)

    # Front view: z (spanwise) vs y (vertical)
    ax_front.scatter(wz, wy, c=col, s=s * 0.5, alpha=0.5, linewidths=0)

# ---------------------------------------------------------------------------
# 3D axes
# ---------------------------------------------------------------------------
ax3d.set_xlabel("x (streamwise)", color="white", fontsize=8)
ax3d.set_ylabel("z (spanwise)",   color="white", fontsize=8)
ax3d.set_zlabel("y (vertical)",   color="white", fontsize=8)
ax3d.tick_params(colors="white", labelsize=7)
ax3d.set_title("3D View (theta=0)", color="white", fontsize=10)
ax3d.view_init(elev=20, azim=-60)

# equal-ish aspect for 3D
all_wx = np.concatenate([v[0] for v in comps.values()])
all_wy = np.concatenate([v[1] for v in comps.values()])
all_wz = np.concatenate([v[2] for v in comps.values()])
xmid = (all_wx.max() + all_wx.min()) / 2
ymid = (all_wy.max() + all_wy.min()) / 2
zmid = (all_wz.max() + all_wz.min()) / 2
half = max(all_wx.max() - all_wx.min(),
           all_wy.max() - all_wy.min(),
           all_wz.max() - all_wz.min()) / 2 + 10
ax3d.set_xlim(xmid - half, xmid + half)
ax3d.set_zlim(ymid - half * 0.5, ymid + half * 0.5)
ax3d.set_ylim(zmid - half, zmid + half)

# ---------------------------------------------------------------------------
# 2D projection axes
# ---------------------------------------------------------------------------
def style_ax(ax, xlabel, ylabel, title):
    ax.set_facecolor("#0d1117")
    ax.set_xlabel(xlabel, color="white", fontsize=9)
    ax.set_ylabel(ylabel, color="white", fontsize=9)
    ax.set_title(title,   color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#222", linewidth=0.4)

style_ax(ax_top,   "x  (streamwise, lu)", "z  (spanwise, lu)",  "Top view  (x-z)")
style_ax(ax_side,  "x  (streamwise, lu)", "y  (vertical, lu)",  "Side view  (x-y)")
style_ax(ax_front, "z  (spanwise, lu)",   "y  (vertical, lu)",  "Front view  (y-z)")

# Domain outlines on projections
def rect(ax, x0_, y0_, w, h, **kw):
    ax.add_patch(mpatches.Rectangle((x0_, y0_), w, h,
                 fill=False, linewidth=0.6, **kw))

# top: domain outline (x in [0,Nx], z in [0,Nz])
from config_moving import Nx, Ny, Nz
rect(ax_top,  0, 0, Nx, Nz,  edgecolor="#555", linestyle="--")
rect(ax_side, 0, 0, Nx, Ny,  edgecolor="#555", linestyle="--")
rect(ax_front, 0, 0, Nz, Ny, edgecolor="#555", linestyle="--")

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
patches = [mpatches.Patch(color=COLORS[n], label=n) for n in COLORS]
fig.legend(handles=patches, loc="lower center", ncol=4,
           framealpha=0.3, labelcolor="white", fontsize=8,
           facecolor="#1a1a2e", edgecolor="#444")

fig.suptitle(
    f"Simplified Airplane Model — {N_MARKERS} markers\n"
    f"Wing: c={c} lu, span={2*b_half} lu  |  "
    f"Fuse: {2*fuse_ax}x{2*fuse_ay}x{2*fuse_ay} lu  |  "
    f"H-stab: c={c_htail} lu, span={2*b_half_htail} lu  |  "
    f"V-fin: c={c_vtail} lu, h={h_vtail} lu  |  arm={x_tail_arm} lu",
    color="white", fontsize=9, y=0.98
)

plt.tight_layout(rect=[0, 0.06, 1, 0.96])

out = "airplane_model.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
