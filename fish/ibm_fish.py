"""
ibm_fish.py — IBM direct-forcing for the 3D bio-inspired fish swimmer.

The ellipse body undergoes prescribed heave + pitch in x-y, extruded uniformly
in z (periodic). Logic identical to airfoil_moving.py; only the config import
is changed to config_fish.

Import AFTER ti.init() has been called.
"""

import taichi as ti
import numpy as np

from config_fish import (Nx, Ny, Nz,
                         N_MARKERS_2D, N_MARKERS,
                         X_marker_np, Y_marker_np, Z_marker_np)

N = N_MARKERS


# ---------------------------------------------------------------------------
# Peskin 4-point delta kernel
# ---------------------------------------------------------------------------
@ti.func
def _phi1d(r: ti.f32) -> ti.f32:
    s      = ti.abs(r)
    result = 0.0
    if s < 1.0:
        result = (3.0 - 2.0 * s
                  + ti.sqrt(ti.max(1.0 + 4.0 * s - 4.0 * s * s, 0.0))) / 8.0
    elif s < 2.0:
        result = (5.0 - 2.0 * s
                  - ti.sqrt(ti.max(-7.0 + 12.0 * s - 4.0 * s * s, 0.0))) / 8.0
    return result


# ---------------------------------------------------------------------------
# FishIBM
# ---------------------------------------------------------------------------
@ti.data_oriented
class FishIBM:
    """
    Immersed boundary method (direct forcing) for the ellipse fish swimmer.
    Motion is purely in x-y; z-direction body force is always zero.
    """

    def __init__(self):
        self.N = N

        self.X0 = ti.field(dtype=ti.f32, shape=(N,))
        self.Y0 = ti.field(dtype=ti.f32, shape=(N,))
        self.Z0 = ti.field(dtype=ti.f32, shape=(N,))
        self.X0.from_numpy(X_marker_np)
        self.Y0.from_numpy(Y_marker_np)
        self.Z0.from_numpy(Z_marker_np)

        self.X_marker = ti.field(dtype=ti.f32, shape=(N,))
        self.Y_marker = ti.field(dtype=ti.f32, shape=(N,))
        self.Z_marker = ti.field(dtype=ti.f32, shape=(N,))

        self.VX_body = ti.field(dtype=ti.f32, shape=(N,))
        self.VY_body = ti.field(dtype=ti.f32, shape=(N,))

        self.U_k  = ti.field(dtype=ti.f32, shape=(N,))
        self.V_k  = ti.field(dtype=ti.f32, shape=(N,))

        self.FX_k = ti.field(dtype=ti.f32, shape=(N,))
        self.FY_k = ti.field(dtype=ti.f32, shape=(N,))

        self.fx = ti.field(dtype=ti.f32, shape=(Nx, Ny, Nz))
        self.fy = ti.field(dtype=ti.f32, shape=(Nx, Ny, Nz))
        self.fz = ti.field(dtype=ti.f32, shape=(Nx, Ny, Nz))

    @ti.kernel
    def _update_markers(self,
                        xc: ti.f32, yc: ti.f32, theta: ti.f32,
                        vx: ti.f32, vy: ti.f32, omega: ti.f32):
        cos_t = ti.cos(theta)
        sin_t = ti.sin(theta)
        for m in range(self.N):
            x0m = self.X0[m];  y0m = self.Y0[m]
            xk  = xc + x0m * cos_t - y0m * sin_t
            yk  = yc + x0m * sin_t + y0m * cos_t
            self.X_marker[m] = xk
            self.Y_marker[m] = yk
            self.Z_marker[m] = self.Z0[m]
            rx = xk - xc;  ry = yk - yc
            self.VX_body[m] = vx - omega * ry
            self.VY_body[m] = vy + omega * rx

    @ti.kernel
    def _zero_forces(self):
        for i, j, k in ti.ndrange(Nx, Ny, Nz):
            self.fx[i, j, k] = 0.0
            self.fy[i, j, k] = 0.0
            self.fz[i, j, k] = 0.0

    @ti.kernel
    def _interpolate(self, ux: ti.template(), uy: ti.template()):
        for m in range(self.N):
            xk     = self.X_marker[m]
            yk     = self.Y_marker[m]
            zk     = self.Z_marker[m]
            i_base = ti.cast(ti.floor(xk), ti.i32) - 1
            j_base = ti.cast(ti.floor(yk), ti.i32) - 1
            k_base = ti.cast(ti.floor(zk), ti.i32) - 1
            u_sum  = 0.0;  v_sum = 0.0
            for di in ti.static(range(4)):
                ii  = ti.max(0, ti.min(Nx - 1, i_base + di))
                phx = _phi1d(ti.cast(ii, ti.f32) - xk)
                for dj in ti.static(range(4)):
                    jj  = (j_base + dj + Ny) % Ny
                    phy = _phi1d(ti.cast(jj, ti.f32) - yk)
                    for dk in ti.static(range(4)):
                        kk   = (k_base + dk + Nz) % Nz
                        phz  = _phi1d(ti.cast(kk, ti.f32) - zk)
                        wijk = phx * phy * phz
                        u_sum += ux[ii, jj, kk] * wijk
                        v_sum += uy[ii, jj, kk] * wijk
            self.U_k[m] = u_sum
            self.V_k[m] = v_sum

    @ti.kernel
    def _marker_forces(self):
        f_max = ti.static(0.05)
        for m in range(self.N):
            err_x = -(self.U_k[m] - self.VX_body[m])
            err_y = -(self.V_k[m] - self.VY_body[m])
            self.FX_k[m] = ti.max(-f_max, ti.min(f_max, err_x))
            self.FY_k[m] = ti.max(-f_max, ti.min(f_max, err_y))

    @ti.kernel
    def _spread(self):
        for m in range(self.N):
            xk     = self.X_marker[m]
            yk     = self.Y_marker[m]
            zk     = self.Z_marker[m]
            fxk    = self.FX_k[m]
            fyk    = self.FY_k[m]
            i_base = ti.cast(ti.floor(xk), ti.i32) - 1
            j_base = ti.cast(ti.floor(yk), ti.i32) - 1
            k_base = ti.cast(ti.floor(zk), ti.i32) - 1
            for di in ti.static(range(4)):
                ii  = ti.max(0, ti.min(Nx - 1, i_base + di))
                phx = _phi1d(ti.cast(ii, ti.f32) - xk)
                for dj in ti.static(range(4)):
                    jj  = (j_base + dj + Ny) % Ny
                    phy = _phi1d(ti.cast(jj, ti.f32) - yk)
                    for dk in ti.static(range(4)):
                        kk   = (k_base + dk + Nz) % Nz
                        phz  = _phi1d(ti.cast(kk, ti.f32) - zk)
                        wijk = phx * phy * phz
                        ti.atomic_add(self.fx[ii, jj, kk], fxk * wijk)
                        ti.atomic_add(self.fy[ii, jj, kk], fyk * wijk)

    @ti.kernel
    def _clip_forces(self):
        f_max = ti.static(0.05)
        for i, j, k in ti.ndrange(Nx, Ny, Nz):
            self.fx[i, j, k] = ti.max(-f_max, ti.min(f_max, self.fx[i, j, k]))
            self.fy[i, j, k] = ti.max(-f_max, ti.min(f_max, self.fy[i, j, k]))
            self.fz[i, j, k] = 0.0

    def compute_force(self, ux, uy,
                      xc: float, yc: float, theta: float,
                      vx: float, vy: float, omega: float):
        self._update_markers(xc, yc, theta, vx, vy, omega)
        self._zero_forces()
        self._interpolate(ux, uy)
        self._marker_forces()
        self._spread()
        self._clip_forces()
        return self.fx, self.fy, self.fz
