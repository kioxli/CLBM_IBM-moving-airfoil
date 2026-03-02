"""
airfoil_moving.py — IBM direct-forcing for 3D simplified drone model.

The drone (main wing + fuselage + wing-tip caps) undergoes heave + pitch
prescribed motion in the x-y plane.  All Lagrangian markers (wing cross-
sections, tip-cap interior fill, fuselage ellipsoid) are stored as a single
flat array; _update_markers rotates/translates the full set simultaneously and
computes the rigid-body velocity at every marker.  z coordinates are fixed
(pitch is a rotation about the spanwise z-axis, so z is invariant).

Import this module AFTER ti.init() has been called.
"""

import taichi as ti
import numpy as np

from config_moving import (Nx, Ny, Nz,
                            N_MARKERS_2D, N_MARKERS,
                            X_marker_np, Y_marker_np, Z_marker_np)

N = N_MARKERS   # Python int used in kernel range bounds


# ---------------------------------------------------------------------------
# Peskin 4-point delta kernel
# ---------------------------------------------------------------------------
@ti.func
def _phi1d(r: ti.f32) -> ti.f32:
    """Scalar Peskin 4-point regularised delta function.  Support: |r| < 2."""
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
# DroneMovingIBM
# ---------------------------------------------------------------------------
@ti.data_oriented
class DroneMovingIBM:
    """
    Immersed boundary method (direct forcing) for a simplified 3D drone model
    undergoing prescribed heave + pitch motion.

    The drone body is represented by a flat list of N_MARKERS Lagrangian
    markers covering:
      - Main wing   : NACA 0012 cross-sections at each spanwise z layer
      - Wing tips   : interior fill at the two tip planes
      - Fuselage    : ellipsoid surface markers centred at the pivot

    Pitch is a rotation in the x-y plane (about the spanwise z-axis); z
    coordinates of markers are therefore invariant.

    Public Taichi fields:
        fx, fy, fz         : (Nx, Ny, Nz)  Eulerian body-force density
        X_marker, Y_marker, Z_marker : (N,)  current marker positions
    """

    def __init__(self):
        self.N = N

        # Template in local body frame (populated from config arrays)
        self.X0 = ti.field(dtype=ti.f32, shape=(N,))
        self.Y0 = ti.field(dtype=ti.f32, shape=(N,))
        self.Z0 = ti.field(dtype=ti.f32, shape=(N,))
        self.X0.from_numpy(X_marker_np)
        self.Y0.from_numpy(Y_marker_np)
        self.Z0.from_numpy(Z_marker_np)   # absolute z (fixed for all steps)

        # Current world positions (updated each step)
        self.X_marker = ti.field(dtype=ti.f32, shape=(N,))
        self.Y_marker = ti.field(dtype=ti.f32, shape=(N,))
        self.Z_marker = ti.field(dtype=ti.f32, shape=(N,))

        # Rigid-body velocity at each marker (x-y only; z motion = 0)
        self.VX_body = ti.field(dtype=ti.f32, shape=(N,))
        self.VY_body = ti.field(dtype=ti.f32, shape=(N,))

        # Interpolated fluid velocity at markers
        self.U_k = ti.field(dtype=ti.f32, shape=(N,))
        self.V_k = ti.field(dtype=ti.f32, shape=(N,))

        # Per-marker IBM force (z component always zero)
        self.FX_k = ti.field(dtype=ti.f32, shape=(N,))
        self.FY_k = ti.field(dtype=ti.f32, shape=(N,))

        # Eulerian body-force arrays (re-zeroed each step)
        self.fx = ti.field(dtype=ti.f32, shape=(Nx, Ny, Nz))
        self.fy = ti.field(dtype=ti.f32, shape=(Nx, Ny, Nz))
        self.fz = ti.field(dtype=ti.f32, shape=(Nx, Ny, Nz))   # always zero

    # ------------------------------------------------------------------
    # Update marker positions and rigid-body velocities
    # ------------------------------------------------------------------
    @ti.kernel
    def _update_markers(self,
                        xc: ti.f32, yc: ti.f32, theta: ti.f32,
                        vx: ti.f32, vy: ti.f32, omega: ti.f32):
        """
        Rotate every marker's local-frame (X0, Y0) by theta and translate to
        (xc, yc); z is unchanged (pitch is about the z-axis).

            x_k = xc + X0*cos(θ) - Y0*sin(θ)
            y_k = yc + X0*sin(θ) + Y0*cos(θ)
            z_k = Z0  (absolute z, fixed)
            vx_body = vx - ω*(y_k - yc)
            vy_body = vy + ω*(x_k - xc)
        """
        cos_t = ti.cos(theta)
        sin_t = ti.sin(theta)
        for m in range(self.N):
            x0m = self.X0[m]
            y0m = self.Y0[m]
            xk  = xc + x0m * cos_t - y0m * sin_t
            yk  = yc + x0m * sin_t + y0m * cos_t
            self.X_marker[m] = xk
            self.Y_marker[m] = yk
            self.Z_marker[m] = self.Z0[m]
            rx = xk - xc
            ry = yk - yc
            self.VX_body[m] = vx - omega * ry
            self.VY_body[m] = vy + omega * rx

    # ------------------------------------------------------------------
    # Clear Eulerian forces
    # ------------------------------------------------------------------
    @ti.kernel
    def _zero_forces(self):
        for i, j, k in ti.ndrange(Nx, Ny, Nz):
            self.fx[i, j, k] = 0.0
            self.fy[i, j, k] = 0.0
            self.fz[i, j, k] = 0.0

    # ------------------------------------------------------------------
    # Velocity interpolation  Eulerian → Lagrangian  (4×4×4 stencil)
    # ------------------------------------------------------------------
    @ti.kernel
    def _interpolate(self, ux: ti.template(), uy: ti.template()):
        """
        Interpolate fluid velocity to each marker via Peskin 4-point kernel.
        x: clamped (non-periodic);  y: periodic;  z: periodic.
        """
        for m in range(self.N):
            xk     = self.X_marker[m]
            yk     = self.Y_marker[m]
            zk     = self.Z_marker[m]
            i_base = ti.cast(ti.floor(xk), ti.i32) - 1
            j_base = ti.cast(ti.floor(yk), ti.i32) - 1
            k_base = ti.cast(ti.floor(zk), ti.i32) - 1
            u_sum  = 0.0
            v_sum  = 0.0
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

    # ------------------------------------------------------------------
    # Marker forces: drive fluid to rigid-body velocity
    # ------------------------------------------------------------------
    @ti.kernel
    def _marker_forces(self):
        """F_k = −(U_k − V_body), clamped.  No z-direction force."""
        f_max = ti.static(0.05)
        for m in range(self.N):
            err_x = -(self.U_k[m] - self.VX_body[m])
            err_y = -(self.V_k[m] - self.VY_body[m])
            self.FX_k[m] = ti.max(-f_max, ti.min(f_max, err_x))
            self.FY_k[m] = ti.max(-f_max, ti.min(f_max, err_y))

    # ------------------------------------------------------------------
    # Force spreading  Lagrangian → Eulerian  (4×4×4 stencil, atomic)
    # ------------------------------------------------------------------
    @ti.kernel
    def _spread(self):
        """
        Spread marker forces to Eulerian grid via Peskin 4-point kernel.
        fz is not spread (motion is purely in x-y).
        """
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

    # ------------------------------------------------------------------
    # Global Eulerian force clamp
    # ------------------------------------------------------------------
    @ti.kernel
    def _clip_forces(self):
        f_max = ti.static(0.05)
        for i, j, k in ti.ndrange(Nx, Ny, Nz):
            self.fx[i, j, k] = ti.max(-f_max, ti.min(f_max, self.fx[i, j, k]))
            self.fy[i, j, k] = ti.max(-f_max, ti.min(f_max, self.fy[i, j, k]))
            self.fz[i, j, k] = 0.0

    # ------------------------------------------------------------------
    # Public: one IBM step
    # ------------------------------------------------------------------
    def compute_force(self, ux, uy,
                      xc: float, yc: float, theta: float,
                      vx: float, vy: float, omega: float):
        """
        Update marker positions, interpolate fluid velocity, compute IBM
        forces, and spread to the Eulerian grid.

        Parameters
        ----------
        ux, uy : ti.field(f32, shape=(Nx, Ny, Nz))   fluid velocity fields
        xc, yc : float   current pivot position
        theta  : float   current pitch angle (rad)
        vx, vy : float   translational velocity of pivot
        omega  : float   angular velocity (rad/step, + = CCW)

        Returns
        -------
        self.fx, self.fy, self.fz : Taichi fields ready for lbm.step()
        """
        self._update_markers(xc, yc, theta, vx, vy, omega)
        self._zero_forces()
        self._interpolate(ux, uy)
        self._marker_forces()
        self._spread()
        self._clip_forces()
        return self.fx, self.fy, self.fz
