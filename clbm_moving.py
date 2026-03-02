"""
clbm_moving.py — Cascaded LBM (CLBM) D3Q19 solver for the moving-airfoil case.

Identical in structure to clbm.py; only the import source differs
(config_moving instead of config) so that the fixed-airfoil code is
not disturbed.

Import this module AFTER ti.init() has been called in main_moving.py.
"""

import math
import taichi as ti
import numpy as np

from config_moving import (Nx, Ny, Nz, cs2, s_v, s_e, s_q, s_pi,
                            U_inf, AoA0_deg, M_raw_inv_np)

# ---------------------------------------------------------------------------
# D3Q19 lattice constants  (Python tuples → ti.static unrolled at compile time)
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

# Pre-computed inlet velocity components (captured by Taichi JIT as literals)
_AoA_rad = math.radians(AoA0_deg)
_UX_IN   = float(U_inf * math.cos(_AoA_rad))
_UY_IN   = float(-U_inf * math.sin(_AoA_rad))


# ---------------------------------------------------------------------------
# CLBMSolver
# ---------------------------------------------------------------------------
@ti.data_oriented
class CLBMSolver:
    """
    CLBM D3Q19 fluid solver on an Nx × Ny × Nz lattice.

    Public Taichi fields (accessible by other modules):
        f   : (Nx, Ny, Nz, 19)  distribution functions
        rho : (Nx, Ny, Nz)      density
        ux  : (Nx, Ny, Nz)      x-velocity
        uy  : (Nx, Ny, Nz)      y-velocity
        uz  : (Nx, Ny, Nz)      z-velocity
    """

    def __init__(self):
        # --- Distribution function (double-buffer) ---
        self.f      = ti.field(dtype=ti.f32, shape=(Nx, Ny, Nz, 19))
        self.f_post = ti.field(dtype=ti.f32, shape=(Nx, Ny, Nz, 19))

        # --- Macroscopic fields ---
        self.rho = ti.field(dtype=ti.f32, shape=(Nx, Ny, Nz))
        self.ux  = ti.field(dtype=ti.f32, shape=(Nx, Ny, Nz))
        self.uy  = ti.field(dtype=ti.f32, shape=(Nx, Ny, Nz))
        self.uz  = ti.field(dtype=ti.f32, shape=(Nx, Ny, Nz))

        # --- M_raw_inv stored as 2D field (avoids slow ti.Matrix(19x19) unroll) ---
        self.M_inv = ti.field(dtype=ti.f32, shape=(19, 19))
        self.M_inv.from_numpy(M_raw_inv_np)

        # Initialise f to Maxwell–Boltzmann equilibrium at free-stream velocity
        self._init_feq()

    # ------------------------------------------------------------------
    # Initialisation kernel
    # ------------------------------------------------------------------
    @ti.kernel
    def _init_feq(self):
        ux0 = ti.static(_UX_IN)
        uy0 = ti.static(_UY_IN)
        u2  = ux0 * ux0 + uy0 * uy0
        for i, j, k in ti.ndrange(Nx, Ny, Nz):
            for q in ti.static(range(19)):
                eu = EX[q] * ux0 + EY[q] * uy0   # uz0 = 0
                self.f[i, j, k, q] = W[q] * (
                    1.0 + eu / cs2
                    + 0.5 * eu * eu / (cs2 * cs2)
                    - 0.5 * u2 / cs2
                )
            self.rho[i, j, k] = 1.0
            self.ux[i, j, k]  = ux0
            self.uy[i, j, k]  = uy0
            self.uz[i, j, k]  = 0.0

    # ------------------------------------------------------------------
    # Macroscopic quantities WITH Guo half-force correction
    # ------------------------------------------------------------------
    @ti.kernel
    def _macro_with_force(self,
                          fx: ti.template(),
                          fy: ti.template(),
                          fz: ti.template()):
        for i, j, k in ti.ndrange(Nx, Ny, Nz):
            r  = 0.0
            mx = 0.0
            my = 0.0
            mz = 0.0
            for q in ti.static(range(19)):
                fq  = self.f[i, j, k, q]
                r  += fq
                mx += EX[q] * fq
                my += EY[q] * fq
                mz += EZ[q] * fq
            self.rho[i, j, k] = r
            self.ux[i, j, k]  = mx / r + 0.5 * fx[i, j, k] / r
            self.uy[i, j, k]  = my / r + 0.5 * fy[i, j, k] / r
            self.uz[i, j, k]  = mz / r + 0.5 * fz[i, j, k] / r

    # ------------------------------------------------------------------
    # CLBM collision kernel  (central-moment cascade + Guo body force)
    # ------------------------------------------------------------------
    @ti.kernel
    def _collide(self,
                 fx: ti.template(),
                 fy: ti.template(),
                 fz: ti.template()):
        """
        Per-cell (i,j,k): compute 3D central moments, cascade-relax,
        back-transform via M_inv, add Guo correction → f_post.
        """
        for i, j, k in ti.ndrange(Nx, Ny, Nz):
            u  = self.ux[i, j, k]
            v  = self.uy[i, j, k]
            w  = self.uz[i, j, k]
            r  = self.rho[i, j, k]
            Fx = fx[i, j, k]
            Fy = fy[i, j, k]
            Fz = fz[i, j, k]

            # ── Step 1: central moments ────────────────────────────────
            k110 = 0.0; k101 = 0.0; k011 = 0.0
            k200 = 0.0; k020 = 0.0; k002 = 0.0
            k210 = 0.0; k120 = 0.0
            k201 = 0.0; k021 = 0.0
            k102 = 0.0; k012 = 0.0
            k220 = 0.0; k202 = 0.0; k022 = 0.0

            for q in ti.static(range(19)):
                fq   = self.f[i, j, k, q]
                xix  = EX[q] - u
                xiy  = EY[q] - v
                xiz  = EZ[q] - w
                xix2 = xix * xix
                xiy2 = xiy * xiy
                xiz2 = xiz * xiz
                k110 += fq * xix  * xiy
                k101 += fq * xix  * xiz
                k011 += fq * xiy  * xiz
                k200 += fq * xix2
                k020 += fq * xiy2
                k002 += fq * xiz2
                k210 += fq * xix2 * xiy
                k120 += fq * xix  * xiy2
                k201 += fq * xix2 * xiz
                k021 += fq * xiy2 * xiz
                k102 += fq * xix  * xiz2
                k012 += fq * xiy  * xiz2
                k220 += fq * xix2 * xiy2
                k202 += fq * xix2 * xiz2
                k022 += fq * xiy2 * xiz2

            # ── Step 2: cascade relaxation ─────────────────────────────
            # Shear moments (s_v)
            k110s = k110 * (1.0 - s_v)
            k101s = k101 * (1.0 - s_v)
            k011s = k011 * (1.0 - s_v)

            # Normal-stress combination: trace (s_e) + deviatoric (s_v)
            kA    = k200 + k020 + k002
            kA_eq = r                        # Tr(second moment) = ρ at eq.
            kD    = k200 - k020
            kE    = k200 - k002
            kAs   = kA - s_e * (kA - kA_eq) # s_e=1 → kAs = r
            kDs   = kD * (1.0 - s_v)
            kEs   = kE * (1.0 - s_v)
            k200s = (kAs + kDs + kEs) / 3.0
            k020s = (kAs - 2.0 * kDs + kEs) / 3.0  # fixed sign
            k002s = (kAs + kDs - 2.0 * kEs) / 3.0

            # Heat-flux / third-order (s_q = 1 → all zero)
            k210s = 0.0; k120s = 0.0
            k201s = 0.0; k021s = 0.0
            k102s = 0.0; k012s = 0.0

            # Fourth-order (s_pi = 1 → relax to equilibrium r/9)
            k220s = r / 9.0
            k202s = r / 9.0
            k022s = r / 9.0

            # ── Step 3: central → raw moments T ───────────────────────
            T000 = r
            T100 = u * r
            T010 = v * r
            T001 = w * r
            T110 = k110s + u * v * r
            T101 = k101s + u * w * r
            T011 = k011s + v * w * r
            T200 = k200s + u * u * r
            T020 = k020s + v * v * r
            T002 = k002s + w * w * r
            T210 = k210s + 2.0*u*k110s + v*k200s + u*u*v*r
            T120 = k120s + 2.0*v*k110s + u*k020s + u*v*v*r
            T201 = k201s + 2.0*u*k101s + w*k200s + u*u*w*r
            T021 = k021s + 2.0*v*k011s + w*k020s + v*v*w*r
            T102 = k102s + 2.0*w*k101s + u*k002s + u*w*w*r
            T012 = k012s + 2.0*w*k011s + v*k002s + v*w*w*r
            T220 = (k220s
                    + 2.0*u*k120s + 2.0*v*k210s + 4.0*u*v*k110s
                    + v*v*k200s   + u*u*k020s   + u*u*v*v*r)
            T202 = (k202s
                    + 2.0*u*k102s + 2.0*w*k201s + 4.0*u*w*k101s
                    + w*w*k200s   + u*u*k002s   + u*u*w*w*r)
            T022 = (k022s
                    + 2.0*v*k012s + 2.0*w*k021s + 4.0*v*w*k011s
                    + w*w*k020s   + v*v*k002s   + v*v*w*w*r)

            T = ti.Vector([
                T000, T100, T010, T001,
                T110, T101, T011,
                T200, T020, T002,
                T210, T120, T201, T021, T102, T012,
                T220, T202, T022
            ])

            # ── Step 4: reconstruct f* via M_raw_inv (19×19) ──────────
            for q in ti.static(range(19)):
                val = 0.0
                for r_idx in ti.static(range(19)):
                    val += self.M_inv[q, r_idx] * T[r_idx]
                self.f_post[i, j, k, q] = val

            # ── Step 5: Guo body-force correction (3D) ─────────────────
            coeff = 1.0 - s_v * 0.5
            for q in ti.static(range(19)):
                eu  = EX[q] * u + EY[q] * v + EZ[q] * w
                cx  = (EX[q] - u) / cs2 + eu * EX[q] / (cs2 * cs2)
                cy  = (EY[q] - v) / cs2 + eu * EY[q] / (cs2 * cs2)
                cz  = (EZ[q] - w) / cs2 + eu * EZ[q] / (cs2 * cs2)
                self.f_post[i, j, k, q] += W[q] * coeff * (
                    cx * Fx + cy * Fy + cz * Fz
                )

    # ------------------------------------------------------------------
    # Streaming  (pull — reads f_post, writes f; periodic y & z, open x)
    # ------------------------------------------------------------------
    @ti.kernel
    def _stream(self):
        for i, j, k in ti.ndrange(Nx, Ny, Nz):
            for q in ti.static(range(19)):
                i_src = (i - EX[q] + Nx) % Nx
                j_src = (j - EY[q] + Ny) % Ny
                k_src = (k - EZ[q] + Nz) % Nz
                self.f[i, j, k, q] = self.f_post[i_src, j_src, k_src, q]

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------
    @ti.kernel
    def _bc_inlet(self):
        """3D Zou-He velocity inlet at x=0.
        East-moving velocities (EX>0): q=1,7,9,11,13 — 5 unknowns."""
        ux_in = ti.static(_UX_IN)
        uy_in = ti.static(_UY_IN)
        for j, k in ti.ndrange(Ny, Nz):
            # Known west-facing + neutral directions
            S_known = (self.f[0,j,k,0]
                       + self.f[0,j,k,3] + self.f[0,j,k,4]
                       + self.f[0,j,k,5] + self.f[0,j,k,6]
                       + self.f[0,j,k,15] + self.f[0,j,k,16]
                       + self.f[0,j,k,17] + self.f[0,j,k,18])
            S_out   = (self.f[0,j,k,2]
                       + self.f[0,j,k,8]  + self.f[0,j,k,10]
                       + self.f[0,j,k,12] + self.f[0,j,k,14])
            rho_in  = (S_known + 2.0 * S_out) / (1.0 - ux_in)

            diff_y  = self.f[0,j,k,3] - self.f[0,j,k,4]   # f_N - f_S
            diff_z  = self.f[0,j,k,5] - self.f[0,j,k,6]   # f_T - f_B

            # 5 unknown east-going distributions (non-equilibrium bounce-back)
            self.f[0,j,k,1]  = self.f[0,j,k,2]  + (2.0/3.0) * rho_in * ux_in
            self.f[0,j,k,7]  = (self.f[0,j,k,10] + (1.0/6.0)*rho_in*ux_in
                                 + 0.5*rho_in*uy_in - 0.5*diff_y)
            self.f[0,j,k,9]  = (self.f[0,j,k,8]  + (1.0/6.0)*rho_in*ux_in
                                 - 0.5*rho_in*uy_in + 0.5*diff_y)
            self.f[0,j,k,11] = self.f[0,j,k,14] + (1.0/6.0)*rho_in*ux_in - 0.5*diff_z
            self.f[0,j,k,13] = self.f[0,j,k,12] + (1.0/6.0)*rho_in*ux_in + 0.5*diff_z

    @ti.kernel
    def _bc_outlet(self):
        """Zero-gradient outlet at x=Nx-1: copy from x=Nx-2."""
        for j, k, q in ti.ndrange(Ny, Nz, 19):
            self.f[Nx-1, j, k, q] = self.f[Nx-2, j, k, q]

    # ------------------------------------------------------------------
    # Post-step macroscopic recompute  (no force correction)
    # ------------------------------------------------------------------
    @ti.kernel
    def _macro_post(self):
        for i, j, k in ti.ndrange(Nx, Ny, Nz):
            r  = 0.0
            mx = 0.0
            my = 0.0
            mz = 0.0
            for q in ti.static(range(19)):
                fq  = self.f[i, j, k, q]
                r  += fq
                mx += EX[q] * fq
                my += EY[q] * fq
                mz += EZ[q] * fq
            self.rho[i, j, k] = r
            self.ux[i, j, k]  = mx / r
            self.uy[i, j, k]  = my / r
            self.uz[i, j, k]  = mz / r

    # ------------------------------------------------------------------
    # Public: advance one time step
    # ------------------------------------------------------------------
    def step(self,
             fx: ti.Field,
             fy: ti.Field,
             fz: ti.Field):
        """
        Advance one LBM step.

        Parameters
        ----------
        fx, fy, fz : ti.field(f32, shape=(Nx, Ny, Nz))
            IBM Eulerian body-force density (Taichi fields).
        """
        self._macro_with_force(fx, fy, fz)
        self._collide(fx, fy, fz)
        self._stream()
        self._bc_inlet()
        self._bc_outlet()
        self._macro_post()
