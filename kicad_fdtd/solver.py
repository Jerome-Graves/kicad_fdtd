"""Yee FDTD on a non-uniform rectilinear grid: CPML on six faces, PEC edge
masks from voxel.assign, lumped ports. CuPy when available, NumPy otherwise.

Layout (node lines x[i], y[j], z[k]; cells between them):
  Ex (nx-1, ny, nz)   at (mx[i], y[j], z[k])      Hx (nx, ny-1, nz-1) at (x[i], my[j], mz[k])
  Ey (nx, ny-1, nz)   at (x[i], my[j], z[k])      Hy (nx-1, ny, nz-1) at (mx[i], y[j], mz[k])
  Ez (nx, ny, nz-1)   at (x[i], y[j], mz[k])      Hz (nx-1, ny-1, nz) at (mx[i], my[j], z[k])
H -= dt/mu0 * curl E   (differences over primal cell lengths)
E += dt/eps * curl H   (differences over dual lengths between H positions)
Outer boundary nodes stay E = 0 (PEC) behind the CPML.

Lumped port: Ez edges inside a box [x0,x1]x[y0,y1] from z0 (plane) to z1
(signal) carry a distributed conductance sigma = L/(R A) (total R) in
series with a voltage source V(t) (Thevenin, as openEMS's lumped port).
Voltage = signal minus plane = -sum(Ez dz) along the box axis; current =
Ampere loop of H around the box at mid height, positive UP into the signal
conductor. Then inc = (V + R I)/2, ref = (V - R I)/2 per port.
"""
import math
import os
import time

import numpy as np

try:
    import cupy as cp
    cp.zeros(1)
    HAVE_GPU = True
except Exception:                                    # pragma: no cover
    cp = np
    HAVE_GPU = False

C0, EPS0, MU0 = 299792458.0, 8.854187817e-12, 4e-7 * math.pi
ETA0 = math.sqrt(MU0 / EPS0)


def gauss_pulse(f0, fc):
    """V(t) = exp(-((t-t0)/tau)^2) cos(2 pi f0 (t-t0)); spectrum covers ~f0 +- fc."""
    tau = 0.5 / fc
    t0 = 3.5 * tau
    return (lambda t: math.exp(-((t - t0) / tau) ** 2) * math.cos(2 * math.pi * f0 * (t - t0))), t0, tau


class Port:
    pass


class FDTD:
    def __init__(self, lines, vox, er_of_z, npml=8, cfl=0.99, use_gpu=True):
        xp = cp if (use_gpu and HAVE_GPU) else np
        self.xp = xp
        self.lines = lines
        x, y, z = [np.asarray(lines[k], float) * 1e-3 for k in "xyz"]
        self.x, self.y, self.z = x, y, z
        nx, ny, nz = len(x), len(y), len(z)
        self.nx, self.ny, self.nz = nx, ny, nz
        dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)
        ddx = np.concatenate([[dx[0]], (dx[:-1] + dx[1:]) / 2, [dx[-1]]])     # dual lengths at nodes
        ddy = np.concatenate([[dy[0]], (dy[:-1] + dy[1:]) / 2, [dy[-1]]])
        ddz = np.concatenate([[dz[0]], (dz[:-1] + dz[1:]) / 2, [dz[-1]]])
        self.dt = cfl / (C0 * math.sqrt(1 / dx.min() ** 2 + 1 / dy.min() ** 2 + 1 / dz.min() ** 2))
        f32 = np.float32
        # permittivity: cells between z planes; edges ON a plane average the two sides
        er_cell = np.array([er_of_z(v) for v in (z[:-1] + z[1:]) / 2])
        self.er_cell = er_cell
        er_plane = np.array([(er_cell[k] if k < nz - 1 else 1.0) / 2 + (er_cell[k - 1] if k > 0 else 1.0) / 2 for k in range(nz)])
        Z = lambda shape: xp.zeros(shape, f32)
        self.Ex, self.Ey, self.Ez = Z((nx - 1, ny, nz)), Z((nx, ny - 1, nz)), Z((nx, ny, nz - 1))
        self.Hx, self.Hy, self.Hz = Z((nx, ny - 1, nz - 1)), Z((nx - 1, ny, nz - 1)), Z((nx - 1, ny - 1, nz))
        cE_plane = (self.dt / (EPS0 * er_plane)).astype(f32)
        cE_cell = (self.dt / (EPS0 * er_cell)).astype(f32)
        self.cEx = xp.asarray(np.broadcast_to(cE_plane[None, None, :], (nx - 1, ny, nz)).copy())
        self.cEy = xp.asarray(np.broadcast_to(cE_plane[None, None, :], (nx, ny - 1, nz)).copy())
        self.cEz = xp.asarray(np.broadcast_to(cE_cell[None, None, :], (nx, ny, nz - 1)).copy())
        self.cEx[xp.asarray(vox["ex"] >= 0)] = 0
        self.cEy[xp.asarray(vox["ey"] >= 0)] = 0
        self.cEz[xp.asarray(vox["ez"] >= 0)] = 0
        self.cH = f32(self.dt / MU0)
        A = lambda v: xp.asarray(np.asarray(v, f32))
        self.i_dx, self.i_dy, self.i_dz = A(1 / dx), A(1 / dy), A(1 / dz)
        self.i_ddx, self.i_ddy, self.i_ddz = A(1 / ddx), A(1 / ddy), A(1 / ddz)
        self.npml = npml
        self._setup_cpml(dx, dy, dz)
        self.ports = []
        self.energy = []

    # ---------------------------------------------------------------- CPML
    def _setup_cpml(self, dx, dy, dz):
        """b, c coefficient vectors for each axis and side, at E (node) and H
        (cell-centre) positions of the npml outer cells. Profiles: sigma =
        sigma_max rho^m, alpha = alpha_max (1 - rho), kappa = 1."""
        n, m, amax = self.npml, 3.0, 0.05
        self.pml = {}
        for axis, d in (("x", dx), ("y", dy), ("z", dz)):
            for side in ("lo", "hi"):
                dcells = d[:n] if side == "lo" else d[-n:]
                smax = 0.8 * (m + 1) / (ETA0 * float(np.mean(dcells)))
                # E positions: nodes 0..n (lo) / N-1-n..N-1 (hi); rho = 1 at the boundary node, 0 at node n
                rho_e = np.array([(n - i) / n for i in range(n + 1)])         # i = distance from boundary in nodes
                rho_h = np.array([(n - i - 0.5) / n for i in range(n)])       # cell centres
                def coef(rho):
                    rho = np.clip(rho, 0, 1)
                    sig = smax * rho ** m
                    a = amax * (1 - rho)
                    b = np.exp(-(sig + a) * self.dt / EPS0)
                    c = np.where(sig + a > 0, sig * (b - 1) / (sig + a + 1e-30), 0.0)
                    return b.astype(np.float32), c.astype(np.float32)
                self.pml[(axis, side)] = {"e": coef(rho_e), "h": coef(rho_h)}
        xp = self.xp
        # psi accumulators, allocated lazily by name
        self.psi = {}
        self.pmld = {}

    def _psi(self, name, shape):
        if name not in self.psi:
            self.psi[name] = self.xp.zeros(shape, np.float32)
        return self.psi[name]

    def _coef(self, axis, side, kind):
        """Device b, c vectors for one PML slab, broadcast along its axis; built once."""
        key = (axis, side, kind)
        if key not in self.pmld:
            n = self.npml
            b, c = self.pml[(axis, side)][kind]
            if kind == "e":
                b, c = b[1:n + 1], c[1:n + 1]                 # interior nodes 1..n from the boundary
            if side == "hi":
                b, c = b[::-1].copy(), c[::-1].copy()
            shape = [1, 1, 1]
            shape["xyz".index(axis)] = n
            self.pmld[key] = (self.xp.asarray(b).reshape(shape), self.xp.asarray(c).reshape(shape))
        return self.pmld[key]

    def _apply_cpml_h(self, dEz_dx, dEy_dx, dEx_dy, dEz_dy, dEy_dz, dEx_dz):
        """psi corrections to H from full derivative arrays (unfused path)."""
        n = self.npml
        s = {}
        for side, sl in (("lo", slice(0, n)), ("hi", slice(-n, None))):
            s["dEz_dx_" + side], s["dEy_dx_" + side] = dEz_dx[sl], dEy_dx[sl]
            s["dEx_dy_" + side], s["dEz_dy_" + side] = dEx_dy[:, sl], dEz_dy[:, sl]
            s["dEy_dz_" + side], s["dEx_dz_" + side] = dEy_dz[:, :, sl], dEx_dz[:, :, sl]
        self._cpml_h_slabs(s)

    def _apply_cpml_e(self, dHz_dx, dHy_dx, dHx_dy, dHz_dy, dHy_dz, dHx_dz):
        """psi corrections to E. The dual derivatives are defined on interior
        nodes 1..N-2 (arrays of size N-2); the PML covers the first/last n of those."""
        n = self.npml
        s = {}
        for side, sl in (("lo", slice(0, n)), ("hi", slice(-n, None))):
            s["dHz_dx_" + side], s["dHy_dx_" + side] = dHz_dx[sl], dHy_dx[sl]
            s["dHx_dy_" + side], s["dHz_dy_" + side] = dHx_dy[:, sl], dHz_dy[:, sl]
            s["dHy_dz_" + side], s["dHx_dz_" + side] = dHy_dz[:, :, sl], dHx_dz[:, :, sl]
        self._cpml_e_slabs(s)

    def _cpml_h_slabs(self, s):
        """psi_h = b psi_h + c dE (slab); H += / -= cH psi_h."""
        n, cH = self.npml, self.cH
        for side, sl in (("lo", slice(0, n)), ("hi", slice(-n, None))):
            b, c = self._coef("x", side, "h")
            p = self._psi("Hy_x_" + side, s["dEz_dx_" + side].shape); p *= b; p += c * s["dEz_dx_" + side]; self.Hy[sl] += cH * p
            p = self._psi("Hz_x_" + side, s["dEy_dx_" + side].shape); p *= b; p += c * s["dEy_dx_" + side]; self.Hz[sl] -= cH * p
            b, c = self._coef("y", side, "h")
            p = self._psi("Hz_y_" + side, s["dEx_dy_" + side].shape); p *= b; p += c * s["dEx_dy_" + side]; self.Hz[:, sl] += cH * p
            p = self._psi("Hx_y_" + side, s["dEz_dy_" + side].shape); p *= b; p += c * s["dEz_dy_" + side]; self.Hx[:, sl] -= cH * p
            b, c = self._coef("z", side, "h")
            p = self._psi("Hx_z_" + side, s["dEy_dz_" + side].shape); p *= b; p += c * s["dEy_dz_" + side]; self.Hx[:, :, sl] += cH * p
            p = self._psi("Hy_z_" + side, s["dEx_dz_" + side].shape); p *= b; p += c * s["dEx_dz_" + side]; self.Hy[:, :, sl] -= cH * p

    def _cpml_e_slabs(self, s):
        n = self.npml
        for side, sl in (("lo", slice(0, n)), ("hi", slice(-n, None))):
            b_, c_ = self._coef("x", side, "e")
            p = self._psi("Ey_x_" + side, s["dHz_dx_" + side].shape); p *= b_; p += c_ * s["dHz_dx_" + side]; self.Ey[1:-1, :, 1:-1][sl] -= self.cEy[1:-1, :, 1:-1][sl] * p
            p = self._psi("Ez_x_" + side, s["dHy_dx_" + side].shape); p *= b_; p += c_ * s["dHy_dx_" + side]; self.Ez[1:-1, 1:-1, :][sl] += self.cEz[1:-1, 1:-1, :][sl] * p
            b_, c_ = self._coef("y", side, "e")
            p = self._psi("Ez_y_" + side, s["dHx_dy_" + side].shape); p *= b_; p += c_ * s["dHx_dy_" + side]; self.Ez[1:-1, 1:-1, :][:, sl] -= self.cEz[1:-1, 1:-1, :][:, sl] * p
            p = self._psi("Ex_y_" + side, s["dHz_dy_" + side].shape); p *= b_; p += c_ * s["dHz_dy_" + side]; self.Ex[:, 1:-1, 1:-1][:, sl] += self.cEx[:, 1:-1, 1:-1][:, sl] * p
            b_, c_ = self._coef("z", side, "e")
            p = self._psi("Ex_z_" + side, s["dHy_dz_" + side].shape); p *= b_; p += c_ * s["dHy_dz_" + side]; self.Ex[:, 1:-1, 1:-1][:, :, sl] -= self.cEx[:, 1:-1, 1:-1][:, :, sl] * p
            p = self._psi("Ey_z_" + side, s["dHx_dz_" + side].shape); p *= b_; p += c_ * s["dHx_dz_" + side]; self.Ey[1:-1, :, 1:-1][:, :, sl] += self.cEy[1:-1, :, 1:-1][:, :, sl] * p

    # ---------------------------------------------------------------- ports
    def add_port(self, x0, x1, y0, y1, z0, z1, R, excite=0.0):
        """Box in mm. z0 = plane, z1 = signal conductor."""
        p = Port()
        p.nr, p.R, p.excite = len(self.ports), R, excite
        xl, yl, zl = self.lines["x"], self.lines["y"], self.lines["z"]
        p.ix = np.where((xl >= x0 - 1e-9) & (xl <= x1 + 1e-9))[0]
        p.jy = np.where((yl >= y0 - 1e-9) & (yl <= y1 + 1e-9))[0]
        p.kz = np.where((zl >= z0 - 1e-9) & (zl < z1 - 1e-9))[0]        # Ez edge k spans z[k]..z[k+1]
        assert len(p.ix) and len(p.jy) and len(p.kz), "port box contains no Ez edge"
        assert p.ix[0] > 0 and p.jy[0] > 0 and p.ix[-1] < self.nx - 1 and p.jy[-1] < self.ny - 1, "port touches the boundary"
        p.ic, p.jc = int(p.ix[len(p.ix) // 2]), int(p.jy[len(p.jy) // 2])
        p.kmid = int(p.kz[len(p.kz) // 2])
        L = (zl[p.kz[-1] + 1] - zl[p.kz[0]]) * 1e-3
        # each Ez edge column is one parallel resistor: total R = R -> per-edge conductance from its dual area
        ddx = np.concatenate([[np.diff(self.x)[0]], (np.diff(self.x)[:-1] + np.diff(self.x)[1:]) / 2, [np.diff(self.x)[-1]]])
        ddy = np.concatenate([[np.diff(self.y)[0]], (np.diff(self.y)[:-1] + np.diff(self.y)[1:]) / 2, [np.diff(self.y)[-1]]])
        area = float(np.sum(ddx[p.ix])) * float(np.sum(ddy[p.jy]))
        p.sigma = L / (R * area)
        p.L = L
        p.V, p.I = [], []
        # per-column edge lengths for V, loop lengths for I
        p.dz = self.xp.asarray((self.z[p.kz + 1] - self.z[p.kz]).astype(np.float32))
        p.lx = self.xp.asarray(ddx[p.ix].astype(np.float32))
        p.ly = self.xp.asarray(ddy[p.jy].astype(np.float32))
        self.ports.append(p)
        return p

    # ---------------------------------------------------------------- run
    def run(self, exc_fn, nsteps, end_criteria=1e-3, log=None, min_steps=1000, log_every=2000, hook=None, hook_every=200):
        """hook(n, sim) is called every hook_every steps (after the energy check
        on those steps); return False from it to stop the run early."""
        xp = self.xp
        Ex, Ey, Ez, Hx, Hy, Hz = self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz
        i_dx, i_dy, i_dz = self.i_dx[:, None, None], self.i_dy[None, :, None], self.i_dz[None, None, :]
        i_ddx, i_ddy, i_ddz = self.i_ddx[1:-1][:, None, None], self.i_ddy[1:-1][None, :, None], self.i_ddz[1:-1][None, None, :]
        cH = self.cH
        for p in self.ports:
            eps = EPS0 * float(self.er_cell[p.kmid])            # the port sits in the dielectric, not in vacuum
            f = self.dt * p.sigma / (2 * eps)
            p.f = np.float32(f)
            p.la, p.lb = np.float32((1 - f) / (1 + f)), np.float32(1 / (1 + f))
            p.src = np.float32(self.dt * p.sigma / eps)
            p.box = np.ix_(p.ix, p.jy, p.kz)
            p.Vd = xp.zeros(nsteps, dtype=np.float32)
            p.Id = xp.zeros(nsteps, dtype=np.float32)
        emax, t_start = 0.0, time.time()
        self.energy_db, self.step_now, self.t_start = 0.0, 0, t_start
        fused = self.xp is not np and not os.environ.get("FDTD_UNFUSED")     # FDTD_UNFUSED=1: slice-based reference path
        if fused:
            from . import kernels as K
            nx, ny, nz = np.int32(self.nx), np.int32(self.ny), np.int32(self.nz)
            npml = np.int32(self.npml)
            i_dx1, i_dy1, i_dz1 = self.i_dx, self.i_dy, self.i_dz
            i_ddx1, i_ddy1, i_ddz1 = self.i_ddx, self.i_ddy, self.i_ddz
            cH = np.float32(self.cH)
            self.pmlv = K.pml_vectors(xp, self.pml, self.npml)
            P = K.psi_arrays(xp, self.nx, self.ny, self.nz, self.npml)
            self.psi_fused = P
            # ports as flat tables for the kernels (index sets are contiguous ranges)
            NP = max(1, len(self.ports))
            pb = np.zeros((NP, 6), np.int32); pc = np.zeros((NP, 3), np.float32); pg = np.zeros((NP, 9), np.int32)
            for p in self.ports:
                assert np.all(np.diff(p.ix) == 1) and np.all(np.diff(p.jy) == 1) and np.all(np.diff(p.kz) == 1), "port box not contiguous"
                pb[p.nr] = [p.ix[0], p.ix[-1], p.jy[0], p.jy[-1], p.kz[0], p.kz[-1]]
                pc[p.nr] = [p.lb, p.f, p.src * (-p.excite / p.L) if p.excite else 0.0]
                pg[p.nr] = [p.ic, p.jc, p.kz[0], p.kz[-1], p.kmid, p.ix[0], p.ix[-1], p.jy[0], p.jy[-1]]
            pb_d, pc_d, pg_d = xp.asarray(pb), xp.asarray(pc), xp.asarray(pg)
            npt = np.int32(len(self.ports))
            dzv = xp.asarray((self.z[1:] - self.z[:-1]).astype(np.float32))
            dxx, dyy = np.diff(self.x), np.diff(self.y)
            ddx_d = xp.asarray(np.concatenate([[dxx[0]], (dxx[:-1] + dxx[1:]) / 2, [dxx[-1]]]).astype(np.float32))
            ddy_d = xp.asarray(np.concatenate([[dyy[0]], (dyy[:-1] + dyy[1:]) / 2, [dyy[-1]]]).astype(np.float32))
            VId = xp.zeros((2, NP, nsteps), np.float32)
            nstep32 = np.int32(nsteps)
        prof = os.environ.get("FDTD_PROFILE") and fused
        if prof:
            sync = cp.cuda.Device().synchronize
            tk = {"H": 0.0, "E": 0.0, "ports": 0.0, "energy": 0.0}
            lap = [time.perf_counter()]

            def tick(key):
                sync(); now = time.perf_counter(); tk[key] += now - lap[0]; lap[0] = now
        else:
            tick = lambda key: None
        for n in range(nsteps):
            t = n * self.dt
            if prof:
                sync(); lap[0] = time.perf_counter()
            if fused:
                V = self.pmlv
                K.kHx(Ey, Ez, i_dy1, i_dz1, cH, nx, ny, nz, npml, V[("y","h","b")], V[("y","h","c")], V[("z","h","b")], V[("z","h","c")], P["Hx_y"], P["Hx_z"], Hx, size=Hx.size)
                K.kHy(Ex, Ez, i_dx1, i_dz1, cH, nx, ny, nz, npml, V[("x","h","b")], V[("x","h","c")], V[("z","h","b")], V[("z","h","c")], P["Hy_x"], P["Hy_z"], Hy, size=Hy.size)
                K.kHz(Ex, Ey, i_dx1, i_dy1, cH, nx, ny, nz, npml, V[("x","h","b")], V[("x","h","c")], V[("y","h","b")], V[("y","h","c")], P["Hz_x"], P["Hz_y"], Hz, size=Hz.size)
                tick("H")
                K.kEx(Hy, Hz, self.cEx, i_ddy1, i_ddz1, cH, nx, ny, nz, npml, V[("y","e","b")], V[("y","e","c")], V[("z","e","b")], V[("z","e","c")], P["Ex_y"], P["Ex_z"], Ex, size=Ex.size)
                K.kEy(Hx, Hz, self.cEy, i_ddx1, i_ddz1, cH, nx, ny, nz, npml, V[("x","e","b")], V[("x","e","c")], V[("z","e","b")], V[("z","e","c")], P["Ey_x"], P["Ey_z"], Ey, size=Ey.size)
                K.kEzP(Hx, Hy, self.cEz, i_ddx1, i_ddy1, cH, nx, ny, nz, npml, pb_d, pc_d, npt, np.float32(exc_fn(t)),
                       V[("x","e","b")], V[("x","e","c")], V[("y","e","b")], V[("y","e","c")], P["Ez_x"], P["Ez_y"], Ez, size=Ez.size)
                tick("E")
                K.kPortVI(Ez, Hx, Hy, pg_d, dzv, ddx_d, ddy_d, nx, ny, nz, nstep32, np.int32(n), VId[0], VId[1], size=int(npt))
                tick("ports")
            else:
                # ---- H
                dEz_dy = (Ez[:, 1:, :] - Ez[:, :-1, :]) * i_dy
                dEy_dz = (Ey[:, :, 1:] - Ey[:, :, :-1]) * i_dz
                dEx_dz = (Ex[:, :, 1:] - Ex[:, :, :-1]) * i_dz
                dEz_dx = (Ez[1:, :, :] - Ez[:-1, :, :]) * i_dx
                dEy_dx = (Ey[1:, :, :] - Ey[:-1, :, :]) * i_dx
                dEx_dy = (Ex[:, 1:, :] - Ex[:, :-1, :]) * i_dy
                Hx -= cH * (dEz_dy - dEy_dz)
                Hy -= cH * (dEx_dz - dEz_dx)
                Hz -= cH * (dEy_dx - dEx_dy)
                self._apply_cpml_h(dEz_dx, dEy_dx, dEx_dy, dEz_dy, dEy_dz, dEx_dz)
                # ---- E (interior)
                for p in self.ports:
                    p.Eold = Ez[p.box].copy()
                dHz_dy = (Hz[:, 1:, 1:-1] - Hz[:, :-1, 1:-1]) * i_ddy
                dHy_dz = (Hy[:, 1:-1, 1:] - Hy[:, 1:-1, :-1]) * i_ddz
                Ex[:, 1:-1, 1:-1] += self.cEx[:, 1:-1, 1:-1] * (dHz_dy - dHy_dz)
                dHx_dz = (Hx[1:-1, :, 1:] - Hx[1:-1, :, :-1]) * i_ddz
                dHz_dx = (Hz[1:, :, 1:-1] - Hz[:-1, :, 1:-1]) * i_ddx
                Ey[1:-1, :, 1:-1] += self.cEy[1:-1, :, 1:-1] * (dHx_dz - dHz_dx)
                dHy_dx = (Hy[1:, 1:-1, :] - Hy[:-1, 1:-1, :]) * i_ddx
                dHx_dy = (Hx[1:-1, 1:, :] - Hx[1:-1, :-1, :]) * i_ddy
                Ez[1:-1, 1:-1, :] += self.cEz[1:-1, 1:-1, :] * (dHy_dx - dHx_dy)
                self._apply_cpml_e(dHz_dx, dHy_dx, dHx_dy, dHz_dy, dHy_dz, dHx_dz)
            # ---- ports (unfused path): E_new = lb*(E' - f*E_old) + lb*src*E_src  (resistor + Thevenin source)
            for p in ([] if fused else self.ports):
                blk = Ez[p.box]
                blk = p.lb * (blk - p.f * p.Eold)
                if p.excite:
                    blk += p.lb * p.src * np.float32(-p.excite * exc_fn(t) / p.L)     # +V on the signal = -Ez
                Ez[p.box] = blk
                # V and I stay on the device (no per-step sync); fetched once after the loop
                p.Vd[n] = -xp.sum(Ez[p.ic, p.jc, p.kz] * p.dz)
                k = p.kmid
                i0, i1, j0, j1 = int(p.ix[0]), int(p.ix[-1]), int(p.jy[0]), int(p.jy[-1])
                p.Id[n] = (xp.sum(Hx[i0:i1 + 1, j0 - 1, k] * p.lx) - xp.sum(Hx[i0:i1 + 1, j1, k] * p.lx)
                           + xp.sum(Hy[i1, j0:j1 + 1, k] * p.ly) - xp.sum(Hy[i0 - 1, j0:j1 + 1, k] * p.ly))
            if n % 100 == 0:
                e = float(xp.sum(Ex * Ex)) + float(xp.sum(Ey * Ey)) + float(xp.sum(Ez * Ez))
                tick("energy")
                emax = max(emax, e)
                self.energy.append((n, e))
                if log and n % log_every == 0:
                    log("step %6d  t %.2f ns  energy %6.1f dB  %.0f s" % (n, t * 1e9, 10 * math.log10(e / emax + 1e-30), time.time() - t_start))
                self.energy_db = 10 * math.log10(e / emax + 1e-30) if emax > 0 else 0.0
                self.step_now = n
                if n > min_steps and emax > 0 and e < end_criteria * emax:
                    break
            if hook is not None and n % hook_every == 0 and hook(n, self) is False:
                break
        self.nsteps_done = n + 1
        self.wall = time.time() - t_start
        if prof:
            tot = sum(tk.values())
            print("PROFILE per step: " + "  ".join("%s %.2f ms" % (k, 1e3 * v / (n + 1)) for k, v in tk.items()) + "  (sum %.2f ms)" % (1e3 * tot / (n + 1)))
        if fused:
            VI = VId.get()
            for p in self.ports:
                p.V, p.I = VI[0, p.nr, :n + 1].astype(np.float64), VI[1, p.nr, :n + 1].astype(np.float64)
        else:
            for p in self.ports:
                p.V = np.asarray(p.Vd[:n + 1].get() if xp is not np else p.Vd[:n + 1], dtype=np.float64)
                p.I = np.asarray(p.Id[:n + 1].get() if xp is not np else p.Id[:n + 1], dtype=np.float64)
        return self

    def port_spectra(self, freqs):
        out = {}
        for p in self.ports:
            V, I = np.array(p.V), np.array(p.I)
            t = np.arange(len(V)) * self.dt
            e = np.exp(-2j * math.pi * np.outer(freqs, t)) * self.dt
            Vf, If = e @ V, e @ I
            out[p.nr] = {"V": Vf, "I": If, "inc": (Vf + p.R * If) / 2, "ref": (Vf - p.R * If) / 2, "Z": Vf / np.where(abs(If) > 0, If, 1e-30)}
        return out
