"""Steady-state thermal and DC conduction on the board's exact copper.

    kicad-fdtd thermal board.kicad_pcb --power U1=1.2 U7=0.4 [--h 10] [--tamb 25] [--cell 0.25]
                         [--net /VBAT --from J1.1 --to U2.4 --current 2.0]

Thermal: finite-volume heat conduction on a whole-board rectilinear grid. Every copper
layer is one cell thick (35 um by default) with conductivity k_cu * copper fraction
(exact polygons sampled 3 x 3 per cell); dielectric cells are FR4 (0.8 W/mK in plane,
0.3 through); via barrels add through-plane copper. Component power goes into the cells
under the footprint's pads. Convection h (W/m2K) to T_amb on both faces, edges adiabatic,
no radiation, nothing above the board (a conservative estimate).

DC: on one net, sheet conduction in the copper layers (sigma t x fraction) and via barrels
between them; current injected at one pad, drawn at another, one node at 0 V. Gives the
IR drop, the current density per cell, and the Joule heat, which is added to the thermal
sources.

Solver: conjugate gradient with a Jacobi preconditioner on the GPU (cupyx) or CPU (scipy).
"""
import json
import math
import os
import sys
import time

import numpy as np
import shapely
from shapely.ops import unary_union

from . import geometry

K_CU, K_FR4_XY, K_FR4_Z, K_AIR = 385.0, 0.8, 0.3, 0.03      # W/mK
RHO_CU = 1.68e-8                                            # ohm m
VIA_WALL = 0.025                                            # mm plating


class Grid:
    """Whole-board rectilinear cells. k index runs from the top copper downwards."""

    def __init__(self, bd, cell=0.25, dz_max=0.4):
        self.bd = bd
        self.cell = cell
        nx = max(2, int(math.ceil(bd.w / cell)) + 1)
        ny = max(2, int(math.ceil(bd.h / cell)) + 1)
        self.x = np.linspace(0.0, bd.w, nx)
        self.y = np.linspace(-bd.h, 0.0, ny)
        # z planes (mm): copper layer i occupies [z_i - t, z_i] ... we put copper ABOVE each z line:
        # copper cell [z_i, z_i + t_cu] except stacking must be monotone, so shift: top copper is
        # [0, -t] and dielectric i is [z_i - t, z_{i+1}] split into <= dz_max cells.
        zs, kind = [0.0], []
        for i, layer in enumerate(bd.layers):
            t = bd.cu_thickness.get(layer, 0.035)
            zs.append(zs[-1] - t); kind.append(("cu", layer))
            if i < len(bd.layers) - 1:
                zt, zb, er, tand = bd.diel[i]
                span = zt - zb
                n = max(1, int(math.ceil(span / dz_max)))
                for j in range(n):
                    zs.append(zs[-1] - span / n); kind.append(("diel", i))
        self.z = np.array(zs)                     # decreasing
        self.kind = kind
        self.nx, self.ny, self.nz = nx - 1, ny - 1, len(zs) - 1      # cell counts
        self.dx = np.diff(self.x); self.dy = np.diff(self.y); self.dz = -np.diff(self.z)
        self.cx = (self.x[:-1] + self.x[1:]) / 2; self.cy = (self.y[:-1] + self.y[1:]) / 2
        self.cz = (self.z[:-1] + self.z[1:]) / 2
        self.k_cu = {layer: k for k, (kd, layer) in enumerate(kind) if kd == "cu"}
        self._frac_cache = {}
        self._build_materials()

    # ---------------------------------------------------------- copper sampling
    def _sub_points(self):
        if not hasattr(self, "_sub"):
            o = np.array([-1 / 3, 0.0, 1 / 3])
            X = self.cx[:, None, None, None] + o[None, None, :, None] * self.dx[:, None, None, None]
            Y = self.cy[None, :, None, None] + o[None, None, None, :] * self.dy[None, :, None, None]
            X, Y = np.broadcast_arrays(X, Y)
            self._sub = (X.ravel(), Y.ravel())
        return self._sub

    def copper_fraction(self, layer, net=None):
        """Fraction of each cell covered by copper on a layer (all nets, or one net)."""
        key = (layer, net)
        if key not in self._frac_cache:
            ps = [P for n, kind, P in self.bd.polys(layer) if net is None or n == net]
            if not ps:
                self._frac_cache[key] = np.zeros((self.nx, self.ny))
            else:
                U = unary_union(ps)
                X, Y = self._sub_points()
                inside = shapely.contains_xy(U, X, Y).reshape(self.nx, self.ny, 9)
                self._frac_cache[key] = inside.mean(axis=2)
        return self._frac_cache[key]

    def inside_outline(self):
        if not hasattr(self, "_in"):
            X, Y = np.meshgrid(self.cx, self.cy, indexing="ij")
            self._in = shapely.contains_xy(self.bd.outline, X.ravel(), Y.ravel()).reshape(self.nx, self.ny)
        return self._in

    def via_copper_area(self, net=None):
        """Copper cross-section of via barrels per (i, j, k) dielectric cell, mm2."""
        A = np.zeros((self.nx, self.ny, self.nz))
        for v in self.bd.vias():
            if net is not None and v["net"] != net:
                continue
            i = int(np.searchsorted(self.x, v["x"]) - 1); j = int(np.searchsorted(self.y, v["y"]) - 1)
            if not (0 <= i < self.nx and 0 <= j < self.ny):
                continue
            # barrel = plated hole wall: drill radius + plating (the exporter's dia is drill + 0.05 for
            # through-hole pads but the pad size for vias, so go from the drill when we have it)
            ri = v["drill"] / 2 if v.get("drill") else max(0.0, v["r"] - VIA_WALL)
            ro = ri + VIA_WALL
            area = math.pi * (ro ** 2 - ri ** 2)
            kt = self.k_cu.get(v.get("top"), 0); kb = self.k_cu.get(v.get("bottom"), self.nz - 1)
            for k in range(min(kt, kb), max(kt, kb) + 1):        # every cell the barrel passes, copper layers included
                A[i, j, k] += area
        return A

    def _build_materials(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        self.kx = np.full((nx, ny, nz), K_FR4_XY); self.kz = np.full((nx, ny, nz), K_FR4_Z)
        self.cu_frac = np.zeros((nx, ny, nz))
        out = ~self.inside_outline()
        for k, (kd, layer) in enumerate(self.kind):
            if kd == "cu":
                f = self.copper_fraction(layer)
                self.cu_frac[:, :, k] = f
                self.kx[:, :, k] = f * K_CU + (1 - f) * K_FR4_XY
                self.kz[:, :, k] = f * K_CU + (1 - f) * K_FR4_Z
        A = self.via_copper_area()
        cellA = self.dx[:, None, None] * self.dy[None, :, None]
        self.kz += A / cellA * K_CU
        self.kx[out] = K_AIR; self.kz[out] = K_AIR
        self.ky = self.kx

    # ---------------------------------------------------------- helpers
    def cell_of(self, x, y):
        i = int(np.clip(np.searchsorted(self.x, x) - 1, 0, self.nx - 1)); j = int(np.clip(np.searchsorted(self.y, y) - 1, 0, self.ny - 1))
        return i, j

    def pad_cells(self, ref, num=None, layer=None):
        """(i, j, k) of the cell under each pad of a footprint (or one pad), on its outer copper layer."""
        out = []
        for p in self.bd.pads:
            if p["ref"] != ref or (num is not None and p["num"] != str(num)) or not p["layers"]:
                continue                                   # (pads with no copper layers: NPTH holes, paste-only)
            lay = layer or (self.bd.layers[0] if self.bd.layers[0] in p["layers"] else (self.bd.layers[-1] if self.bd.layers[-1] in p["layers"] else p["layers"][0]))
            x, y = self.bd.pt(p["x"], p["y"])
            i, j = self.cell_of(x, y)
            out.append((i, j, self.k_cu[lay]))
        return out


# ---------------------------------------------------------------- linear algebra
def _cg(rows, cols, vals, diag, b, tol=1e-9, maxiter=20000, use_gpu=True):
    """Solve (D - offdiag) x = b, SPD, Jacobi-preconditioned CG. rows/cols/vals: off-diagonal
    couplings (both directions), diag: diagonal entries."""
    n = len(diag)
    try:
        import cupy as cp
        import cupyx.scipy.sparse as csp
        from cupyx.scipy.sparse.linalg import cg
        if not use_gpu:
            raise ImportError
        A = csp.coo_matrix((cp.asarray(np.concatenate([-vals, diag])), (cp.asarray(np.concatenate([rows, np.arange(n)])), cp.asarray(np.concatenate([cols, np.arange(n)])))), shape=(n, n)).tocsr()
        M = csp.diags(1.0 / cp.asarray(diag))
        try:
            x, info = cg(A, cp.asarray(b), rtol=tol, maxiter=maxiter, M=M)
        except TypeError:                                   # older cupyx spelling
            x, info = cg(A, cp.asarray(b), tol=tol, maxiter=maxiter, M=M)
        x = cp.asnumpy(x)
    except ImportError:
        import scipy.sparse as sp
        from scipy.sparse.linalg import cg
        A = sp.coo_matrix((np.concatenate([-vals, diag]), (np.concatenate([rows, np.arange(n)]), np.concatenate([cols, np.arange(n)]))), shape=(n, n)).tocsr()
        M = sp.diags(1.0 / diag)
        x, info = cg(A, b, rtol=tol, maxiter=maxiter, M=M)
    if info != 0:
        raise RuntimeError("CG did not converge (info %s)" % info)
    return x


def _links(g, kx, ky, kz, active=None):
    """Conductances between neighbouring cells (W/K or S): arrays of (a, b, G) in flat indices."""
    nx, ny, nz = g.nx, g.ny, g.nz
    idx = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    dx, dy, dz = g.dx[:, None, None] * 1e-3, g.dy[None, :, None] * 1e-3, g.dz[None, None, :] * 1e-3
    out = []
    for axis, kk, d in ((0, kx, dx), (1, ky, dy), (2, kz, dz)):
        sl_a = [slice(None)] * 3; sl_b = [slice(None)] * 3
        sl_a[axis] = slice(0, -1); sl_b[axis] = slice(1, None)
        sl_a, sl_b = tuple(sl_a), tuple(sl_b)
        area = (dy * dz if axis == 0 else dx * dz if axis == 1 else dx * dy)
        area = np.broadcast_to(area, (nx, ny, nz))[sl_a]
        da, db = np.broadcast_to(d, (nx, ny, nz))[sl_a], np.broadcast_to(d, (nx, ny, nz))[sl_b]
        ka, kb = kk[sl_a], kk[sl_b]
        with np.errstate(divide="ignore", invalid="ignore"):
            G = area / (da / (2 * ka) + db / (2 * kb))
        G = np.where(np.isfinite(G), G, 0.0)
        a, b = idx[sl_a].ravel(), idx[sl_b].ravel(); G = G.ravel()
        keep = G > 0
        if active is not None:
            act = active.ravel()
            keep &= act[a] & act[b]
        out.append((a[keep], b[keep], G[keep]))
    a = np.concatenate([o[0] for o in out]); b = np.concatenate([o[1] for o in out]); G = np.concatenate([o[2] for o in out])
    return a, b, G


# ---------------------------------------------------------------- DC
def solve_dc(g, net, src, sink, current, use_gpu=True):
    """DC conduction on one net. src/sink = (ref, num). Returns dict with V (volts per cell,
    nan off-net), drop, J (A/mm2 per cell), joule (W per cell)."""
    bd = g.bd
    nx, ny, nz = g.nx, g.ny, g.nz
    sig = 1.0 / RHO_CU
    # in-plane: sheet conductance sigma * t * f ; vertical: via barrels
    sx = np.zeros((nx, ny, nz)); sz = np.zeros((nx, ny, nz))
    active = np.zeros((nx, ny, nz), bool)
    for k, (kd, layer) in enumerate(g.kind):
        if kd == "cu":
            f = g.copper_fraction(layer, net)
            sx[:, :, k] = sig * f                       # conductivity x fraction; thickness is the cell dz
            active[:, :, k] = f > 1e-3
    A = g.via_copper_area(net)                          # mm2 of barrel copper in every cell the via passes
    cellA = g.dx[:, None, None] * g.dy[None, :, None]
    via = A > 0
    sz[via] = sig * (A / cellA)[via]                    # the barrel conducts through planes it has no copper on (anti-pads)
    active |= via
    sx = np.where(active, sx, 0.0); sz = np.where(active, sz, 0.0)
    a, b, G = _links(g, sx, sx, sz, active)
    # keep only the component connected to the source; ground one sink cell
    src_cells = [c for c in g.pad_cells(*src) if active[c]]
    sink_cells = [c for c in g.pad_cells(*sink) if active[c]]
    if not src_cells or not sink_cells:
        raise ValueError("source or sink pad has no copper of net %s under it at this cell size" % net)
    flat = lambda c: (c[0] * ny + c[1]) * nz + c[2]
    n = nx * ny * nz
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    adj = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    ncomp, lab = connected_components(adj, directed=False)
    comp = lab[flat(src_cells[0])]
    if any(lab[flat(c)] != comp for c in sink_cells):
        raise ValueError("source and sink are not connected through copper of %s on this grid (try a smaller --cell)" % net)
    keep = (lab[a] == comp)
    a, b, G = a[keep], b[keep], G[keep]
    members = np.where(lab == comp)[0]
    loc = -np.ones(n, np.int64); loc[members] = np.arange(len(members))
    ra, rb = loc[a], loc[b]
    m = len(members)
    diag = np.zeros(m)
    np.add.at(diag, ra, G); np.add.at(diag, rb, G)
    rhs = np.zeros(m)
    for c in src_cells:
        rhs[loc[flat(c)]] += current / len(src_cells)
    for c in sink_cells:
        rhs[loc[flat(c)]] -= current / len(sink_cells)
    gnd = loc[flat(sink_cells[0])]
    diag[gnd] += 1e3 * diag.max()                     # pin ~0 V
    x = _cg(np.concatenate([ra, rb]), np.concatenate([rb, ra]), np.concatenate([G, G]), diag, rhs, use_gpu=use_gpu)
    V = np.full(n, np.nan); V[members] = x - x[gnd]
    I_link = G * (x[ra] - x[rb])
    joule = np.zeros(n)
    np.add.at(joule, a, 0.5 * G * (x[ra] - x[rb]) ** 2); np.add.at(joule, b, 0.5 * G * (x[ra] - x[rb]) ** 2)
    # current density: in-plane links only, J = I / (t * f * width) with width = the transverse cell size
    J = np.zeros(n)
    ka, kb = a % nz, b % nz
    ia, ja = (a // nz) // ny, (a // nz) % ny
    ib, jb = (b // nz) // ny, (b // nz) % ny
    inplane = (ka == kb)
    t = g.dz[ka] * 1e-3
    f = np.maximum(sx.ravel()[a], sx.ravel()[b]) / sig
    width = np.where(ia != ib, g.dy[np.clip(ja, 0, ny - 1)], g.dx[np.clip(ia, 0, nx - 1)]) * 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        Jl = np.where(inplane & (f > 0), np.abs(I_link) / (t * f * width), 0.0) / 1e6      # A/mm2
    np.maximum.at(J, a, Jl); np.maximum.at(J, b, Jl)
    Vs = np.mean([V[flat(c)] for c in src_cells])
    return {"net": net, "current": current, "drop_V": float(Vs), "R_ohm": float(Vs / current) if current else 0.0,
            "V": V.reshape(nx, ny, nz), "J": J.reshape(nx, ny, nz), "joule": joule.reshape(nx, ny, nz),
            "joule_W": float(joule.sum()), "Jmax": float(J.max()), "cells": int(m)}


# ---------------------------------------------------------------- thermal
def solve_thermal(g, powers, h=10.0, tamb=25.0, extra_q=None, use_gpu=True):
    """powers: {ref: watts}. extra_q: per-cell W array (e.g. Joule heat). Returns dict with T (C) per cell."""
    nx, ny, nz = g.nx, g.ny, g.nz
    n = nx * ny * nz
    a, b, G = _links(g, g.kx, g.ky, g.kz)
    diag = np.zeros(n)
    np.add.at(diag, a, G); np.add.at(diag, b, G)
    q = np.zeros((nx, ny, nz))
    placed = {}
    for ref, P in powers.items():
        cells = g.pad_cells(ref)
        if not cells:
            raise ValueError("no pads for footprint %s" % ref)
        for c in cells:
            q[c] += P / len(cells)
        placed[ref] = len(cells)
    if extra_q is not None:
        q += extra_q
    # convection on the top face of the top cells and the bottom face of the bottom cells
    faceA = (g.dx[:, None] * g.dy[None, :]) * 1e-6
    hA = np.zeros((nx, ny, nz)); hA[:, :, 0] += h * faceA; hA[:, :, -1] += h * faceA
    diag += hA.ravel()
    rhs = q.ravel() + hA.ravel() * tamb
    T = _cg(np.concatenate([a, b]), np.concatenate([b, a]), np.concatenate([G, G]), diag, rhs, use_gpu=use_gpu)
    T = T.reshape(nx, ny, nz)
    return {"T": T, "q": q, "h": h, "tamb": tamb, "P_total": float(q.sum()), "Tmax": float(T.max()), "placed": placed}


def report(g, th, dc=None):
    bd = g.bd
    L = ["thermal: %.2f W in, h %.0f W/m2K both faces, T_amb %.0f C, grid %d x %d x %d cells of %.2f mm"
         % (th["P_total"], th["h"], th["tamb"], g.nx, g.ny, g.nz, g.cell)]
    T = th["T"]
    i, j, k = np.unravel_index(np.argmax(T), T.shape)
    L.append("max %.1f C (+%.1f K) at (%.1f, %.1f) mm on %s" % (T.max(), T.max() - th["tamb"], g.cx[i], g.cy[j], g.kind[k][1] if g.kind[k][0] == "cu" else "dielectric"))
    for layer, k in g.k_cu.items():
        L.append("   %-7s max %.1f C  mean %.1f C" % (layer, T[:, :, k].max(), T[:, :, k][g.inside_outline()].mean()))
    rows = []
    for fp in bd.g.get("footprints", []):
        cells = g.pad_cells(fp["ref"])
        if cells:
            t = max(T[c] for c in cells)
            rows.append((t, fp["ref"], fp["value"]))
    rows.sort(reverse=True)
    L.append("hottest footprints: " + ", ".join("%s (%s) %.1f C" % (r, v, t) for t, r, v in rows[:8]))
    if dc:
        L.append("DC %s: %.2f A from source to sink, drop %.1f mV (%.1f mohm), Joule %.3f W, max current density %.1f A/mm2 over %d copper cells"
                 % (dc["net"], dc["current"], 1e3 * dc["drop_V"], 1e3 * dc["R_ohm"], dc["joule_W"], dc["Jmax"], dc["cells"]))
    return "\n".join(L)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="kicad-fdtd thermal", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("geometry")
    ap.add_argument("--power", nargs="*", default=[], help="REF=W ...")
    ap.add_argument("--h", type=float, default=10.0, help="convection W/m2K on both faces")
    ap.add_argument("--tamb", type=float, default=25.0)
    ap.add_argument("--cell", type=float, default=0.25, help="in-plane cell mm")
    ap.add_argument("--net"); ap.add_argument("--from", dest="src"); ap.add_argument("--to", dest="sink")
    ap.add_argument("--current", type=float, default=0.0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="out")
    a = ap.parse_args(argv)
    bd = geometry.Board(a.geometry)
    t = time.time()
    g = Grid(bd, cell=a.cell)
    print("grid %d x %d x %d cells (%.1f s)" % (g.nx, g.ny, g.nz, time.time() - t))
    dc = None
    if a.net:
        if not (a.src and a.sink and a.current):
            sys.exit("--net needs --from REF.PAD --to REF.PAD --current A")
        dc = solve_dc(g, a.net, tuple(a.src.split(".")), tuple(a.sink.split(".")), a.current, use_gpu=not a.cpu)
    powers = {}
    for s in a.power:
        r, w = s.split("="); powers[r] = float(w)
    th = solve_thermal(g, powers, h=a.h, tamb=a.tamb, extra_q=dc["joule"] if dc else None, use_gpu=not a.cpu)
    print(report(g, th, dc))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "thermal_%s.json" % os.path.basename(a.geometry).split("_")[0])
    json.dump(result_json(g, th, dc), open(path, "w"))
    print("saved", path)
    return 0


def result_json(g, th, dc=None):
    out = {"cell": g.cell, "x": g.x.tolist(), "y": g.y.tolist(), "layers": list(g.k_cu), "h": th["h"], "tamb": th["tamb"],
           "P_total": th["P_total"], "Tmax": th["Tmax"], "report": report(g, th, dc),
           "T": {layer: th["T"][:, :, k].tolist() for layer, k in g.k_cu.items()}}
    if dc:
        out["dc"] = {"net": dc["net"], "current": dc["current"], "drop_V": dc["drop_V"], "R_ohm": dc["R_ohm"], "joule_W": dc["joule_W"], "Jmax": dc["Jmax"],
                     "V": {layer: np.nan_to_num(dc["V"][:, :, k], nan=-1.0).tolist() for layer, k in g.k_cu.items()},
                     "J": {layer: dc["J"][:, :, k].tolist() for layer, k in g.k_cu.items()}}
    return out


if __name__ == "__main__":
    sys.exit(main())
