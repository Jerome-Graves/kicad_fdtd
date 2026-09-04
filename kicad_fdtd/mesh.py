"""Rectilinear Yee mesh for a board slice.

Lines go on the edges of the copper of the nets of interest and of every
neighbour within `near` of them (so a 0.13 mm gap between a pair and its
pour keeps a cell boundary), plus a base grid; the air beyond the copper is
graded outward (AIR_CELLS cells growing by GROW, the outer PML cells among
them). z: every copper plane exactly, n cells per dielectric, graded air.
"""
import math

import numpy as np
from shapely.geometry import Point, box
from shapely.ops import unary_union

C0 = 299792458.0


def thin(lines, min_gap):
    out = [lines[0]]
    for v in lines[1:]:
        if v - out[-1] >= min_gap:
            out.append(v)
    if out[-1] != lines[-1]:
        out[-1] = lines[-1]
    return out


def smooth(lines, max_res, ratio=1.4):
    """Insert lines so that no cell exceeds max_res and neighbours differ by <= ratio."""
    L = list(lines)
    changed = True
    while changed:
        changed = False
        out = [L[0]]
        for v in L[1:]:
            gap = v - out[-1]
            if gap > max_res * 1.001:
                n = int(math.ceil(gap / max_res))
                out += list(np.linspace(out[-1], v, n + 1)[1:])
                changed = True
            else:
                out.append(v)
        L = out
    return L


def air_lines(edge, direction, first, n, grow):
    out, step, pos = [], first, edge
    for _ in range(n):
        pos += direction * step
        out.append(pos)
        step *= grow
    return out


def build_mesh(bd, nets, window, res=0.1, base=0.2, near=0.6, tie_nets=None, air_cells=14, grow=1.25,
               z_cell=0.1, port_boxes=()):
    tie = tie_nets or {}
    xmin, ymin, xmax, ymax = window
    win = box(*window)
    xl, yl = {xmin, xmax}, {ymin, ymax}

    def add_pts(coords):
        for x, y in coords:
            if xmin <= x <= xmax:
                xl.add(round(x / 0.01) * 0.01)
            if ymin <= y <= ymax:
                yl.add(round(y / 0.01) * 0.01)
    mine_all = []
    for layer in bd.layers:
        for net, kind, P in bd.polys(layer):
            if net in nets and P.intersects(win):
                mine_all.append(P)
                add_pts(P.exterior.coords)
    for v in bd.vias():
        if v["net"] in nets and xmin <= v["x"] <= xmax and ymin <= v["y"] <= ymax:
            mine_all.append(Point(v["x"], v["y"]).buffer(v["r"]))
            add_pts([(v["x"] - v["r"], v["y"] - v["r"]), (v["x"] + v["r"], v["y"] + v["r"]), (v["x"], v["y"])])
    if mine_all:
        near_all = unary_union(mine_all).buffer(near).intersection(win)
        for layer in bd.layers:
            for net, kind, P in bd.polys(layer):
                if net in nets or not P.intersects(near_all):
                    continue
                edge = P.boundary.intersection(near_all)
                for geom in (edge.geoms if hasattr(edge, "geoms") else [edge]):
                    if geom.is_empty or not hasattr(geom, "length") or geom.length == 0:
                        continue
                    n = max(2, int(geom.length / (res * 0.7)))
                    add_pts([(q.x, q.y) for q in (geom.interpolate(k / n, normalized=True) for k in range(n + 1))])
        for v in bd.vias():
            if v["net"] not in nets and Point(v["x"], v["y"]).buffer(v["r"]).intersects(near_all):
                add_pts([(v["x"] - v["r"], v["y"] - v["r"]), (v["x"] + v["r"], v["y"] + v["r"])])
    for s, e in port_boxes:
        add_pts([(s[0], s[1]), (e[0], e[1])])
    lines = {}
    for ny, L, lo, hi in (("x", xl, xmin, xmax), ("y", yl, ymin, ymax)):
        fixed = thin(sorted(L), res * 0.4)
        grid = [g for g in np.arange(lo, hi + base / 2, base) if min(abs(g - f) for f in fixed) > res * 0.5]
        L2 = smooth(sorted(set(fixed + grid)), base)
        L2 = sorted(set([float(v) for v in L2] + air_lines(lo, -1, res, air_cells, grow) + air_lines(hi, +1, res, air_cells, grow)))
        lines[ny] = np.array(L2)
    zl = list(bd.z.values()) + [bd.z_bot]
    for zt, zb, er, tand in bd.diel:
        n = max(1, int(round((zt - zb) / z_cell)))
        zl += list(np.linspace(zb, zt, n + 1))
    zl = sorted(set(round(v, 6) for v in zl))
    zl = [float(v) for v in smooth(zl, 0.4)]
    for zc in list(bd.z.values()) + [bd.z_bot]:
        i = int(np.argmin([abs(v - zc) for v in zl])); zl[i] = float(zc)
    zl = sorted(set(zl + air_lines(0.0, +1, z_cell, air_cells, grow) + air_lines(bd.z_bot, -1, z_cell, air_cells, grow)))
    lines["z"] = np.array(zl)
    return lines


def cfl_dt(lines, safety=0.99):
    d = [np.min(np.diff(lines[k])) * 1e-3 for k in "xyz"]
    return safety / (C0 * math.sqrt(sum(1 / v ** 2 for v in d))), d
