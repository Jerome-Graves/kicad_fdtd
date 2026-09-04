"""Assign every Yee E-edge to a conductor, exactly as the solver will use it,
and check for merged conductors BEFORE solving.

Yee layout on node lines x[i], y[j], z[k]:
  Ex(i, j, k) is the edge from node (i, j, k) to (i+1, j, k), midpoint (mx[i], y[j], z[k])
  Ey(i, j, k)                        (i, j+1, k)            (x[i], my[j], z[k])
  Ez(i, j, k)                        (i, j, k+1)            (x[i], y[j], mz[k])
Copper sheets sit on z planes: an Ex/Ey edge on that plane is metal if its
midpoint is inside a polygon of that layer. Via barrels are solid cylinders:
an edge is metal if its midpoint is within r of the axis and inside the
barrel's z range. The result is one integer net id per edge (-1 = none).
"""
import numpy as np
import shapely
from shapely.geometry import box


def assign(bd, lines, tie_nets=None, skip_via_near=(), window=None):
    tie = tie_nets or {}
    x, y, z = lines["x"], lines["y"], lines["z"]
    nx, ny, nz = len(x), len(y), len(z)
    mx, my, mz = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2, (z[:-1] + z[1:]) / 2
    ex = np.full((nx - 1, ny, nz), -1, np.int16)
    ey = np.full((nx, ny - 1, nz), -1, np.int16)
    ez = np.full((nx, ny, nz - 1), -1, np.int16)
    names = []

    def nid(net):
        net = tie.get(net, net)
        if net not in names:
            names.append(net)
        return names.index(net)
    zidx = {layer: int(np.argmin(abs(z - bd.z[layer]))) for layer in bd.layers}
    for layer in bd.layers:
        assert abs(z[zidx[layer]] - bd.z[layer]) < 1e-6, "copper plane %s is not on a z line" % layer
    win = box(*window) if window else None
    XX, YY = np.meshgrid(mx, y, indexing="ij")
    XY, YX = np.meshgrid(x, my, indexing="ij")
    for layer in bd.layers:
        k = zidx[layer]
        for net, kind, P in bd.polys(layer):
            if win is not None and not P.intersects(win):
                continue
            b = P.bounds
            ix = np.where((mx >= b[0] - 1e-9) & (mx <= b[2] + 1e-9))[0]
            jy = np.where((y >= b[1] - 1e-9) & (y <= b[3] + 1e-9))[0]
            if len(ix) and len(jy):
                sub = shapely.contains_xy(P, XX[np.ix_(ix, jy)].ravel(), YY[np.ix_(ix, jy)].ravel()).reshape(len(ix), len(jy))
                if sub.any():
                    blk = ex[np.ix_(ix, jy, [k])][:, :, 0]
                    blk[sub] = nid(net)
                    ex[np.ix_(ix, jy, [k])] = blk[:, :, None]
            ix = np.where((x >= b[0] - 1e-9) & (x <= b[2] + 1e-9))[0]
            jy = np.where((my >= b[1] - 1e-9) & (my <= b[3] + 1e-9))[0]
            if len(ix) and len(jy):
                sub = shapely.contains_xy(P, XY[np.ix_(ix, jy)].ravel(), YX[np.ix_(ix, jy)].ravel()).reshape(len(ix), len(jy))
                if sub.any():
                    blk = ey[np.ix_(ix, jy, [k])][:, :, 0]
                    blk[sub] = nid(net)
                    ey[np.ix_(ix, jy, [k])] = blk[:, :, None]
    n_via = 0
    for v in bd.vias():
        if win is not None and not win.contains(shapely.geometry.Point(v["x"], v["y"])):
            continue
        if any(np.hypot(v["x"] - px, v["y"] - py) < 0.6 for px, py in skip_via_near):
            continue
        k = nid(v["net"])
        r = v["r"]
        zb, zt = min(v["zb"], v["zt"]), max(v["zb"], v["zt"])
        kz = np.where((z >= zb - 1e-9) & (z <= zt + 1e-9))[0]            # planes the barrel crosses
        kzm = np.where((mz >= zb - 1e-9) & (mz <= zt + 1e-9))[0]         # z-edges inside the barrel
        ix = np.where((mx >= v["x"] - r) & (mx <= v["x"] + r))[0]; jy = np.where((y >= v["y"] - r) & (y <= v["y"] + r))[0]
        if len(ix) and len(jy):
            d = np.hypot(XX[np.ix_(ix, jy)] - v["x"], YY[np.ix_(ix, jy)] - v["y"]) <= r
            for kk in kz:
                blk = ex[np.ix_(ix, jy, [kk])][:, :, 0]; blk[d] = k; ex[np.ix_(ix, jy, [kk])] = blk[:, :, None]
        ix = np.where((x >= v["x"] - r) & (x <= v["x"] + r))[0]; jy = np.where((my >= v["y"] - r) & (my <= v["y"] + r))[0]
        if len(ix) and len(jy):
            d = np.hypot(XY[np.ix_(ix, jy)] - v["x"], YX[np.ix_(ix, jy)] - v["y"]) <= r
            for kk in kz:
                blk = ey[np.ix_(ix, jy, [kk])][:, :, 0]; blk[d] = k; ey[np.ix_(ix, jy, [kk])] = blk[:, :, None]
        ix = np.where((x >= v["x"] - r) & (x <= v["x"] + r))[0]; jy = np.where((y >= v["y"] - r) & (y <= v["y"] + r))[0]
        if len(ix) and len(jy) and len(kzm):
            NX, NY = np.meshgrid(x[ix], y[jy], indexing="ij")
            d = np.hypot(NX - v["x"], NY - v["y"]) <= r
            for kk in kzm:
                blk = ez[np.ix_(ix, jy, [kk])][:, :, 0]; blk[d] = k; ez[np.ix_(ix, jy, [kk])] = blk[:, :, None]
        n_via += 1
    return {"ex": ex, "ey": ey, "ez": ez, "names": names, "n_via": n_via}


def merge_check(vox, lines, nets_of_interest, tie_nets=None):
    """Nodes touched by edges of two different conductors = a short in the
    solver. Returns [(netA, netB, x, y, z)] for nodes involving nets_of_interest."""
    tie = tie_nets or {}
    ex, ey, ez, names = vox["ex"], vox["ey"], vox["ez"], vox["names"]
    x, y, z = lines["x"], lines["y"], lines["z"]
    nx, ny, nz = len(x), len(y), len(z)
    NONE = -1
    # per node: collect ids of the up to 6 touching edges
    ids = [np.full((nx, ny, nz), NONE, np.int16) for _ in range(6)]
    ids[0][1:, :, :] = ex; ids[1][:-1, :, :] = ex
    ids[2][:, 1:, :] = ey; ids[3][:, :-1, :] = ey
    ids[4][:, :, 1:] = ez; ids[5][:, :, :-1] = ez
    stack = np.stack(ids, axis=-1)
    mn = np.where(stack >= 0, stack, 32000).min(axis=-1)
    mx_ = stack.max(axis=-1)
    bad = (mx_ >= 0) & (mn < 32000) & (mn != mx_)
    out = []
    watch = {tie.get(n, n) for n in nets_of_interest}
    for i, j, k in zip(*np.where(bad)):
        s = sorted({int(v) for v in stack[i, j, k] if v >= 0})
        nm = [names[v] for v in s]
        if any(n in watch for n in nm):
            out.append((nm[0], nm[1], float(x[i]), float(y[j]), float(z[k])))
    return out
