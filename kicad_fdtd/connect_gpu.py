"""What is joined to what on a board, and what touches that should not, on the GPU.

    python -m kicad_fdtd.connect_gpu geometry.json

The fast stand-in for the two things a DRC is run for between routing steps: the connections still open, and
copper of two nets in contact. KiCad's DRC answers both exactly and takes about a minute of one CPU core for a board
of 8,000 tracks, a board file read and a zone fill included; a routing run asks many times. Here the board's copper
(export.board_geometry: every track, via, pad and filled zone as polygons by layer, the same dict the field solvers
start from) is drawn once on the card and read back:

  draw      Each layer's polygons are filled on a grid of `cell` mm (voxel_gpu's fill: one thread a polygon edge).
            Two fills are kept, of the nets' numbers and of their squares, so a cell under copper of two different
            nets is known for what it is: the mean of the numbers alone cannot tell nets 1 and 3 from net 2 twice.
  label     The copper of each layer is split into its connected pieces (cupyx.scipy.ndimage.label, 4-connected).
  join      A via or a plated hole joins the pieces it stands on, layer to layer: a union-find over (layer, piece),
            a few thousand unions, on the CPU.
  read      A net's pads that end up in more than one group are open connections (groups - 1, as KiCad counts
            its ratsnest). A piece holding cells of two nets is a short, reported once with the place of a cell
            where the two meet.

It is a stand-in, not the judge. On a grid, copper is where a cell's centre is covered: two tracks at exactly their
clearance stay apart by whole empty cells, but a join that holds by less than a cell of overlap (the sliver joins
kiPlace's joins.py closes) can read as open, and clearance short of contact is not looked for at all. KiCad's DRC
stays the last word, once, on the finished board; tests/check_connect_gpu.py holds this to it on saved boards.
"""
import json
import sys
import time

import cupy as cp
import numpy as np
from cupyx.scipy import ndimage
from shapely.geometry import Point

from . import geometry, voxel_gpu

CELL = 0.05                 # mm: the router's own grid pitch
VIA_ARC = 6                 # segments a quarter of a via's pad is drawn with
FOUR = cp.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def _fill(polys, ids, xs, ys):
    """How many polygons hold each grid point, and the sum of their ids: (count, sum) int32 [len(ys), len(xs)]."""
    lay = voxel_gpu.Layer(polys, ids)
    cnt = cp.zeros((len(ys), len(xs)), cp.int32)
    tot = cp.zeros((len(ys), len(xs)), cp.int32)
    if lay.n:
        voxel_gpu._launch("fill", lay.n, lay.e, lay.turn, lay.cid, np.int32(lay.n), xs, np.int32(len(xs)), ys,
                          np.int32(len(ys)), cnt, tot)
        cnt, tot = cp.cumsum(cnt, axis=1, dtype=cp.int32), cp.cumsum(tot, axis=1, dtype=cp.int32)
    return cnt, tot


def span(bd, v):
    """The layers a via or plated hole passes, top to bottom."""
    top = bd.layers.index(v["top"]) if v.get("top") in bd.layers else 0
    bottom = bd.layers.index(v["bottom"]) if v.get("bottom") in bd.layers else len(bd.layers) - 1
    return bd.layers[min(top, bottom):max(top, bottom) + 1]


class _Groups(object):
    """Union-find over (layer, piece)."""

    def __init__(self):
        self.up = {}

    def find(self, a):
        up = self.up
        root = a
        while up.setdefault(root, root) != root:
            root = up[root]
        while up[a] != root:
            up[a], a = root, up[a]
        return root

    def join(self, a, b):
        self.up[self.find(a)] = self.find(b)


def check(g, cell=CELL, zone_fill=None):
    """The board geometry dict `g` (or a path to its JSON) read as above. zone_fill: the pours as fill_gpu.fills
    made them, {layer: net number at each cell}, in place of the fills the file holds (stale once copper has been
    added since KiCad last filled them). Returns
        {"open": {net: open connections}, "open_at": {net: [[x, y] of a pad in each group cut off from the net's
         largest]}, "shorts": [{"nets": [a, b], "layer", "x", "y"}], "nets": nets with pads, "cells": [nx, ny],
         "seconds": s}
    x, y in board mm."""
    t0 = time.time()
    bd = geometry.Board(g)
    names = [""] + sorted(n for n in (bd.g.get("nets") or []) if n)
    number = {n: i for i, n in enumerate(names)}
    xs = cp.asarray(np.arange(cell / 2, bd.w, cell))
    ys = cp.asarray(np.arange(-bd.h + cell / 2, 0.0, cell))
    pieces, owner, shorts = {}, {}, []
    vias = bd.vias()
    on_layer = {layer: [] for layer in bd.layers}        # a via's pad on every layer it passes (the geometry's copper
    for v in vias:                                       # holds tracks, pads and fills; vias come as a list)
        if v["kind"] == "via" and number.get(v["net"], 0):
            disc = Point(v["x"], v["y"]).buffer(v["r"], VIA_ARC)
            for layer in span(bd, v):
                on_layer[layer].append((number[v["net"]], disc))
    for layer in bd.layers:
        polys = [(number.get(net, 0), P) for net, kind, P in bd.polys(layer)
                 if number.get(net, 0) and not (zone_fill is not None and kind == "zone")] + on_layer[layer]
        ps = [P for _i, P in polys]
        cnt, tot = _fill(ps, [i - 1 for i, _P in polys], xs, ys)                # (Layer stores id + 1: the net number)
        _c, sq = _fill(ps, [i * i - 1 for i, _P in polys], xs, ys)
        mixed = (cnt > 1) & (cnt.astype(cp.int64) * sq != tot.astype(cp.int64) * tot)
        net = cp.where(cnt > 0, tot // cp.maximum(cnt, 1), 0).astype(cp.int32)  # (where mixed: no net's own number)
        if zone_fill is not None:                # the pours as made on the card: copper where nothing else is
            pour = zone_fill[layer]
            mixed |= (pour != 0) & (cnt > 0) & (net != pour)
            net = cp.where(cnt > 0, net, pour)
            cnt = cp.where(pour != 0, cp.maximum(cnt, 1), cnt)
        lab, n = ndimage.label(cnt > 0, structure=FOUR)
        pieces[layer] = lab
        # each piece's lowest and highest net number: the same for a piece of one net
        idx = cp.arange(1, n + 1)
        clean = cp.where(mixed, 0, net)
        lo = ndimage.minimum(cp.where(clean > 0, clean, len(names)), lab, idx) if n else cp.zeros(0, cp.int32)
        hi = ndimage.maximum(clean, lab, idx) if n else cp.zeros(0, cp.int32)
        lo, hi = cp.asnumpy(lo).astype(np.int64), cp.asnumpy(hi).astype(np.int64)
        owner[layer] = hi                                                       # a piece's net (its highest, if two)
        bad = np.nonzero((lo != hi) & (lo < len(names)))[0]
        if len(bad) or bool(mixed.any()):
            # the place: a cell of the piece where the other net begins
            marks = cp.asnumpy(cp.stack(cp.nonzero(mixed), axis=1)[:1])
            for k in bad:
                # (of the two nets, the one with fewer cells in the piece: that is where they meet, or near it)
                both = [(lab == int(k) + 1) & (clean == int(n)) for n in (lo[k], hi[k])]
                where = cp.asnumpy(cp.stack(cp.nonzero(min(both, key=lambda m: int(m.sum()))), axis=1)[:1])
                r, c = (where[0] if len(where) else (marks[0] if len(marks) else (0, 0)))
                shorts.append({"nets": [names[int(lo[k])], names[int(hi[k])]], "layer": layer,
                               "x": round(float(xs[int(c)]) + bd.x0, 3), "y": round(bd.y0 - float(ys[int(r)]), 3)})
            if not len(bad) and len(marks):
                r, c = marks[0]
                shorts.append({"nets": ["?", "?"], "layer": layer, "x": round(float(xs[int(c)]) + bd.x0, 3),
                               "y": round(bd.y0 - float(ys[int(r)]), 3)})
    nx, ny = len(xs), len(ys)

    def cell_of(x, y):
        return (min(ny - 1, max(0, int((y + bd.h) / cell))), min(nx - 1, max(0, int(x / cell))))

    def piece_at(layer, x, y):
        r, c = cell_of(x, y)
        return int(pieces[layer][r, c])

    groups = _Groups()
    for v in vias:                                       # vias and plated holes: the pieces they stand on, joined
        on = [(layer, piece_at(layer, v["x"], v["y"])) for layer in span(bd, v)]
        on = [(layer, p) for layer, p in on if p]
        for a, b in zip(on, on[1:]):
            groups.join(a, b)
    by_net = {}
    for p in bd.pads:
        if not p["net"] or not p["layers"]:
            continue
        x, y = bd.pt(p["x"], p["y"])
        mine = [(layer, piece_at(layer, x, y)) for layer in p["layers"] if layer in pieces]
        mine = [m for m in mine if m[1]]
        for a, b in zip(mine, mine[1:]):                 # a pad on several layers is one piece of copper
            groups.join(a, b)
        by_net.setdefault(p["net"], []).append((mine[0] if mine else ("pad", p["ref"], p["num"]), p["x"], p["y"]))
    opens, open_at = {}, {}
    for net, nodes in by_net.items():
        at = {}                                          # each group of the net's pads: one of its pads' places
        for n, x, y in nodes:
            at.setdefault(groups.find(n), []).append([x, y])
        if len(at) > 1:
            opens[net] = len(at) - 1
            # the pads cut off from the net's largest group, a place for each group: where a join is wanted
            rest = sorted(at.values(), key=len)[:-1]
            open_at[net] = [g[0] for g in rest]
    return {"open": opens, "open_at": open_at, "shorts": shorts, "nets": len(by_net), "cells": [nx, ny],
            "seconds": round(time.time() - t0, 2)}


def main(argv):
    if len(argv) != 1:
        sys.exit(__doc__)
    with open(argv[0]) as fh:
        res = check(json.load(fh))
    print("%d open connection(s) on %d net(s), %d short(s); %d x %d cells, %.1f s" % (
        sum(res["open"].values()), len(res["open"]), len(res["shorts"]), res["cells"][0], res["cells"][1],
        res["seconds"]))
    for net, n in sorted(res["open"].items()):
        print("  open: %s, %d" % (net, n))
    for s in res["shorts"]:
        print("  short: %s / %s on %s at (%.2f, %.2f)" % (s["nets"][0], s["nets"][1], s["layer"], s["x"], s["y"]))
    return 1 if res["open"] or res["shorts"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
