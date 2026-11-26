"""Zone fills on the GPU: where each pour's copper ends up, from its outline and the copper round it.

    python -m kicad_fdtd.fill_gpu geometry.json          the fills made here against the ones the file holds

KiCad's zone filler is exact and takes 9 to 17 s of one CPU core on a board of 100 zones and 8,000 tracks, and a
routing run calls it six to eight times: every step that adds or moves copper fills again so the next can ask where
the pours are. Here the same question is answered on the card, on the grid connect_gpu already draws the board on:

  per layer   the tracks, pads and vias are drawn once with their nets (connect_gpu's fill of polygon edges), and
              the board's edge and the no-fill rule areas once.
  per zone    highest priority first:
                room      inside the zone's outline and the board (less the edge clearance), outside the no-fill
                          areas;
                clear     further than the zone's clearance from every cell of another net's copper and of another
                          net's higher-priority fill (a distance transform: one pass over the grid);
                wide      what is left less the necks narrower than the zone's minimum width (opened by half the
                          width: two more distance transforms);
                joined    the pieces that hold copper of the zone's own net (a pad, a via, a track), when the zone
                          removes islands: the rest are dropped.
              Its fill then stands in the way of every lower-priority zone of another net.

                relieved  where the zone joins pads by thermal reliefs, the gap round each such pad of its net is
                          taken out and the spokes laid back across it, as wide as the zone says (or the pad, where
                          it has its own). A spoke is laid only when all of it may be fill: a pin walled in by its
                          neighbours' clearances gets a gap and no spoke, and reads as open, which it is (MP-BOY
                          _kl17: the ground pin between two pairs at the CM5 socket, 0.2 mm wide on a 0.4 mm pitch,
                          where a 0.5 mm spoke cannot pass; with the pads taken as solid the check called it joined).

What it leaves out, on purpose and said: corner smoothing and hatching (none of this board's zones use them), and
the netclass clearance of the other net's item (the zone's own clearance is used for all). The grid's pitch is the
error bar on every edge: the cells kept are the ones whose centres are clear, so a fill here may differ from KiCad's
by a cell along its boundary. It is for asking where the pours are between routing steps; the board that is made is
filled by KiCad.

tests/check_fill_gpu.py holds the result to the fills KiCad made, layer by layer and net by net.
"""
import json
import sys
import time

import cupy as cp
import numpy as np
from cupyx.scipy import ndimage
from shapely.geometry import Point, Polygon

from . import connect_gpu, geometry

ISLANDS_ALWAYS = 0          # KiCad's ISLAND_REMOVAL_MODE_ALWAYS: a piece joined to nothing of its net is not kept
# KiCad's ZONE_CONNECTION values: a pad's own (INHERIT: the zone's), and the zone's
INHERIT, NO_JOIN, THERMAL, SOLID, THERMAL_THROUGH_HOLE = -1, 0, 1, 2, 3
SPOKES = 4                  # a relief's spokes, a quarter turn apart


def _poly(bd, outline, holes):
    p = Polygon(bd.xy(outline), [bd.xy(h) for h in holes or [] if len(h) >= 3])
    return p if p.is_valid else p.buffer(0)


def _mask(polys, xs, ys):
    """True where any of the polygons holds the grid point: bool [len(ys), len(xs)]."""
    if not polys:
        return cp.zeros((len(ys), len(xs)), bool)
    cnt, _tot = connect_gpu._fill(polys, [0] * len(polys), xs, ys)
    return cnt > 0


def _far(obstacle, mm, cell):
    """True where the nearest obstacle cell is further off than mm (everywhere, where there is no obstacle)."""
    if mm <= 0 or not bool(obstacle.any()):
        return ~obstacle
    return ndimage.distance_transform_edt(~obstacle) * cell > mm


def _relieved(bd, zone, layer, xs, ys, cell):
    """(gaps, spokes, count) for a zone on a layer: a bool map of the relief gaps round the pads of its net it
    joins by thermal reliefs, and the spokes across them as an int map of spoke numbers 1..count (0: no spoke);
    (None, None, 0) when it joins no pad that way."""
    import math
    pads, rects = [], []
    for p in bd.pads:
        if p["net"] != zone["net"] or layer not in p["layers"] or "connect" not in p:
            continue
        how = zone.get("connect", SOLID) if p["connect"] == INHERIT else p["connect"]
        if how == THERMAL_THROUGH_HOLE:
            how = THERMAL if p.get("pth") else SOLID
        if how not in (THERMAL, NO_JOIN):
            continue
        x, y = bd.pt(p["x"], p["y"])
        hit = [P for net, kind, P in bd.polys(layer) if kind == "pad" and net == p["net"] and P.contains(Point(x, y))]
        if not hit:
            continue
        gap = p["gap"] if p.get("gap") is not None else zone.get("thermal_gap", 0.0)
        pads.append((hit[0], gap))
        if how == NO_JOIN:
            continue
        w, h = p["size"]
        width = p["spoke"] if p.get("spoke") is not None else zone.get("spoke_width", 0.0)
        reach = max(w, h) / 2.0 + gap + width
        for k in range(SPOKES):
            # (the board's y runs down and this frame's up: the turn is mirrored with it)
            a = -math.radians(p.get("angle", 0.0) + p.get("spoke_deg", 0.0) + k * 360.0 / SPOKES)
            ux, uy = math.cos(a), math.sin(a)
            rects.append(Polygon([(x - uy * width / 2, y + ux * width / 2), (x + uy * width / 2, y - ux * width / 2),
                                  (x + uy * width / 2 + ux * reach, y - ux * width / 2 + uy * reach),
                                  (x - uy * width / 2 + ux * reach, y + ux * width / 2 + uy * reach)]))
    if not pads:
        return None, None, 0
    gaps = cp.zeros((len(ys), len(xs)), bool)
    for width_mm in sorted({g_ for _P, g_ in pads}):
        own = _mask([P for P, g_ in pads if g_ == width_mm], xs, ys)
        gaps |= ~own & ~_far(own, width_mm, cell)
    # the spokes, each with its own number (0: none; where two cross, neither's)
    cnt, tot = connect_gpu._fill(rects, list(range(len(rects))), xs, ys)
    return gaps, cp.where(cnt == 1, tot, 0), len(rects)



def fills(g, cell=connect_gpu.CELL):
    """The geometry dict `g` (export.board_geometry, with its "zones"), every zone filled as above. Returns
    ({layer: int32 [ny, nx] of the net number filled at each cell, 0 for none}, names, xs, ys, seconds): the net
    numbers index `names` (0 is no net), xs / ys are the grid's points in the board's own frame."""
    t0 = time.time()
    bd = geometry.Board(g)
    names = [""] + sorted(n for n in (bd.g.get("nets") or []) if n)
    number = {n: i for i, n in enumerate(names)}
    xs = cp.asarray(np.arange(cell / 2, bd.w, cell))
    ys = cp.asarray(np.arange(-bd.h + cell / 2, 0.0, cell))
    edge_clr = float(g.get("edge_clearance") or 0.0)
    inside = _mask([bd.outline], xs, ys)
    board_ok = inside & _far(~inside, edge_clr, cell)
    vias = [v for v in bd.vias() if v["kind"] == "via" and number.get(v["net"], 0)]
    out = {}
    for layer in bd.layers:
        # highest priority first; between two of one priority, KiCad's own rule: the later UUID
        # (ZONE::HigherPriority), which decides who gets the ground where two such zones of different nets overlap
        zones = sorted((z for z in g.get("zones") or [] if z["layer"] == layer and number.get(z["net"], 0)),
                       key=lambda z: (z["priority"], z.get("uuid", "")), reverse=True)
        filled = cp.zeros((len(ys), len(xs)), cp.int32)
        out[layer] = filled
        if not zones:
            continue
        # the layer's own copper, by net: tracks and pads from the geometry, the vias that pass it as discs
        polys = [(number[net], P) for net, kind, P in bd.polys(layer) if kind != "zone" and number.get(net, 0)]
        polys += [(number[v["net"]], Point(v["x"], v["y"]).buffer(v["r"], connect_gpu.VIA_ARC)) for v in vias
                  if layer in connect_gpu.span(bd, v)]
        cnt, tot = connect_gpu._fill([P for _i, P in polys], [i - 1 for i, _P in polys], xs, ys)
        net = cp.where(cnt > 0, tot // cp.maximum(cnt, 1), 0).astype(cp.int32)
        # unnetted copper (a mounting pad with no net) stands in every zone's way
        bare = _mask([P for n_, kind, P in bd.polys(layer) if kind != "zone" and not number.get(n_, 0)], xs, ys)
        # ... and so does every hole that is not plated, out to the hole clearance less the zone's own (which every
        # obstacle is then kept clear by)
        bare |= _mask([Point(*bd.pt(h["x"], h["y"])).buffer(h["d"] / 2.0 + float(g.get("hole_clearance") or 0.0),
                                                           connect_gpu.VIA_ARC) for h in g.get("holes") or []], xs, ys)
        keep_out = _mask([_poly(bd, r["outline"], r["holes"]) for r in g.get("no_fill") or [] if r["layer"] == layer],
                         xs, ys)
        room_all = board_ok & ~keep_out
        for z in zones:
            n = number[z["net"]]
            room = _mask([_poly(bd, z["outline"], z["holes"])], xs, ys) & room_all
            if not bool(room.any()):
                continue
            other = ((net != 0) & (net != n)) | bare | ((filled != 0) & (filled != n))
            piece = room & _far(other, float(z["clearance"]), cell) & (filled == 0)
            half = float(z["min_width"]) / 2.0
            if half > cell:                                  # necks under the minimum width go: opened by half of it
                core = ndimage.distance_transform_edt(piece) * cell > half
                piece &= ndimage.distance_transform_edt(~core) * cell <= half if bool(core.any()) else core
            gaps, spokes, count = _relieved(bd, z, layer, xs, ys, cell)
            if gaps is not None:                             # the relief gaps out, and back the spokes that fit whole
                may = room & _far(other, float(z["clearance"]), cell) & (filled == 0)
                across = spokes * gaps
                need = cp.bincount(across.ravel(), minlength=count + 1)
                have = cp.bincount((across * may).ravel(), minlength=count + 1)
                whole = (need > 0) & (have == need)
                whole[0] = False
                piece = (piece & ~gaps) | (whole[across] & may)
            if z["islands"] == ISLANDS_ALWAYS:
                lab, count = ndimage.label(piece | (net == n), structure=connect_gpu.FOUR)
                if count:
                    # a piece is joined when it shares a labelled region with copper of the net
                    held = cp.zeros(count + 1, bool)
                    held[cp.unique(lab[net == n])] = True
                    held[0] = False
                    piece &= held[lab]
            filled[piece] = n
    return out, names, xs, ys, round(time.time() - t0, 2)


def file_fills(g, names, xs, ys):
    """The fills the file holds (the geometry's copper of kind "zone"), on the same grid: {layer: int32 net map}."""
    bd = geometry.Board(g)
    number = {n: i for i, n in enumerate(names)}
    out = {}
    for layer in bd.layers:
        polys = [(number[net], P) for net, kind, P in bd.polys(layer) if kind == "zone" and number.get(net, 0)]
        cnt, tot = connect_gpu._fill([P for _i, P in polys], [i - 1 for i, _P in polys], xs, ys)
        out[layer] = cp.where(cnt > 0, tot // cp.maximum(cnt, 1), 0).astype(cp.int32)
    return out


def compare(g, cell=connect_gpu.CELL):
    """The fills made here against the file's: [{"layer", "net", "file_mm2", "here_mm2", "only_file_mm2",
    "only_here_mm2"}] for every net filled on a layer by either, and the seconds the fill took."""
    mine, names, xs, ys, secs = fills(g, cell)
    theirs = file_fills(g, names, xs, ys)
    rows, area = [], cell * cell
    for layer in mine:
        a, b = mine[layer], theirs[layer]
        for n in cp.asnumpy(cp.unique(cp.concatenate([a.ravel()[::1], b.ravel()[::1]]))):
            if not n:
                continue
            ia, ib = a == n, b == n
            rows.append({"layer": layer, "net": names[int(n)], "file_mm2": float(ib.sum()) * area,
                         "here_mm2": float(ia.sum()) * area, "only_file_mm2": float((ib & ~ia).sum()) * area,
                         "only_here_mm2": float((ia & ~ib).sum()) * area})
    return rows, secs


def main(argv):
    if len(argv) != 1:
        sys.exit(__doc__)
    with open(argv[0]) as fh:
        rows, secs = compare(json.load(fh))
    print("filled in %.1f s" % secs)
    for r in rows:
        print("  %-7s %-16s file %8.1f mm2, here %8.1f, only in the file %7.2f, only here %7.2f" % (
            r["layer"], r["net"][-16:], r["file_mm2"], r["here_mm2"], r["only_file_mm2"], r["only_here_mm2"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
