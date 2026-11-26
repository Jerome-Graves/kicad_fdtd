"""The board on the card for as long as a routing run lasts: one process that answers for the pours.

    python -m kicad_fdtd.gpu_server            prints "PORT <n>" and serves on 127.0.0.1 until told to quit

The fill and the connectivity check (fill_gpu, connect_gpu) take seconds; what made them no faster than KiCad's own
between routing steps was everything round them: a process started for each question (Python, CuPy and its kernels:
several seconds) and none of it kept. Here the process is started once, when the run starts, and is warm by the time
a step needs it; a step sends the board's geometry once and asks as often as it likes.

A request is one JSON object on a line, the answer one JSON object on a line ({"error": text} when it failed):

  {"op": "load", "path": GEOMETRY.json}       the board (export.board_geometry's JSON). Its pours are filled here,
                                              once, the first time a question needs them.
  {"op": "open"}                              -> {"open": {net: connections}, "shorts": [...], "seconds"}: what is
                                              still to join and what touches that should not, on the fills made here
  {"op": "stitch", "net", "via_mm",           what a stitching pass needs to know of a net's pours, written to `out`
   "clearance_mm", "out": FILE.npz}           (numpy) and returned:
        in the file    for each layer the net is poured on, where a via pad of via_mm lies whole in its fill
                       ("in:<layer>", bytes); how many layers that is at each cell ("count"); where the pad would
                       come within the clearance of another net's pour on an outer layer ("foreign"); and the grid
                       ("frame": x0, y0, height, cell - a board point (x, y) mm is cell [int((height - (y - y0)) /
                       cell), int((x - x0) / cell)])
        returned       {"islands": [{"layer", "mm2", "spots": [[x, y], ...], "pads": ["REF.NUM", ...]}], "seconds"}:
                       each piece of the net's fill that nothing carries to another layer (no via or plated hole of
                       the net stands in it), with the spots in it where a via would join it to the net's fill on
                       another layer, nearest its middle first, and the net's pads that lie in it
  {"op": "quit"}

One client at a time: a routing run's steps come one after another.
"""
import json
import socket
import sys
import time

import cupy as cp
import numpy as np
from cupyx.scipy import ndimage

from . import connect_gpu, fill_gpu, geometry

SPOTS = 60                  # the spots returned for an island: the caller tries them in order and takes the first clear
HOST = "127.0.0.1"


class Server(object):
    def __init__(self):
        self.g = self.bd = None
        self.pours = None        # (per-layer net maps, names, xs, ys) once filled

    def _filled(self):
        if self.pours is None:
            maps, names, xs, ys, _secs = fill_gpu.fills(self.g)
            self.pours = (maps, names, xs, ys)
        return self.pours

    def op_load(self, path):
        with open(path) as fh:
            self.g = json.load(fh)
        self.bd = geometry.Board(self.g)
        self.pours = None
        return {"layers": self.bd.layers, "zones": len(self.g.get("zones") or [])}

    def op_open(self):
        t0 = time.time()
        maps = self._filled()[0]
        res = connect_gpu.check(self.g, zone_fill=maps)
        return {"open": res["open"], "shorts": res["shorts"], "seconds": round(time.time() - t0, 2)}

    def op_stitch(self, net, via_mm, clearance_mm, out):
        t0 = time.time()
        maps, names, xs, ys = self._filled()
        bd, cell = self.bd, connect_gpu.CELL
        if net not in names:
            raise ValueError("no net %s on the board" % net)
        n = names.index(net)
        nx, ny = len(xs), len(ys)

        def cell_of(x, y):
            lx, ly = bd.pt(x, y)
            return (min(ny - 1, max(0, int((ly + bd.h) / cell))), min(nx - 1, max(0, int(lx / cell))))

        arrays, count = {}, cp.zeros((ny, nx), cp.uint8)
        fits = {}
        for layer, m in maps.items():
            own = m == n
            if not bool(own.any()):
                continue
            # a pad of via_mm lies whole in the fill where the nearest cell out of it is further than its radius
            fits[layer] = ndimage.distance_transform_edt(own) * cell > via_mm / 2.0
            count += fits[layer].astype(cp.uint8)
            arrays["in:" + layer] = cp.asnumpy(fits[layer].astype(cp.uint8))
        foreign = cp.zeros((ny, nx), bool)
        for layer in (bd.layers[0], bd.layers[-1]):
            other = (maps[layer] != 0) & (maps[layer] != n)
            if bool(other.any()):
                foreign |= ndimage.distance_transform_edt(~other) * cell <= via_mm / 2.0 + clearance_mm
        arrays["count"] = cp.asnumpy(count)
        arrays["foreign"] = cp.asnumpy(foreign.astype(cp.uint8))
        arrays["frame"] = np.array([bd.x0, bd.y0, bd.h, cell])
        np.savez(out, **arrays)

        # the net's own vias and plated holes, and its pads, by cell
        carriers = [cell_of(v["x"] + bd.x0, bd.y0 - v["y"]) for v in bd.vias() if v["net"] == net]
        islands = []
        for layer, m in maps.items():
            own = m == n
            if layer not in fits:
                continue
            lab, pieces = ndimage.label(own, structure=connect_gpu.FOUR)
            if not pieces:
                continue
            carried = np.zeros(pieces + 1, bool)
            lab_cpu = cp.asnumpy(lab)
            for r, c in carriers:
                carried[lab_cpu[r, c]] = True
            carried[0] = True
            sizes = np.bincount(lab_cpu.ravel(), minlength=pieces + 1)
            pads_in = {}
            for p in bd.pads:
                if p["net"] == net and layer in p["layers"]:
                    r, c = cell_of(p["x"], p["y"])
                    pads_in.setdefault(int(lab_cpu[r, c]), []).append("%s.%s" % (p["ref"], p["num"]))
            good = cp.asnumpy((count >= 2) & fits[layer])
            for k in np.nonzero(~carried)[0]:
                rows, cols = np.nonzero((lab_cpu == k) & good)
                spots = []
                if len(rows):
                    where = np.nonzero(lab_cpu == k)
                    cr, cc = where[0].mean(), where[1].mean()
                    order = np.argsort((rows - cr) ** 2 + (cols - cc) ** 2)[:SPOTS]
                    spots = [[round(float(xs[int(cols[i])]) + bd.x0, 3), round(bd.y0 - float(ys[int(rows[i])]), 3)]
                             for i in order]
                islands.append({"layer": layer, "mm2": round(float(sizes[k]) * cell * cell, 3), "spots": spots,
                                "pads": pads_in.get(int(k), [])})
        return {"islands": islands, "layers": sorted(fits), "seconds": round(time.time() - t0, 2)}


def serve():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((HOST, 0))
    srv.listen(1)
    # warm: the kernels compiled and the transforms loaded before the first question
    ndimage.distance_transform_edt(cp.ones((8, 8), bool))
    print("PORT %d" % srv.getsockname()[1], flush=True)
    state = Server()
    while True:
        conn, _addr = srv.accept()
        with conn, conn.makefile("r", encoding="utf8") as lines:
            for raw in lines:
                if not raw.strip():
                    continue
                req = json.loads(raw)
                op = req.pop("op")
                if op == "quit":
                    conn.sendall(b"{}\n")
                    return 0
                try:
                    ans = getattr(state, "op_" + op)(**req)
                except Exception as e:      # the caller falls back to KiCad's own fill: it must hear why
                    ans = {"error": "%s: %s" % (type(e).__name__, e)}
                conn.sendall((json.dumps(ans) + "\n").encode("utf8"))


if __name__ == "__main__":
    sys.exit(serve())
