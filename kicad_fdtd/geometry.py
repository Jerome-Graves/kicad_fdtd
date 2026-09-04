"""Board geometry from the KiCad exporter's JSON (same file as
bugbot_sim/tools/openems/out/<board>_geometry.json).

Model frame: x right, y UP (KiCad's y is flipped), z = 0 at the top copper,
origin at the board's bounding-box corner. All mm.
"""
import json
import math

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, box
from shapely.ops import unary_union

PORT_W = 0.16          # lumped-port face side; fits inside a 0.3 mm trace at 45 deg


class Board:
    def __init__(self, path):
        g = json.load(open(path))
        self.g = g
        self.layers = g["layers"]
        x0, y0, w, h = g["bbox"]
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        z, self.z, self.diel = 0.0, {}, []          # diel: (z_top, z_bottom, er, tand)
        cu = [s for s in g["stackup"] if s["type"] == "copper"]
        dl = [s for s in g["stackup"] if s["type"] in ("prepreg", "core")]
        assert len(cu) == len(self.layers) and len(dl) == len(cu) - 1
        for i, layer in enumerate(self.layers):
            self.z[layer] = z
            if i < len(dl):
                t = dl[i]["thickness_mm"]
                self.diel.append((z, z - t, dl[i]["er"] or 4.5, dl[i]["tand"] or 0.02))
                z -= t
        self.z_bot = z
        self.outline = Polygon(self.xy(g["outline"]))
        self.pads = g["pads"]
        self._poly_cache = {}

    def xy(self, pts):
        return [(p[0] - self.x0, -(p[1] - self.y0)) for p in pts]

    def pt(self, x, y):
        return (x - self.x0, -(y - self.y0))

    def pad(self, ref, num):
        p = next(p for p in self.pads if p["ref"] == ref and p["num"] == str(num))
        return p, self.pt(p["x"], p["y"])

    def polys(self, layer):
        """[(net, kind, shapely Polygon)] for a layer, cached."""
        if layer not in self._poly_cache:
            out = []
            for poly in self.g["copper"][layer]:
                P = Polygon(self.xy(poly["outline"]), [self.xy(h) for h in poly["holes"]])
                if not P.is_valid:
                    P = P.buffer(0)
                if not P.is_empty:
                    out.append((poly["net"], poly["kind"], P))
            self._poly_cache[layer] = out
        return self._poly_cache[layer]

    def vias(self):
        out = []
        for v in self.g["vias"]:
            x, y = self.pt(v["x"], v["y"])
            out.append({"x": x, "y": y, "r": v["dia"] / 2, "zt": self.z[v["top"]], "zb": self.z[v["bottom"]],
                        "net": v["net"], "kind": v["kind"]})
        return out

    def copper_at(self, layer, x, y):
        pt = Point(x, y)
        return {net for net, kind, P in self.polys(layer) if P.contains(pt)}

    def net_bbox(self, nets, margin=2.0):
        xs, ys = [], []
        for layer in self.layers:
            for net, kind, P in self.polys(layer):
                if net in nets:
                    b = P.bounds
                    xs += [b[0], b[2]]; ys += [b[1], b[3]]
        for v in self.vias():
            if v["net"] in nets:
                xs.append(v["x"]); ys.append(v["y"])
        return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)

    # ---- ports
    def port_layer_options(self, ref, num):
        p, _ = self.pad(ref, num)
        top, bot = ("F.Cu", self.layers[1], "GND"), ("B.Cu", self.layers[-2], "GND")
        inner = (self.layers[2], self.layers[1], "GND")
        if p["layers"] == ["B.Cu"]:
            return [bot]
        if p["layers"] == ["F.Cu"]:
            return [top]
        return [top, bot, inner]

    def port_site(self, ref, num, tie_nets=None, max_shift=3.5, w=PORT_W):
        """Where a lumped port for this pad can go: (x, y, signal layer, plane
        layer, shift). The WHOLE w x w face must be on the pad's net on the
        signal layer and on the reference net on the plane layer; walk along
        the pad's track, then search a ring, until that holds."""
        tie = tie_nets or {}
        p, (x0, y0) = self.pad(ref, num)
        net = p["net"]
        for sig, plane, pnet in self.port_layer_options(ref, num):
            def ok(x, y):
                for dx, dy in ((0, 0), (-w / 2, -w / 2), (w / 2, -w / 2), (-w / 2, w / 2), (w / 2, w / 2)):
                    if net not in self.copper_at(sig, x + dx, y + dy):
                        return False
                    if {tie.get(n, n) for n in self.copper_at(plane, x + dx, y + dy)} != {pnet}:
                        return False
                return True
            tracks = [P for n, kind, P in self.polys(sig) if kind == "track" and n == net]
            cands = sorted((math.hypot(P.centroid.x - x0, P.centroid.y - y0), P.centroid.x, P.centroid.y) for P in tracks)
            dirs = [((cx - x0) / d, (cy - y0) / d) for d, cx, cy in cands[:3] if d > 1e-6]
            for shift in np.arange(0.0, max_shift + 0.05, 0.1):
                for dx, dy in (dirs or [(0.0, 0.0)]):
                    x, y = x0 + dx * shift, y0 + dy * shift
                    if ok(x, y):
                        return x, y, sig, plane, pnet, float(shift)
            for shift in np.arange(0.2, max_shift + 0.05, 0.1):
                for ang in range(0, 360, 20):
                    x = x0 + shift * math.cos(math.radians(ang)); y = y0 + shift * math.sin(math.radians(ang))
                    if ok(x, y):
                        return x, y, sig, plane, pnet, float(shift)
        return None
