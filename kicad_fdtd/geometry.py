"""Board geometry from the exporter's JSON (kicad_fdtd.export).

Model frame: x right, y UP (KiCad's y is flipped), z = 0 at the top copper,
origin at the board's bounding-box corner. All mm.
"""
import json
import math
import re

import numpy as np
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union

PORT_W = 0.16          # lumped-port face side; fits inside a 0.3 mm trace at 45 deg
DEFAULT_ER, DEFAULT_TAND, DEFAULT_TOTAL, DEFAULT_CU = 4.5, 0.02, 1.6, 0.035

# name patterns that mark the two halves of a differential pair
PAIR_SUFFIXES = [("_P", "_N"), ("+", "-"), ("P", "N"), ("_DP", "_DM"), ("DP", "DM"), ("_H", "_L"), ("_p", "_n")]


class Board:
    def __init__(self, path, ref_net=None):
        g = json.load(open(path))
        self.g = g
        self.path = path
        self.layers = g["layers"]
        x0, y0, w, h = g["bbox"]
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self._build_stackup(g.get("stackup") or [])
        self.outline = Polygon(self.xy(g["outline"])) if len(g.get("outline") or []) >= 3 else box(0, -h, w, 0)
        self.pads = g["pads"]
        self._poly_cache = {}
        self.ref_net = ref_net or self.guess_ref_net()

    # ---------------------------------------------------------------- stackup
    def _build_stackup(self, stack):
        """z of every copper layer and the dielectric list (z_top, z_bottom, er, tand).
        Falls back to a symmetric FR4 stack when the file has no usable stackup."""
        cu = [s for s in stack if s["type"] == "copper"]
        dl = [s for s in stack if s["type"] in ("prepreg", "core")]
        n = len(self.layers)
        usable = len(cu) == n and len(dl) == n - 1 and all(d.get("thickness_mm") for d in dl)
        if usable:
            self.stackup_source = "file"
            cu_t = [c.get("thickness_mm") or DEFAULT_CU for c in cu]
            dl_t = [(d["thickness_mm"], d.get("er") or DEFAULT_ER, d.get("tand") or DEFAULT_TAND) for d in dl]
        else:
            self.stackup_source = "default (no stackup in the file: %.1f mm FR4, er %.1f)" % (DEFAULT_TOTAL, DEFAULT_ER)
            cu_t = [DEFAULT_CU] * n
            t = (DEFAULT_TOTAL - n * DEFAULT_CU) / max(1, n - 1)
            dl_t = [(t, DEFAULT_ER, DEFAULT_TAND)] * (n - 1)
        self.cu_thickness = dict(zip(self.layers, cu_t))
        z, self.z, self.diel = 0.0, {}, []
        for i, layer in enumerate(self.layers):
            self.z[layer] = z
            if i < n - 1:
                t, er, tand = dl_t[i]
                self.diel.append((z, z - t, er, tand))
                z -= t
        self.z_bot = z

    # ---------------------------------------------------------------- frames
    def xy(self, pts):
        return [(p[0] - self.x0, -(p[1] - self.y0)) for p in pts]

    def pt(self, x, y):
        return (x - self.x0, -(y - self.y0))

    def pad(self, ref, num):
        p = next((p for p in self.pads if p["ref"] == ref and p["num"] == str(num)), None)
        if p is None:
            raise KeyError("no pad %s.%s" % (ref, num))
        return p, self.pt(p["x"], p["y"])

    def polys(self, layer):
        """[(net, kind, shapely Polygon)] for a layer, cached."""
        if layer not in self._poly_cache:
            out = []
            for poly in self.g["copper"].get(layer, []):
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
            out.append({"x": x, "y": y, "r": v["dia"] / 2, "zt": self.z.get(v["top"], 0.0), "zb": self.z.get(v["bottom"], self.z_bot),
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
        if not xs:
            raise ValueError("no copper on nets %s" % (nets,))
        return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)

    # ---------------------------------------------------------------- reference net, stats
    def zone_area_by_net(self):
        area = {}
        for layer in self.layers:
            for net, kind, P in self.polys(layer):
                if kind == "zone" and net:
                    area[net] = area.get(net, 0.0) + P.area
        return area

    def guess_ref_net(self):
        """The net with the most zone copper (GND on nearly every board)."""
        area = self.zone_area_by_net()
        if not area:
            return "GND" if "GND" in self.g.get("nets", []) else None
        return max(area, key=area.get)

    def diff_pairs(self):
        """[(P, N)] pairs of net names that look like a differential pair."""
        nets = set(self.g.get("nets", []))
        out = []
        for n in sorted(nets):
            for sp, sn in PAIR_SUFFIXES:
                if n.endswith(sp) and n[:-len(sp)] + sn in nets and len(n) > len(sp):
                    out.append((n, n[:-len(sp)] + sn))
        return sorted(set(out))

    def stats(self):
        """Board statistics for the CLI and the GUI."""
        g = self.g
        per_layer = []
        board_area = self.outline.area
        for layer in self.layers:
            ps = self.polys(layer)
            cu = unary_union([P for _, _, P in ps]) if ps else None
            kinds = {}
            for _, kind, _ in ps:
                kinds[kind] = kinds.get(kind, 0) + 1
            per_layer.append({"layer": layer, "z_mm": self.z[layer], "polygons": len(ps), "kinds": kinds,
                              "copper_mm2": cu.area if cu else 0.0, "fill": (cu.area / board_area) if (cu and board_area) else 0.0})
        vias = self.vias()
        area = self.zone_area_by_net()
        ref_cands = sorted(area.items(), key=lambda kv: -kv[1])[:5]
        net_polys = {}
        for layer in self.layers:
            for net, kind, P in self.polys(layer):
                if net:
                    d = net_polys.setdefault(net, {"tracks": 0, "pads": 0, "zones": 0, "layers": set()})
                    d[{"track": "tracks", "pad": "pads", "zone": "zones"}[kind]] += 1
                    d["layers"].add(layer)
        for v in vias:
            if v["net"]:
                net_polys.setdefault(v["net"], {"tracks": 0, "pads": 0, "zones": 0, "layers": set()}).setdefault("vias", 0)
                net_polys[v["net"]]["vias"] = net_polys[v["net"]].get("vias", 0) + 1
        nets = [{"net": n, **{k: (sorted(v) if isinstance(v, set) else v) for k, v in d.items()}} for n, d in sorted(net_polys.items())]
        return {
            "board": g.get("board"), "kicad": g.get("kicad"),
            "size_mm": [self.w, self.h], "board_area_mm2": board_area, "outline_points": len(g.get("outline") or []),
            "layers": self.layers, "thickness_mm": -self.z_bot + sum(self.cu_thickness.values()),
            "stackup_source": self.stackup_source,
            "dielectrics": [{"from": zt, "to": zb, "thickness_mm": zt - zb, "er": er, "tand": tand} for zt, zb, er, tand in self.diel],
            "per_layer": per_layer,
            "footprints": len(g.get("footprints") or {p["ref"] for p in self.pads}),
            "pads": len(self.pads), "pth": sum(1 for v in vias if v["kind"] == "pth"), "vias": sum(1 for v in vias if v["kind"] == "via"),
            "nets": len(g.get("nets") or net_polys), "ref_net": self.ref_net,
            "ref_candidates": [{"net": n, "zone_mm2": a} for n, a in ref_cands],
            "diff_pairs": self.diff_pairs(),
            "net_detail": nets,
        }

    # ---------------------------------------------------------------- ports
    def port_layer_options(self, ref, num):
        """(signal layer, reference layer) pairs to try for a pad, nearest reference layer first.
        Works for 2 to N copper layers: the reference is any other copper layer, searched
        away from the pad's side of the board."""
        p, _ = self.pad(ref, num)
        idx = {l: i for i, l in enumerate(self.layers)}
        n = len(self.layers)
        on = [l for l in p["layers"] if l in idx]
        if not on:
            return []
        sides = []
        if self.layers[0] in on:
            sides.append((self.layers[0], list(range(1, n))))                # F.Cu pad: planes below
        if self.layers[-1] in on and n > 1:
            sides.append((self.layers[-1], list(range(n - 2, -1, -1))))      # B.Cu pad: planes above
        for l in on:                                                          # inner pads (rare)
            i = idx[l]
            if 0 < i < n - 1:
                order = sorted(range(n), key=lambda j: (abs(j - i), j))
                sides.append((l, [j for j in order if j != i]))
        out = []
        for sig, plane_idx in sides:
            for j in plane_idx:
                out.append((sig, self.layers[j]))
        return out

    def port_site(self, ref, num, tie_nets=None, max_shift=3.5, w=PORT_W, ref_net=None):
        """Where a lumped port for this pad can go: (x, y, signal layer, plane
        layer, reference net, shift). The WHOLE w x w face must be on the pad's net
        on the signal layer and on the reference net on the plane layer; walk along
        the pad's track, then search a ring, until that holds."""
        tie = tie_nets or {}
        pnet = ref_net or self.ref_net
        p, (x0, y0) = self.pad(ref, num)
        net = p["net"]
        for sig, plane in self.port_layer_options(ref, num):
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


def format_stats(s):
    """Plain-text report."""
    L = ["%s  (KiCad %s)" % (s["board"], s["kicad"] or "?"),
         "size %.1f x %.1f mm, outline area %.0f mm2, %d copper layers, thickness %.2f mm" % (s["size_mm"][0], s["size_mm"][1], s["board_area_mm2"], len(s["layers"]), s["thickness_mm"]),
         "stackup: %s" % s["stackup_source"]]
    for d in s["dielectrics"]:
        L.append("   dielectric %.3f mm  er %.2f  tand %.3f" % (d["thickness_mm"], d["er"], d["tand"]))
    L.append("%-8s %8s %9s %6s  %s" % ("layer", "z mm", "copper", "fill", "polygons"))
    for p in s["per_layer"]:
        L.append("%-8s %8.3f %7.0f mm2 %5.0f%%  %s" % (p["layer"], p["z_mm"], p["copper_mm2"], 100 * p["fill"], ", ".join("%d %s" % (v, k) for k, v in p["kinds"].items())))
    L.append("%d footprints, %d pads (%d plated holes), %d vias, %d nets" % (s["footprints"], s["pads"], s["pth"], s["vias"], s["nets"]))
    L.append("reference net: %s   (candidates by zone area: %s)" % (s["ref_net"], ", ".join("%s %.0f mm2" % (c["net"], c["zone_mm2"]) for c in s["ref_candidates"])))
    if s["diff_pairs"]:
        L.append("differential pairs by name: " + ", ".join("%s/%s" % pr for pr in s["diff_pairs"]))
    return "\n".join(L)
