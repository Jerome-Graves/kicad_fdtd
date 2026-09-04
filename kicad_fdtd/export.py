"""Export a KiCad board's copper geometry to JSON.

Two ways to run it:

    kicad-fdtd export board.kicad_pcb [-o out.json]      # from your own Python: finds KiCad's
                                                          # bundled Python and runs this file in it
    "C:/Program Files/KiCad/10.0/bin/python.exe" -m kicad_fdtd.export board.kicad_pcb out.json

The pcbnew API only exists inside KiCad's own Python, so the geometry goes
through a file and the solver runs in any Python with NumPy/CuPy.

Everything is exact polygons from KiCad's own geometry engine, no raster:
  copper[layer] = [ {net, outline:[[x,y],...], holes:[[[x,y],...],...], kind} ]
     kind = pad | track | zone   (zone fills carry their holes)
  vias = [ {x, y, drill, dia, top, bottom, net, kind} ]  (through-hole pads too, kind "pth")
  outline = [[x,y],...]           board edge
  stackup = [ {name, type, thickness_mm, er, tand} ]   from the .kicad_pcb setup (may be empty)
  pads = [ {ref, num, net, x, y, layers} ]             for port placement
  footprints = [ {ref, value, x, y, layer} ]
Coordinates in mm, KiCad's frame (y down); geometry.Board flips y.
"""
import glob
import json
import os
import re
import subprocess
import sys


def kicad_python():
    """Path of KiCad's bundled Python, or None."""
    env = os.environ.get("KICAD_PYTHON")
    if env and os.path.exists(env):
        return env
    cands = []
    if sys.platform == "win32":
        for root in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", "")):
            if root:
                cands += glob.glob(os.path.join(root, "KiCad", "*", "bin", "python.exe"))
    elif sys.platform == "darwin":
        cands += glob.glob("/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/*/bin/python3")
    else:
        for p in ("/usr/bin/python3", "/usr/local/bin/python3"):
            if os.path.exists(p):
                cands.append(p)          # distro KiCad installs pcbnew into the system Python
    def ver(p):
        m = re.search(r"KiCad[\\/](\d+)\.(\d+)", p)
        return tuple(int(v) for v in m.groups()) if m else (0, 0)
    cands.sort(key=ver, reverse=True)
    return cands[0] if cands else None


def export(pcb, out=None):
    """Run the exporter in KiCad's Python. Returns the JSON path."""
    if out is None:
        out = os.path.splitext(os.path.basename(pcb))[0] + "_geometry.json"
    try:
        import pcbnew  # noqa: F401  (already inside KiCad's Python)
        _export_here(pcb, out)
        return out
    except ImportError:
        pass
    py = kicad_python()
    if not py:
        raise RuntimeError("KiCad's Python not found; set KICAD_PYTHON to its python executable")
    r = subprocess.run([py, os.path.abspath(__file__), pcb, out], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("export failed:\n" + r.stdout + r.stderr)
    print(r.stdout.strip())
    return out


def _export_here(PCB, OUT):
    import pcbnew
    if not os.path.isfile(PCB):
        sys.exit("no such board file: %s" % PCB)
    b = pcbnew.LoadBoard(PCB)
    if b is None:
        sys.exit("pcbnew could not load %s" % PCB)
    mm = pcbnew.ToMM
    CU = [(b.GetLayerName(l), l) for l in b.GetEnabledLayers().CuStack()]
    MAXERR = pcbnew.FromMM(0.005)

    def poly_pts(o):
        return [[round(mm(o.CPoint(i).x), 4), round(mm(o.CPoint(i).y), 4)] for i in range(o.PointCount())]

    def polyset(ps, net, kind):
        out = []
        for i in range(ps.OutlineCount()):
            o = poly_pts(ps.Outline(i))
            if len(o) < 3:
                continue
            holes = [poly_pts(ps.Hole(i, h)) for h in range(ps.HoleCount(i))]
            out.append({"net": net, "kind": kind, "outline": o, "holes": [h for h in holes if len(h) >= 3]})
        return out

    copper = {name: [] for name, _ in CU}
    vias, pads, fps = [], [], []
    n_items = 0
    for fp in b.GetFootprints():
        fps.append({"ref": fp.GetReference(), "value": fp.GetValue(), "x": round(mm(fp.GetPosition().x), 4),
                    "y": round(mm(fp.GetPosition().y), 4), "layer": b.GetLayerName(fp.GetLayer())})
        for pad in fp.Pads():
            net = pad.GetNetname()
            on = [name for name, l in CU if pad.IsOnLayer(l)]
            pads.append({"ref": fp.GetReference(), "num": pad.GetNumber(), "net": net,
                         "x": round(mm(pad.GetPosition().x), 4), "y": round(mm(pad.GetPosition().y), 4), "layers": on})
            for name, l in CU:
                if not pad.IsOnLayer(l):
                    continue
                copper[name] += polyset(pad.GetEffectivePolygon(l, pcbnew.ERROR_INSIDE), net, "pad")
                n_items += 1
            d = mm(pad.GetDrillSize().x)
            if d > 0 and pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                # the barrel is the plated hole (drill + plating), not the copper ring: the ring
                # is already in the pad polygons, and a ring-sized barrel shorts to plane anti-pads
                vias.append({"x": round(mm(pad.GetPosition().x), 4), "y": round(mm(pad.GetPosition().y), 4), "drill": d,
                             "dia": round(d + 0.05, 4), "top": CU[0][0], "bottom": CU[-1][0], "net": net, "kind": "pth"})
    for t in b.GetTracks():
        cls = t.GetClass()
        if cls in ("PCB_TRACK", "PCB_ARC"):
            ps = pcbnew.SHAPE_POLY_SET()
            t.TransformShapeToPolygon(ps, t.GetLayer(), 0, MAXERR, pcbnew.ERROR_INSIDE)
            copper[b.GetLayerName(t.GetLayer())] += polyset(ps, t.GetNetname(), "track")
            n_items += 1
        elif cls == "PCB_VIA":
            vias.append({"x": round(mm(t.GetPosition().x), 4), "y": round(mm(t.GetPosition().y), 4), "drill": mm(t.GetDrillValue()),
                         "dia": mm(t.GetWidth(pcbnew.F_Cu)), "top": b.GetLayerName(t.TopLayer()), "bottom": b.GetLayerName(t.BottomLayer()),
                         "net": t.GetNetname(), "kind": "via"})
    for z in b.Zones():
        for name, l in CU:
            if z.IsOnLayer(l):
                copper[name] += polyset(z.GetFilledPolysList(l), z.GetNetname(), "zone")
                n_items += 1

    outline = pcbnew.SHAPE_POLY_SET()
    b.GetBoardPolygonOutlines(outline, True)
    edge = poly_pts(outline.Outline(0)) if outline.OutlineCount() else []

    # stackup from the file text (the SWIG stackup descriptor is not usable)
    txt = open(PCB, encoding="utf-8").read()
    m = re.search(r"\(stackup(.*?)\n\t\t\)\n", txt, re.S)
    stack = []
    if m:
        for lm in re.finditer(r'\(layer "([^"]+)"\s*\n\s*\(type "([^"]+)"\)(.*?)(?=\n\t\t\t\(layer|\Z)', m.group(1), re.S):
            nm, typ, body = lm.groups()
            th = re.search(r"\(thickness ([\d.]+)", body)
            er = re.search(r"\(epsilon_r ([\d.]+)", body)
            td = re.search(r"\(loss_tangent ([\d.]+)", body)
            stack.append({"name": nm, "type": typ, "thickness_mm": float(th.group(1)) if th else None,
                          "er": float(er.group(1)) if er else None, "tand": float(td.group(1)) if td else None})
    nets = sorted(b.FindNet(c).GetNetname() for c in range(b.GetNetCount()) if b.FindNet(c))
    bb = b.GetBoardEdgesBoundingBox()
    json.dump({"board": os.path.abspath(PCB), "kicad": pcbnew.GetBuildVersion(), "layers": [n for n, _ in CU],
               "bbox": [mm(bb.GetX()), mm(bb.GetY()), mm(bb.GetWidth()), mm(bb.GetHeight())],
               "outline": edge, "stackup": stack, "copper": copper, "vias": vias, "pads": pads, "footprints": fps, "nets": nets},
              open(OUT, "w"))
    print("%s: %d copper items -> polygons %s, %d vias/PTH, %d pads, %d footprints, stackup %d entries -> %s"
          % (os.path.basename(PCB), n_items, {k: len(v) for k, v in copper.items()}, len(vias), len(pads), len(fps), len(stack), OUT))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    export(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
