"""Fast, GPU-free tests of the geometry layer on synthetic boards.
    python tests/test_geometry.py      or      pytest tests/test_geometry.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kicad_fdtd import geometry, mesh, voxel


def synth(layers, stackup=True, ref="GND", W=0.2, L=12.0, x0=100.0, y0=50.0):
    """A microstrip on the top layer over a reference zone on the next layer, plus a
    named differential pair on the top layer. KiCad frame (y down)."""
    copper = {l: [] for l in layers}
    rect = lambda xa, ya, xb, yb: [[xa, ya], [xb, ya], [xb, yb], [xa, yb]]
    copper[layers[0]].append({"net": "/T", "kind": "track", "outline": rect(x0 + 4, y0 + 4 - W / 2, x0 + 4 + L, y0 + 4 + W / 2), "holes": []})
    copper[layers[0]].append({"net": "/CLK_P", "kind": "track", "outline": rect(x0 + 4, y0 + 2 - W / 2, x0 + 4 + L, y0 + 2 + W / 2), "holes": []})
    copper[layers[0]].append({"net": "/CLK_N", "kind": "track", "outline": rect(x0 + 4, y0 + 2.5 - W / 2, x0 + 4 + L, y0 + 2.5 + W / 2), "holes": []})
    copper[layers[1]].append({"net": ref, "kind": "zone", "outline": rect(x0, y0, x0 + 20, y0 + 8), "holes": []})
    st = []
    if stackup:
        dl = [0.1, 1.24, 0.1][:len(layers) - 1] if len(layers) == 4 else [1.53] * (len(layers) - 1)
        for i, l in enumerate(layers):
            st.append({"name": l, "type": "copper", "thickness_mm": 0.035, "er": None, "tand": None})
            if i < len(layers) - 1:
                st.append({"name": "d%d" % i, "type": "prepreg", "thickness_mm": dl[i], "er": 4.5, "tand": 0.02})
    return {"layers": layers, "bbox": [x0, y0, 20.0, 8.0], "outline": rect(x0, y0, x0 + 20, y0 + 8), "stackup": st, "copper": copper,
            "vias": [{"x": x0 + 10, "y": y0 + 6, "drill": 0.3, "dia": 0.35, "top": layers[0], "bottom": layers[-1], "net": ref, "kind": "via"}],
            "pads": [{"ref": "P", "num": "1", "net": "/T", "x": x0 + 4.3, "y": y0 + 4, "layers": [layers[0]]},
                     {"ref": "P", "num": "2", "net": "/T", "x": x0 + 4 + L - 0.3, "y": y0 + 4, "layers": [layers[0]]},
                     {"ref": "J", "num": "1", "net": "/T", "x": x0 + 1, "y": y0 + 1, "layers": layers}],
            "footprints": [{"ref": "P", "value": "port", "x": x0 + 4, "y": y0 + 4, "layer": layers[0]}],
            "nets": ["/T", "/CLK_P", "/CLK_N", ref]}


def _board(g, **kw):
    p = os.path.join(tempfile.gettempdir(), "kicad_fdtd_test_%d.json" % os.getpid())
    json.dump(g, open(p, "w"))
    return geometry.Board(p, **kw)


def test_two_layer_ports_and_stats():
    bd = _board(synth(["F.Cu", "B.Cu"]))
    assert bd.ref_net == "GND"
    assert bd.stackup_source == "file" and abs(bd.z_bot + 1.53) < 1e-9
    assert bd.port_layer_options("P", "1") == [("F.Cu", "B.Cu")]
    site = bd.port_site("P", "1")
    assert site and site[2] == "F.Cu" and site[3] == "B.Cu" and site[4] == "GND"
    s = bd.stats()
    assert len(s["layers"]) == 2 and s["diff_pairs"] == [("/CLK_P", "/CLK_N")] and s["vias"] == 1
    assert s["per_layer"][1]["fill"] > 0.99
    print(geometry.format_stats(s))


def test_four_layer_default_stackup_and_other_ref():
    g = synth(["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"], stackup=False, ref="AGND")
    bd = _board(g)
    assert bd.stackup_source.startswith("default")
    assert abs(bd.z_bot - (-(1.6 - 4 * 0.035))) < 1e-9
    assert bd.ref_net == "AGND", bd.ref_net
    assert bd.port_layer_options("P", "1") == [("F.Cu", "In1.Cu"), ("F.Cu", "In2.Cu"), ("F.Cu", "B.Cu")]
    # a through-hole pad offers both sides, nearest plane first
    opts = bd.port_layer_options("J", "1")
    assert opts[0] == ("F.Cu", "In1.Cu") and ("B.Cu", "In2.Cu") in opts
    site = bd.port_site("P", "2")
    assert site and site[3] == "In1.Cu" and site[4] == "AGND"
    bd2 = _board(g, ref_net="/T")
    assert bd2.ref_net == "/T"


def test_mesh_and_merge_check():
    bd = _board(synth(["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]))
    nets = ("/T",)
    window = bd.net_bbox(nets, 2.0)
    w = geometry.PORT_W
    boxes = []
    for num in ("1", "2"):
        x, y, sig, plane, pnet, sh = bd.port_site("P", num)
        boxes.append(([x - w / 2, y - w / 2, bd.z[plane]], [x + w / 2, y + w / 2, bd.z[sig]]))
    lines = mesh.build_mesh(bd, nets, window, res=0.1, base=0.2, port_boxes=boxes, z_cell=0.1)
    vox = voxel.assign(bd, lines, window=window)
    assert set(vox["names"]) >= {"/T", "GND"}
    assert not voxel.merge_check(vox, lines, nets)
    dt, dmin = mesh.cfl_dt(lines)
    assert 1e-16 < dt < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
