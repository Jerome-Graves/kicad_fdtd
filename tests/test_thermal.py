"""Analytic checks for the thermal and DC solvers (no GPU needed).
    python tests/test_thermal.py      or      pytest tests/test_thermal.py

1. A copper-clad plate with P watts and convection h on both faces settles at
   T_amb + P / (2 h A) when the copper spreads the heat: mean rise within 3 %, spread small.
2. A 0.2 mm x 35 um track carrying 1 A over 11.4 mm drops I * rho L / (w t) = 27.4 mV.
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kicad_fdtd import geometry, thermal
from test_geometry import synth


def _board(g):
    p = os.path.join(tempfile.gettempdir(), "kicad_fdtd_thermal_%d.json" % os.getpid())
    json.dump(g, open(p, "w"))
    return geometry.Board(p)


def test_plate_convection():
    g = synth(["F.Cu", "B.Cu"])
    x0, y0 = 100.0, 50.0
    full = [[x0, y0], [x0 + 20, y0], [x0 + 20, y0 + 8], [x0, y0 + 8]]
    g["copper"]["F.Cu"] = [{"net": "GND", "kind": "zone", "outline": full, "holes": []}]
    g["copper"]["B.Cu"] = [{"net": "GND", "kind": "zone", "outline": full, "holes": []}]
    bd = _board(g)
    gr = thermal.Grid(bd, cell=0.5)
    P, h = 1.0, 10.0
    th = thermal.solve_thermal(gr, {"P": P}, h=h, tamb=25.0, use_gpu=False)
    A = 20e-3 * 8e-3
    expect = P / (2 * h * A)
    T = th["T"]
    rise = T[:, :, gr.k_cu["F.Cu"]].mean() - 25.0
    print("mean rise %.1f K, expected %.1f K, max %.1f, min %.1f" % (rise, expect, T.max() - 25, T.min() - 25))
    assert abs(rise - expect) / expect < 0.03, (rise, expect)
    assert (T.max() - T.min()) < 0.15 * expect


def test_track_ir_drop():
    g = synth(["F.Cu", "B.Cu"])
    bd = _board(g)
    gr = thermal.Grid(bd, cell=0.1)
    dc = thermal.solve_dc(gr, "/T", ("P", "1"), ("P", "2"), 1.0, use_gpu=False)
    L, w, t = 11.4e-3, 0.2e-3, 35e-6
    expect = 1.0 * thermal.RHO_CU * L / (w * t)
    print("drop %.2f mV, expected %.2f mV, Jmax %.0f A/mm2 (expected %.0f)" % (1e3 * dc["drop_V"], 1e3 * expect, dc["Jmax"], 1.0 / (w * t) / 1e6))
    assert abs(dc["drop_V"] - expect) / expect < 0.1, (dc["drop_V"], expect)
    assert abs(dc["Jmax"] - 1.0 / (w * t) / 1e6) / (1.0 / (w * t) / 1e6) < 0.25
    # Joule heat must equal I^2 R
    assert abs(dc["joule_W"] - 1.0 * dc["drop_V"]) / (1.0 * dc["drop_V"]) < 0.02


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
