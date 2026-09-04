"""Milestone test: a 12 mm x 0.2 mm microstrip over a 0.1 mm FR4 (er 4.5)
plane, 50-ohm lumped ports at both ends. Expect S21 ~ 0 dB, S11 < -15 dB,
Zin ~ 52 ohm (the 2D cross-section solver gives 45-52 depending on pour).
    venv\\Scripts\\python.exe tests\\test_msl.py [--cpu]
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kicad_fdtd import geometry, mesh, voxel, solver

L, W, H = 12.0, 0.2, 0.1
x0, y0 = 100.0, 50.0                       # any KiCad origin
synth = {
    "layers": ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"],
    "bbox": [x0, y0, 20.0, 8.0],
    "outline": [[x0, y0], [x0 + 20, y0], [x0 + 20, y0 + 8], [x0, y0 + 8]],
    "stackup": [{"name": "F.Cu", "type": "copper", "thickness_mm": 0.035},
                {"name": "d1", "type": "prepreg", "thickness_mm": H, "er": 4.5, "tand": 0.02},
                {"name": "In1.Cu", "type": "copper", "thickness_mm": 0.035},
                {"name": "d2", "type": "core", "thickness_mm": 1.24, "er": 4.5, "tand": 0.02},
                {"name": "In2.Cu", "type": "copper", "thickness_mm": 0.035},
                {"name": "d3", "type": "prepreg", "thickness_mm": 0.1, "er": 4.5, "tand": 0.02},
                {"name": "B.Cu", "type": "copper", "thickness_mm": 0.035}],
    "copper": {"F.Cu": [{"net": "/T", "kind": "track", "outline": [[x0 + 4, y0 + 4 - W / 2], [x0 + 4 + L, y0 + 4 - W / 2], [x0 + 4 + L, y0 + 4 + W / 2], [x0 + 4, y0 + 4 + W / 2]], "holes": []}],
               "In1.Cu": [{"net": "GND", "kind": "zone", "outline": [[x0, y0], [x0 + 20, y0], [x0 + 20, y0 + 8], [x0, y0 + 8]], "holes": []}],
               "In2.Cu": [], "B.Cu": []},
    "vias": [],
    "pads": [{"ref": "P", "num": "1", "net": "/T", "x": x0 + 4.3, "y": y0 + 4, "layers": ["F.Cu"]},
             {"ref": "P", "num": "2", "net": "/T", "x": x0 + 4 + L - 0.3, "y": y0 + 4, "layers": ["F.Cu"]}],
    "nets": ["/T", "GND"],
}
os.makedirs("out", exist_ok=True)
p = os.path.join("out", "msl_geometry.json")
json.dump(synth, open(p, "w"))
bd = geometry.Board(p)
nets = ("/T",)
window = (0.0, -8.0, 20.0, 0.0)
use_gpu = "--cpu" not in sys.argv
res = 0.1

ports_xy = []
for ref, num in (("P", "1"), ("P", "2")):
    site = bd.port_site(ref, num)
    assert site, "no port site for %s.%s" % (ref, num)
    ports_xy.append(site)
w = geometry.PORT_W
boxes = [([x - w / 2, y - w / 2, bd.z[plane]], [x + w / 2, y + w / 2, bd.z[sig]]) for x, y, sig, plane, pnet, sh in ports_xy]
lines = mesh.build_mesh(bd, nets, window, res=res, base=0.2, port_boxes=boxes, z_cell=0.1)
vox = voxel.assign(bd, lines, window=window)
shorts = voxel.merge_check(vox, lines, nets)
print("mesh %d x %d x %d = %.2f Mcells, conductors %s, merge check: %d" % (len(lines["x"]), len(lines["y"]), len(lines["z"]),
      np.prod([len(lines[k]) for k in "xyz"]) / 1e6, vox["names"], len(shorts)))
assert not shorts, shorts[:5]


def er_of_z(z):
    for zt, zb, er, tand in bd.diel:
        if zb <= z <= zt:
            return er
    return 1.0


sim = solver.FDTD(lines, vox, er_of_z, npml=8, use_gpu=use_gpu)
print("engine:", "GPU" if sim.xp is not np else "CPU", " dt %.1f fs" % (sim.dt * 1e15))
f0, fc = 1.5e9, 1.5e9
exc, t0, tau = solver.gauss_pulse(f0, fc)
for i, (s, e) in enumerate(boxes):
    sim.add_port(s[0], e[0], s[1], e[1], s[2], e[2], 50.0, excite=1.0 if i == 0 else 0.0)
steps = int(12e-9 / sim.dt)
print("running up to %d steps (12 ns) ..." % steps)
sim.run(exc, steps, end_criteria=1e-3, log=print, log_every=5000)
print("done: %d steps in %.0f s = %.0f Mcell-updates/s" % (sim.nsteps_done, sim.wall, sim.nsteps_done * np.prod([len(lines[k]) for k in "xyz"]) / sim.wall / 1e6))
f = np.linspace(200e6, 3e9, 100)
S = sim.port_spectra(f)
s11 = S[0]["ref"] / S[0]["inc"]
s21 = S[1]["ref"] / S[0]["inc"]
Z = S[0]["Z"]
i1 = np.argmin(abs(f - 1e9)); sel = (f > 300e6) & (f < 2e9)
print("RESULT: S21 %.2f dB  S11 %.1f dB at 1 GHz | Zin median %.0f ohm (300 MHz-2 GHz) | expected S21 ~0, S11 < -15, Z ~52"
      % (20 * np.log10(abs(s21[i1]) + 1e-12), 20 * np.log10(abs(s11[i1]) + 1e-12), np.median(Z.real[sel])))
ok = abs(20 * np.log10(abs(s21[i1]) + 1e-12)) < 1.0 and 20 * np.log10(abs(s11[i1]) + 1e-12) < -12 and 35 < np.median(Z.real[sel]) < 70
print("PASS" if ok else "FAIL")
