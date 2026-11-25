"""voxel_gpu.assign against voxel.assign: the same conductor on every mesh edge.

    python tests/check_voxel_gpu.py                               the synthetic board of test_geometry (seconds)
    python tests/check_voxel_gpu.py GEOMETRY.json NET_P NET_N     a pair on a real board (the CPU side is slow:
                                                                  minutes on a board with planes), ports at the
                                                                  pair's ends (pair.end_ports)

Edges are compared by conductor name (the two number their conductors in different orders). Exits 1 on any
difference. MP-BOY, DSI_C on the 8-layer _kl13 board (7.45 Mcells, 2026-11-25): GPU 0.23 s.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kicad_fdtd import geometry, mesh, pair, voxel, voxel_gpu  # noqa: E402


def named(vox, key):
    names = np.array(vox["names"] + [None], dtype=object)
    return names[vox[key]]                       # -1 indexes the None at the end


def compare(bd, nets, window, boxes, skip=()):
    lines = mesh.build_mesh(bd, nets, window, port_boxes=boxes)
    cells = int(np.prod([len(lines[k]) for k in "xyz"]))
    t = time.time()
    g = voxel_gpu.assign(bd, lines, skip_via_near=skip, window=window)
    tg = time.time() - t
    t = time.time()
    c = voxel.assign(bd, lines, skip_via_near=skip, window=window)
    tc = time.time() - t
    diff = 0
    for key in ("ex", "ey", "ez"):
        d = named(c, key) != named(g, key)
        diff += int(d.sum())
        if d.any():
            i, j, k = (int(a[0]) for a in np.nonzero(d))
            print("%s differs at %d edge(s), first (%d, %d, %d): CPU %s, GPU %s"
                  % (key, int(d.sum()), i, j, k, named(c, key)[i, j, k], named(g, key)[i, j, k]))
    sc = sorted(voxel.merge_check(c, lines, nets))
    sg = sorted(voxel_gpu.merge_check(g, lines, nets))
    print("%.2f Mcells: CPU %.1f s, GPU %.2f s; %d edge(s) differ; merge check %d / %d node(s)"
          % (cells / 1e6, tc, tg, diff, len(sc), len(sg)))
    return diff + (len(sc) != len(sg))


def main(argv):
    if argv:
        bd = geometry.Board(argv[0])
        p, n = argv[1], argv[2]
        sites = [bd.port_site(*rp.split(".")) for rp in pair.end_ports(bd, p, n)]
        nets, skip = [p, n], [(s[0], s[1]) for s in sites]
    else:
        import tempfile
        import json
        from test_geometry import synth
        path = os.path.join(tempfile.mkdtemp(prefix="fdtd_vox_"), "g.json")
        json.dump(synth(["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]), open(path, "w"))
        bd = geometry.Board(path)
        nets, skip = ["/T"], []
        sites = [bd.port_site("P", num) for num in ("1", "2")]
    w = geometry.PORT_W
    boxes = []
    for x, y, sig, plane, pnet, sh in sites:
        z1, z2 = sorted((bd.z[plane], bd.z[sig]))
        boxes.append(([x - w / 2, y - w / 2, z1], [x + w / 2, y + w / 2, z2]))
    return 1 if compare(bd, nets, bd.net_bbox(nets, 2.0 if not argv else 2.5), boxes, skip) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
