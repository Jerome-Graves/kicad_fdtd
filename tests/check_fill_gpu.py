"""fill_gpu held to KiCad's zone filler on saved boards.

    python tests/check_fill_gpu.py BOARD.kicad_pcb [BOARD.kicad_pcb ...]

For each board, two questions:

  the copper   the fills made on the card against the fills the file holds, as area: how much of KiCad's fill is not
               in ours (must stay under ONLY_FILE_PCT: boundary cells, the safe side) and how much of ours is not in
               KiCad's (thermal relief gaps, which fill_gpu leaves solid on purpose, and boundary cells: reported).
               The file's fills can be stale (copper added since KiCad last filled), so this is a guide, not the test.
  the answer   connect_gpu on the card's fills against kicad-cli's DRC run with --refill-zones, the same question
               asked of a fresh fill by each: the nets with open connections must be the same, and both must agree
               whether there is a short. This is the test.

Exit code: the boards whose answers differ.
"""
import collections
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from kicad_fdtd import connect_gpu, fill_gpu  # noqa: E402
import check_connect_gpu as base  # noqa: E402

ONLY_FILE_PCT = 1.0


def kicad_refilled(board):
    """kicad-cli's DRC of a copy of the board with the zones filled again: ({net: opens}, shorts, seconds)."""
    out = os.path.join(tempfile.gettempdir(), "fillchk_%d.json" % os.getpid())
    t0 = time.time()
    subprocess.run([os.path.join(base.KICAD_BIN, "kicad-cli.exe"), "pcb", "drc", "--refill-zones", "--format", "json",
                    "--severity-error", "--units", "mm", "-o", out, board], capture_output=True, text=True)
    secs = time.time() - t0
    with open(out, encoding="utf-8") as fh:
        rep = json.load(fh)
    os.remove(out)
    opens = collections.Counter()
    for u in rep.get("unconnected_items", []):
        d = next((i["description"] for i in u.get("items", []) if "[" in i.get("description", "")), "")
        opens[d[d.find("[") + 1:d.find("]")]] += 1
    shorts = [v["type"] for v in rep.get("violations", [])
              if v["type"] in base.SHORTS or (v["type"] == "clearance" and base.TOUCHING in v.get("description", ""))]
    return dict(opens), shorts, secs


def main(boards):
    differ = 0
    for board in boards:
        name = os.sep.join(os.path.normpath(board).split(os.sep)[-4:])
        g, t_geo = base.geometry(board)
        rows, _secs = fill_gpu.compare(g)
        tot = [sum(r[k] for r in rows) for k in ("file_mm2", "only_file_mm2", "only_here_mm2")]
        t0 = time.time()
        pours, _names, _xs, _ys, t_fill = fill_gpu.fills(g)
        res = connect_gpu.check(g, zone_fill=pours)
        t_gpu = time.time() - t0
        opens, shorts, t_drc = kicad_refilled(board)
        missing = 100.0 * tot[1] / tot[0] if tot[0] else 0.0
        same = res["open"] == opens and bool(res["shorts"]) == bool(shorts) and missing <= ONLY_FILE_PCT
        differ += not same
        print("%-5s %s" % ("ok" if same else "DIFF", name))
        print("      copper: of KiCad's %.0f mm2, %.2f %% is not in ours; ours has %.2f %% that is not in KiCad's"
              % (tot[0], missing, 100.0 * tot[2] / tot[0] if tot[0] else 0.0))
        print("      KiCad, refilled %5.1f s: %d open %s, %d short(s)" % (t_drc, sum(opens.values()), opens or "",
                                                                         len(shorts)))
        print("      GPU fill + check %4.1f s (fill %.1f; + %.1f s reading the board): %d open %s, %d short(s) %s" % (
            t_gpu, t_fill, t_geo, sum(res["open"].values()), res["open"] or "", len(res["shorts"]),
            ["%s/%s %s (%.2f, %.2f)" % (s["nets"][0], s["nets"][1], s["layer"], s["x"], s["y"])
             for s in res["shorts"][:3]] or ""), flush=True)
    print("\n%d board(s), %d differ" % (len(boards), differ))
    return differ


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
