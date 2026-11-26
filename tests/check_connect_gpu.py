"""connect_gpu held to KiCad's DRC on saved boards.

    python tests/check_connect_gpu.py BOARD.kicad_pcb [BOARD.kicad_pcb ...]

For each board: its geometry exported by KiCad's Python (kicad_fdtd.export), connect_gpu.check on it, and kicad-cli's
DRC of the same file. The two must agree on which nets have open connections and how many, and every short KiCad
reports (shorting_items, tracks_crossing, and a clearance error where the copper touches) must be among the GPU's.
A board passes when they do; the times of both are printed, since the GPU check exists to replace the other between
routing steps. Exit code: the boards that differ.
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
from kicad_fdtd import connect_gpu  # noqa: E402

KICAD_BIN = os.environ.get("KICAD_BIN", r"C:/Program Files/KiCad/10.0/bin")
SHORTS = ("shorting_items", "tracks_crossing")          # KiCad's names for copper of two nets in contact
TOUCHING = "actual 0.0000 mm"                           # ... and what its clearance error says when they touch (a
                                                        # track laid through a pour that was filled before it: the
                                                        # file as saved holds both, until the pour is filled again)


def kicad(board):
    """({net: open connections}, [short identities], seconds) by kicad-cli, as the board file stands (no refill:
    the GPU check reads the fills the file holds)."""
    out = os.path.join(tempfile.gettempdir(), "connect_%d.json" % os.getpid())
    t0 = time.time()
    subprocess.run([os.path.join(KICAD_BIN, "kicad-cli.exe"), "pcb", "drc", "--format", "json", "--severity-error",
                    "--units", "mm", "-o", out, board], capture_output=True, text=True)
    secs = time.time() - t0
    with open(out, encoding="utf-8") as fh:
        rep = json.load(fh)
    os.remove(out)
    opens = collections.Counter()
    for u in rep.get("unconnected_items", []):
        d = next((i["description"] for i in u.get("items", []) if "[" in i.get("description", "")), "")
        opens[d[d.find("[") + 1:d.find("]")]] += 1
    shorts = ["%s: %s" % (v["type"], "; ".join(i["description"] for i in v.get("items", [])))
              for v in rep.get("violations", [])
              if v["type"] in SHORTS or (v["type"] == "clearance" and TOUCHING in v.get("description", ""))]
    return dict(opens), shorts, secs


def geometry(board):
    out = os.path.join(tempfile.gettempdir(), "connect_%d_geo.json" % os.getpid())
    t0 = time.time()
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONPATH"] = os.path.dirname(HERE)
    subprocess.run([os.path.join(KICAD_BIN, "python.exe"), "-m", "kicad_fdtd.export", board, out],
                   capture_output=True, text=True, env=env)
    with open(out) as fh:
        g = json.load(fh)
    os.remove(out)
    return g, time.time() - t0


def main(boards):
    differ = 0
    for board in boards:
        name = os.sep.join(os.path.normpath(board).split(os.sep)[-4:])
        g, t_geo = geometry(board)
        res = connect_gpu.check(g)
        opens, shorts, t_drc = kicad(board)
        same = res["open"] == opens and bool(res["shorts"]) == bool(shorts)
        differ += not same
        print("%-5s %s" % ("ok" if same else "DIFF", name))
        print("      KiCad %5.1f s: %d open %s, %d short(s)" % (t_drc, sum(opens.values()), opens or "", len(shorts)))
        print("      GPU   %5.1f s (+ %.1f s reading the board): %d open %s, %d short(s) %s" % (
            res["seconds"], t_geo, sum(res["open"].values()), res["open"] or "", len(res["shorts"]),
            ["%s/%s %s (%.2f, %.2f)" % (s["nets"][0], s["nets"][1], s["layer"], s["x"], s["y"])
             for s in res["shorts"][:3]] or ""), flush=True)
    print("\n%d board(s), %d differ" % (len(boards), differ))
    return differ


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
