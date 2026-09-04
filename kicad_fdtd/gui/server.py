"""Browser GUI: import a KiCad board, see its stats, pick nets and ports, mesh and check,
run on the GPU, watch the field, read S-parameters. Standard library only.

    kicad-fdtd gui [--port 8765] [--boards DIR ...]

Endpoints (JSON):
  GET  /api/boards                       geometry files found in the board dirs
  POST /api/export      {pcb}            run the exporter on a .kicad_pcb (KiCad's Python), returns the JSON name
  GET  /api/board?file=NAME[&ref=NET]    layers, outline, copper polygons, vias, pads, nets, stats
  POST /api/setup       {file, nets, ports, res, base, fmax, margin, tie, z0, ref}
  GET  /api/setup                        the last setup (for reloads)
  POST /api/run         {end, tmax, snap_every}
  POST /api/stop
  GET  /api/status
  GET  /api/snap?i=N                     Ez slice N (float16, base64) on the snapshot plane
  GET  /api/result
"""
import argparse
import base64
import glob
import json
import os
import sys
import threading
import time
import traceback
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from kicad_fdtd import geometry, pair, export as kexport

STATIC = os.path.join(HERE, "static")
BOARD_DIRS = ["out"]
STATE = {"board": None, "file": None, "ref": None, "setup": None, "sim": None, "running": False, "stop": False,
         "snaps": [], "snap_z": None, "result": None, "error": None, "log": [], "nsteps": None}
LOCK = threading.Lock()


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def find_file(name):
    for d in BOARD_DIRS:
        for f in glob.glob(os.path.join(d, "*_geometry.json")):
            if os.path.basename(f) == name:
                return f
    raise FileNotFoundError(name)


def load_board(name, ref=None):
    path = find_file(name)
    if STATE["file"] != path or STATE["ref"] != ref:
        STATE["board"], STATE["file"], STATE["ref"] = geometry.Board(path, ref_net=ref or None), path, ref
        STATE["setup"], STATE["result"], STATE["snaps"] = None, None, []
    return STATE["board"]


def board_json(bd):
    layers = []
    for layer in bd.layers:
        polys = []
        for net, kind, P in bd.polys(layer):
            for G in (P.geoms if hasattr(P, "geoms") else [P]):
                G = G.simplify(0.01)
                if G.is_empty:
                    continue
                polys.append({"net": net, "kind": kind,
                              "ext": [[round(x, 3), round(y, 3)] for x, y in G.exterior.coords],
                              "holes": [[[round(x, 3), round(y, 3)] for x, y in h.coords] for h in G.interiors]})
        layers.append({"name": layer, "z": bd.z[layer], "polys": polys})
    s = bd.stats()
    return {"w": bd.w, "h": bd.h, "z_bot": bd.z_bot, "diel": bd.diel, "ref_net": bd.ref_net, "stackup_source": bd.stackup_source,
            "outline": [[round(x, 3), round(y, 3)] for x, y in bd.outline.exterior.coords],
            "layers": layers, "vias": bd.vias(),
            "pads": [{"ref": p["ref"], "num": p["num"], "net": p["net"], "layers": p["layers"], **dict(zip(("x", "y"), bd.pt(p["x"], p["y"])))} for p in bd.pads],
            "nets": bd.g.get("nets") or sorted({p["net"] for p in bd.pads if p["net"]}),
            "stats": {k: v for k, v in s.items() if k != "net_detail"}, "stats_text": geometry.format_stats(s)}


def do_setup(req):
    bd = load_board(req["file"], req.get("ref") or None)
    log = []
    st = pair.setup(bd, list(req["nets"]), list(req["ports"]), res=float(req.get("res", 0.1)), base=float(req.get("base", 0.2)),
                    margin=float(req.get("margin", 2.5)), tie=req.get("tie", []), log=log.append)
    z = st["lines"]["z"]
    assign = []
    for layer in bd.layers:
        k = int(np.argmin(abs(z - bd.z[layer])))
        assign.append({"layer": layer, "k": k, "ex": b64(st["vox"]["ex"][:, :, k].astype(np.int8)), "ey": b64(st["vox"]["ey"][:, :, k].astype(np.int8))})
    out = {"file": req["file"], "ref": bd.ref_net, "nets": st["nets"], "ports": st["ports"], "tie": list(st["tie"]), "sites": st["sites"],
           "window": st["window"], "lines": {k: st["lines"][k].tolist() for k in "xyz"}, "cells": st["cells"], "dt": st["dt"],
           "dmin": [float(v) for v in st["dmin"]], "shorts": [list(s) for s in st["shorts"][:50]], "n_shorts": len(st["shorts"]),
           "n_via": st["vox"]["n_via"], "names": st["vox"]["names"], "assign": assign, "log": log,
           "fmax": float(req.get("fmax", 3e9)), "z0": float(req.get("z0", 50.0))}
    STATE["setup"], STATE["_st"] = out, st
    STATE["snaps"], STATE["result"] = [], None
    return out


def run_thread(req):
    try:
        bd, setup, st = STATE["board"], STATE["setup"], STATE["_st"]
        snap_every = int(req.get("snap_every", 500))
        z = st["lines"]["z"]
        zmid = (setup["sites"][0]["z0"] + setup["sites"][0]["z1"]) / 2
        kz = int(np.argmin(abs((z[:-1] + z[1:]) / 2 - zmid)))
        STATE["snap_z"] = float((z[kz] + z[kz + 1]) / 2)
        tmax = float(req.get("tmax", 12e-9))

        def hook(n, sim):
            if STATE["sim"] is None:
                STATE["sim"] = sim
                STATE["nsteps"] = int(tmax / sim.dt)
            if STATE["stop"]:
                return False
            if n % snap_every == 0:
                ez = sim.Ez[:, :, kz]
                STATE["snaps"].append({"n": n, "t": n * sim.dt, "data": (ez.get() if sim.xp is not np else ez).astype(np.float16)})
            return True
        r, sim = pair.solve(bd, st, fmax=setup["fmax"], z0=setup["z0"], tmax=tmax, end=float(req.get("end", 1e-3)),
                            log=lambda s: STATE["log"].append(s), hook=hook, hook_every=100)
        STATE["sim"] = sim
        del r["port_VI"]
        STATE["result"] = r
    except Exception:
        STATE["error"] = traceback.format_exc()
    finally:
        STATE["running"] = False


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=STATIC, **k)

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                self.path = "/index.html"
                return super().do_GET()
            if u.path == "/api/boards":
                files = []
                for d in BOARD_DIRS:
                    files += sorted(glob.glob(os.path.join(d, "*_geometry.json")))
                return self._json({"files": [{"name": os.path.basename(f), "path": f} for f in files], "dirs": BOARD_DIRS,
                                   "kicad_python": kexport.kicad_python()})
            if u.path == "/api/board":
                with LOCK:
                    return self._json(board_json(load_board(q["file"][0], (q.get("ref") or [None])[0])))
            if u.path == "/api/setup":
                if STATE["setup"] is None:
                    return self._json({"error": "no setup"}, 404)
                return self._json(STATE["setup"])
            if u.path == "/api/status":
                sim = STATE["sim"]
                st = {"running": STATE["running"], "error": STATE["error"], "snapshots": len(STATE["snaps"]), "snap_z": STATE["snap_z"],
                      "nsteps": STATE["nsteps"], "log": STATE["log"][-6:], "has_result": STATE["result"] is not None, "has_setup": STATE["setup"] is not None}
                if sim is not None and hasattr(sim, "step_now") and STATE["setup"]:
                    el = time.time() - sim.t_start if STATE["running"] else getattr(sim, "wall", 0)
                    st.update({"step": sim.step_now, "energy_db": sim.energy_db, "elapsed": el,
                               "speed": (sim.step_now * STATE["setup"]["cells"] / el / 1e6) if el > 0 else 0})
                return self._json(st)
            if u.path == "/api/snap":
                s = STATE["snaps"][int(q["i"][0])]
                return self._json({"n": s["n"], "t": s["t"], "shape": list(s["data"].shape), "data": b64(s["data"]), "z": STATE["snap_z"]})
            if u.path == "/api/result":
                return self._json(STATE["result"] or {"error": "no result"})
            return super().do_GET()
        except Exception:
            return self._json({"error": traceback.format_exc()}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        try:
            if u.path == "/api/export":
                pcb = req["pcb"].strip().strip('"')
                if not os.path.isfile(pcb):
                    return self._json({"error": "no such file: %s" % pcb}, 400)
                os.makedirs(BOARD_DIRS[0], exist_ok=True)
                out = os.path.join(BOARD_DIRS[0], os.path.splitext(os.path.basename(pcb))[0] + "_geometry.json")
                kexport.export(pcb, out)
                return self._json({"name": os.path.basename(out)})
            if u.path == "/api/setup":
                if STATE["running"]:
                    return self._json({"error": "a run is in progress"}, 409)
                with LOCK:
                    return self._json(do_setup(req))
            if u.path == "/api/run":
                if STATE["running"]:
                    return self._json({"error": "already running"}, 409)
                if STATE["setup"] is None:
                    return self._json({"error": "setup first"}, 400)
                STATE.update({"running": True, "stop": False, "error": None, "snaps": [], "result": None, "log": [], "sim": None})
                threading.Thread(target=run_thread, args=(req,), daemon=True).start()
                return self._json({"ok": True})
            if u.path == "/api/stop":
                STATE["stop"] = True
                return self._json({"ok": True})
            return self._json({"error": "unknown endpoint"}, 404)
        except Exception:
            return self._json({"error": traceback.format_exc()}, 500)


def main(argv=None):
    global BOARD_DIRS
    ap = argparse.ArgumentParser(prog="kicad-fdtd gui")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--boards", nargs="*", default=["out"], help="directories with *_geometry.json (first one receives exports)")
    a = ap.parse_args(argv)
    BOARD_DIRS = [os.path.abspath(d) for d in a.boards]
    os.makedirs(BOARD_DIRS[0], exist_ok=True)
    print("boards from", BOARD_DIRS, " KiCad python:", kexport.kicad_python())
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print("kicad_fdtd GUI on http://127.0.0.1:%d" % a.port, flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
